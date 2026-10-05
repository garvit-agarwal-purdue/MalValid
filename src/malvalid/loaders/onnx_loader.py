"""ONNX loader (``model_kind: onnx``).

``.onnx`` graphs are a safe, non-executable format: they are scored with **onnxruntime** on the
CPU execution provider (no custom-op libraries are registered), and never unpickled. Requires the
``onnx`` extra (``onnxruntime``; ``onnx`` for tree access).

Scoring (:class:`ONNXModel`): the graph must take one ``(n, d)`` float/double tensor. The
malicious-class probability is picked from the outputs without running the model:

* an output named ``*prob*`` is preferred, then a ``seq(map(...))`` (skl2onnx/onnxmltools ZipMap)
  output, then the first float tensor output; label outputs (int64/string tensors) are ignored;
* ZipMap rows use the key for class 1 (``1``, ``"1"``, ``True``, ``"malicious"``/``"malware"``,
  else the larger of two keys); ``(n, 2)`` tensors use the column of class 1 (from the graph's
  ``classlabels_*`` when present, else column 1); ``(n,)``/``(n, 1)`` tensors are used as-is;
* the result must lie in [0, 1] — a logit/score output raises a clear error instead of being
  silently treated as a probability. Override with ``ONNXModel(path, output=..., positive_class=...)``.

onnxruntime ignores ``OMP_NUM_THREADS``; the session's intra-op thread count is set explicitly
from the sandbox's thread budget (``MALVALID_THREADS``/``OMP_NUM_THREADS``, default <= 4).

Tree access: a single ``ai.onnx.ml`` ``TreeEnsembleClassifier`` / ``TreeEnsembleRegressor`` node that
reads the graph input directly (``Cast``/``Identity`` in between are fine) is converted:

* ``BRANCH_LEQ``/``BRANCH_LT``/``BRANCH_GTE``/``BRANCH_GT`` become ``x <= t`` splits (strict
  comparisons via ``nextafter``; ``GT``/``GTE`` swap the children); ``BRANCH_EQ``/``NEQ``/``MEMBER``
  are unsupported. NaN follows ``nodes_missing_value_tracks_true``.
* leaf weights per class and ``base_values`` are combined into one raw score, and the output
  transform (logistic / identity, incl. onnxruntime's binary-classifier conventions and 2-class
  softmax) is **identified empirically**: each candidate interpretation is scored against the
  session's own probabilities on probe rows built from the graph's split values (exact ties, both
  float32 neighbours, NaN). If none reproduces the graph (e.g. extra post-processing ops), tree
  access raises :class:`UnsupportedTreeError` and M5/M7 degrade to black-box methods.
* ONNX graphs carry no training cover; leaves get a uniform cover of 1 (``meta["cover"] ==
  "uniform"``), which keeps TreeSHAP additive but makes cover-based heuristics uninformative.
"""

from __future__ import annotations

import logging
import os
import weakref
from collections import OrderedDict
from pathlib import Path
from typing import Any, ClassVar

import numpy as np

from malvalid.core import UnsupportedTreeError
from malvalid.loaders.base import ModelLoader
from malvalid.loaders.lightgbm_loader import (
    as_float_matrix,
    build_tree,
    describe_type,
    env_threads,
    probe_matrix,
)
from malvalid.loaders.trees import Tree, TreeEnsemble, strict_lt_to_le

log = logging.getLogger("malvalid.loaders")

_POSITIVE_KEYS: tuple[Any, ...] = (1, "1", True, "malicious", "malware", "1.0")
_FLOAT_TENSORS = {"tensor(float)": np.float32, "tensor(double)": np.float64}
_TREE_OPS = ("TreeEnsembleClassifier", "TreeEnsembleRegressor")


def _default_threads() -> int:
    return env_threads() or max(1, min(4, os.cpu_count() or 1))


def _make_session(source: str | bytes, threads: int | None) -> Any:
    import onnxruntime as ort

    so = ort.SessionOptions()
    so.intra_op_num_threads = int(threads or _default_threads())
    so.inter_op_num_threads = 1
    so.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
    so.log_severity_level = 3  # errors only; ORT's info/warning chatter goes nowhere useful
    return ort.InferenceSession(source, sess_options=so, providers=["CPUExecutionProvider"])


def _class_labels(proto: Any) -> list[Any] | None:
    """Class labels declared by a ZipMap / TreeEnsembleClassifier / other ai.onnx.ml classifier."""
    from onnx import helper

    for node in proto.graph.node:
        for a in node.attribute:
            if a.name in ("classlabels_int64s", "classlabels_ints"):
                return [int(v) for v in helper.get_attribute_value(a)]
            if a.name == "classlabels_strings":
                return [v.decode() if isinstance(v, bytes) else str(v) for v in helper.get_attribute_value(a)]
    return None


class ONNXModel:
    """An ONNX detector: an onnxruntime session plus the choice of its malicious-class output.

    ``source`` is a path, serialized model bytes, an ``onnx.ModelProto`` or an existing
    ``onnxruntime.InferenceSession``. ``output`` pins the output name, ``positive_class`` the
    ZipMap key / class label of the malicious class.
    """

    def __init__(
        self,
        source: str | Path | bytes | Any,
        *,
        output: str | None = None,
        positive_class: Any = None,
        threads: int | None = None,
    ) -> None:
        self.path: Path | None = None
        self._bytes: bytes | None = None
        self._proto: Any = None
        self.positive_class = positive_class
        self._threads = threads
        self._margin: Any = None  # float64-sigmoid scoring path: None = untried, False = unavailable/off
        if isinstance(source, (str, Path)):
            self.path = Path(source)
            self.session = _make_session(str(self.path), threads)
        elif isinstance(source, (bytes, bytearray)):
            self._bytes = bytes(source)
            self.session = _make_session(self._bytes, threads)
        elif type(source).__name__ == "ModelProto":
            self._proto = source
            self._bytes = source.SerializeToString()
            self.session = _make_session(self._bytes, threads)
        elif callable(getattr(source, "get_inputs", None)) and callable(getattr(source, "run", None)):
            self.session = source
            mp = getattr(source, "_model_path", None)
            mb = getattr(source, "_model_bytes", None)
            self.path = Path(mp) if mp else None
            self._bytes = bytes(mb) if mb else None
        else:
            raise TypeError(f"cannot build an ONNX model from a {describe_type(source)}")

        inputs = self.session.get_inputs()
        if len(inputs) != 1:
            raise ValueError(
                f"ONNX model has {len(inputs)} inputs ({[i.name for i in inputs]}); malvalid feeds a single "
                "(n, d) feature tensor — export the model with one input or wrap it in an adapter"
            )
        inp = inputs[0]
        if inp.type not in _FLOAT_TENSORS:
            raise ValueError(f"ONNX model input {inp.name!r} has type {inp.type}; expected tensor(float) or tensor(double)")
        self.input_name: str = inp.name
        self.input_dtype = _FLOAT_TENSORS[inp.type]
        shape = list(inp.shape or [])
        self.n_features: int | None = int(shape[-1]) if len(shape) == 2 and isinstance(shape[-1], int) else None

        self.output_name, self.output_type = self._select_output(output)
        self._labels = self._graph_class_labels()
        self._map_key: Any = None
        self._pos_idx: int | None = None
        self._margin_labels: list[Any] | None = None

    # ---- output selection ----------------------------------------------------------------------

    def _select_output(self, output: str | None) -> tuple[str, str]:
        outs = self.session.get_outputs()
        if output is not None:
            for o in outs:
                if o.name == output:
                    return o.name, o.type
            raise ValueError(f"ONNX model has no output {output!r}; outputs: {[o.name for o in outs]}")
        cands = [o for o in outs if o.type.startswith("seq(map(") or o.type in _FLOAT_TENSORS]
        if not cands:
            raise ValueError(
                f"ONNX model has no probability-like output (outputs: {[(o.name, o.type) for o in outs]}); "
                "export class probabilities or wrap the model in an adapter"
            )
        named = [o for o in cands if "prob" in o.name.lower()]
        maps = [o for o in cands if o.type.startswith("seq(map(")]
        pick = (named or maps or cands)[0]
        if len(cands) > 1 and not named:
            log.info("ONNX model has several float outputs %s; scoring with %r", [o.name for o in cands], pick.name)
        return pick.name, pick.type

    def _graph_class_labels(self) -> list[Any] | None:
        if self.output_type.startswith("seq(map(") and self.positive_class is None:
            return None  # map keys are the labels
        try:
            proto = self.model_proto()
        except Exception:  # noqa: BLE001 - onnx not installed or proto unavailable: use defaults
            return None
        return _class_labels(proto)

    def _positive_index(self, k: int) -> int:
        if self._pos_idx is not None:
            return self._pos_idx
        target = self.positive_class
        labels = self._labels
        idx: int | None = None
        if labels is not None and len(labels) == k:
            for cand in ((target,) if target is not None else _POSITIVE_KEYS):
                if cand in labels:
                    idx = labels.index(cand)
                    break
        if idx is None and target is not None and isinstance(target, int) and 0 <= target < k:
            idx = int(target)
        if idx is None:
            if target is not None:
                raise ValueError(
                    f"ONNX output {self.output_name!r} has no class {target!r} (class labels {labels})"
                )
            idx = k - 1
            if labels is not None:  # labels exist but none is recognisably "malicious"
                log.warning(
                    "ONNX output %r: none of the class labels %s is recognisably malicious; using the last "
                    "column (%r), as scikit-learn does for sorted classes — pass positive_class to override",
                    self.output_name, labels, labels[idx] if len(labels) == k else idx,
                )
        self._pos_idx = idx
        return idx

    def _pick_key(self, row: dict[Any, Any]) -> Any:
        keys = list(row.keys())
        if self.positive_class is not None:
            if self.positive_class not in row:
                raise ValueError(f"ONNX output {self.output_name!r} has no class {self.positive_class!r}; keys {keys}")
            return self.positive_class
        for k in _POSITIVE_KEYS:
            if k in row:
                return k
        if len(keys) == 2:
            key = sorted(keys, key=str)[-1]
            log.warning(
                "ONNX output %r: none of the classes %s is recognisably malicious; treating the larger "
                "key %r as malicious (scikit-learn's sorted-classes convention) — pass positive_class "
                "to override",
                self.output_name, keys, key,
            )
            return key
        raise ValueError(f"cannot tell which of the ONNX output's classes {keys} is malicious; pass positive_class")

    # ---- scoring -------------------------------------------------------------------------------

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        X = as_float_matrix(X)
        n = X.shape[0]
        if self.n_features is not None and X.shape[1] != self.n_features:
            raise ValueError(f"X has {X.shape[1]} features; the ONNX model expects {self.n_features}")
        (out,) = self.session.run([self.output_name], {self.input_name: X.astype(self.input_dtype, copy=False)})
        if isinstance(out, list):  # ZipMap: list of {class: prob}
            if not out:
                p = np.zeros(0)
            else:
                if self._map_key is None:
                    self._map_key = self._pick_key(out[0])
                p = np.fromiter((row[self._map_key] for row in out), dtype=np.float64, count=len(out))
        else:
            p = np.asarray(out, dtype=np.float64)
            if p.ndim == 2 and p.shape[1] > 1:
                if p.shape[1] != 2:
                    raise ValueError(
                        f"ONNX output {self.output_name!r} has {p.shape[1]} classes; malvalid evaluates binary "
                        "detectors — wrap the model in an adapter that returns one malicious score"
                    )
                p = p[:, self._positive_index(2)]
            elif p.ndim == 2:
                p = p[:, 0]
        if p.shape != (n,):
            raise ValueError(f"ONNX output {self.output_name!r} has shape {p.shape} for {n} rows")
        if n and self._margin is not False:
            p = self._refine_with_margin(X, p)
        if p.size and (not np.all(np.isfinite(p)) or p.min() < -1e-6 or p.max() > 1 + 1e-6):
            raise ValueError(
                f"ONNX output {self.output_name!r} is not a probability (range [{np.nanmin(p):.4g}, "
                f"{np.nanmax(p):.4g}]); export class probabilities (e.g. skl2onnx options={{'zipmap': False}}) "
                "or pass ONNXModel(output=...) / wrap the model in an adapter"
            )
        return np.clip(p, 0.0, 1.0)

    # ---- float64 tail resolution -----------------------------------------------------------------

    def _build_margin_session(self) -> Any:
        """A session over the same graph whose tree classifier skips its LOGISTIC post-transform.

        onnxruntime applies the logistic in float32, so margins above ~16.6 all come out as exactly 1.0
        (and tiny probabilities lose resolution). Only a single binary ``TreeEnsembleClassifier`` with
        ``post_transform == LOGISTIC`` feeding the chosen output (directly or through ZipMap) is handled;
        for it, ``post_transform = NONE`` yields ``[-margin, margin]`` and ``sigmoid(margin)`` can be taken
        in float64. Anything else returns None and the graph's own probabilities are used.
        """
        from onnx import TensorProto, helper

        proto = self.model_proto()
        producers = {o: nd for nd in proto.graph.node for o in nd.output}
        node = producers.get(self.output_name)
        labels = None
        while node is not None and node.op_type == "Identity":  # skl2onnx: tree node -> Identity -> output
            node = producers.get(node.input[0])
        if node is not None and node.op_type == "ZipMap":
            labels = list(_attrs(node).get("classlabels_int64s") or _attrs(node).get("classlabels_strings") or [])
            node = producers.get(node.input[0])
            while node is not None and node.op_type == "Identity":
                node = producers.get(node.input[0])
        if node is None or node.op_type != "TreeEnsembleClassifier" or node.domain not in ("ai.onnx.ml", ""):
            return None
        a = _attrs(node)
        if a.get("post_transform") not in (b"LOGISTIC", "LOGISTIC"):
            return None
        cls = list(a.get("classlabels_int64s") or a.get("classlabels_strings") or [])
        if len(cls) != 2 or len(node.output) < 2:
            return None
        new = type(proto)()
        new.CopyFrom(proto)
        for nd in new.graph.node:
            if nd.output and nd.output[:] == node.output[:]:
                for at in nd.attribute:
                    if at.name == "post_transform":
                        at.s = b"NONE"
        del new.graph.output[:]
        new.graph.output.append(helper.make_tensor_value_info(node.output[1], TensorProto.FLOAT, [None, 2]))
        self._margin_labels = [x.decode() if isinstance(x, bytes) else x for x in (labels or cls)]
        return _make_session(new.SerializeToString(), self._threads)

    def _margin_positive_col(self, p_ref: np.ndarray) -> int | None:
        """Which of the two margin columns is the malicious class (-s, s)."""
        if self.output_type.startswith("seq(map("):
            key = self._map_key
            labs = self._margin_labels or []
            for i, lab in enumerate(labs):
                if lab == key or str(lab) == str(key):
                    return i
            return None
        return self._positive_index(2)

    def _refine_with_margin(self, X: np.ndarray, p_graph: np.ndarray) -> np.ndarray:
        """Float64 sigmoid of the graph's own margin; falls back (permanently) if it cannot be verified."""
        try:
            if self._margin is None:
                self._margin = self._build_margin_session() or False
                if self._margin is False:
                    return p_graph
                self._margin_col = self._margin_positive_col(p_graph)
                if self._margin_col is None:
                    self._margin = False
                    return p_graph
                self._margin_verified = False
            m = self._margin.run(None, {self.input_name: X.astype(self.input_dtype, copy=False)})[0]
            m = np.asarray(m, dtype=np.float64)
            if m.shape != (X.shape[0], 2):
                raise ValueError(f"margin output has shape {m.shape}")
            from scipy.special import expit

            p = expit(m[:, self._margin_col])
            if float(np.max(np.abs(p - p_graph))) > 1e-5 or not np.all(np.isfinite(p)):
                raise ValueError("margin sigmoid disagrees with the graph's probabilities")
            return p
        except Exception as e:  # never let the refinement break scoring
            log.warning("ONNX float64 tail resolution unavailable (%s); using the graph's float32 probabilities "
                        "(scores above ~0.99999994 will tie at 1.0)", e)
            self._margin = False
            return p_graph

    def model_proto(self) -> Any:
        """The graph as an ``onnx.ModelProto`` (needs the ``onnx`` package; external data not loaded)."""
        if self._proto is None:
            import onnx

            if self._bytes is not None:
                self._proto = onnx.load_model_from_string(self._bytes)
            elif self.path is not None:
                self._proto = onnx.load_model(str(self.path), load_external_data=False)
            else:
                raise UnsupportedTreeError(
                    "the onnxruntime session does not expose its model; pass the .onnx path or use "
                    "malvalid.loaders.onnx_loader.ONNXModel"
                )
        return self._proto

    def info(self) -> dict[str, Any]:
        return {
            "path": str(self.path) if self.path else None,
            "input": self.input_name,
            "input_dtype": np.dtype(self.input_dtype).name,
            "n_features": self.n_features,
            "output": self.output_name,
            "output_type": self.output_type,
        }


# ---- native-object coercion (small caches so repeated predict calls reuse sessions) --------------

_session_cache: "weakref.WeakKeyDictionary[Any, ONNXModel]" = weakref.WeakKeyDictionary()
_path_cache: "OrderedDict[tuple[str, int, int], ONNXModel]" = OrderedDict()


def as_onnx_model(native: Any) -> ONNXModel:
    if isinstance(native, ONNXModel):
        return native
    if isinstance(native, (str, Path)):
        p = Path(native).resolve()
        st = p.stat()
        key = (str(p), st.st_mtime_ns, st.st_size)
        if key not in _path_cache:
            _path_cache[key] = ONNXModel(p)
            while len(_path_cache) > 4:
                _path_cache.popitem(last=False)
        return _path_cache[key]
    if type(native).__name__ == "ModelProto":
        return ONNXModel(native)
    try:
        cached = _session_cache.get(native)
    except TypeError:  # not weak-referenceable
        return ONNXModel(native)
    if cached is None:
        cached = ONNXModel(native)
        _session_cache[native] = cached
    return cached


# ---- ai.onnx.ml tree ensembles -> TreeEnsemble ----------------------------------------------------


def _attrs(node: Any) -> dict[str, Any]:
    from onnx import helper

    return {a.name: helper.get_attribute_value(a) for a in node.attribute}


def _floats(attrs: dict[str, Any], name: str) -> tuple[np.ndarray, bool]:
    """(values as float64, stored_as_double) for an attribute with an optional ``*_as_tensor`` twin."""
    from onnx import numpy_helper

    t = attrs.get(f"{name}_as_tensor")
    if t is not None:
        return numpy_helper.to_array(t).astype(np.float64).ravel(), True
    return np.asarray(attrs.get(name, []) or [], dtype=np.float64), False


def _s(v: Any) -> str:
    return v.decode() if isinstance(v, bytes) else str(v)


def _find_tree_node(proto: Any, input_name: str) -> Any:
    g = proto.graph
    nodes = [n for n in g.node if n.op_type in _TREE_OPS and n.domain == "ai.onnx.ml"]
    if not nodes:
        if any(n.op_type == "TreeEnsemble" for n in g.node):
            raise UnsupportedTreeError(
                "the ai.onnx.ml opset-5 TreeEnsemble operator is not supported yet; re-export with "
                "target_opset={'ai.onnx.ml': 3} (TreeEnsembleClassifier/Regressor)"
            )
        raise UnsupportedTreeError(
            "ONNX graph has no TreeEnsembleClassifier/Regressor operator (e.g. a neural network); "
            "tree-based analyses fall back to model-agnostic methods"
        )
    if len(nodes) > 1:
        raise UnsupportedTreeError(f"ONNX graph has {len(nodes)} tree-ensemble operators; only one is supported")
    node = nodes[0]
    producers = {o: n for n in g.node for o in n.output}
    src = node.input[0]
    while src in producers:
        p = producers[src]
        if p.op_type not in ("Cast", "Identity"):
            raise UnsupportedTreeError(
                f"the ONNX tree ensemble reads the output of a {p.op_type!r} op, not the raw feature vector"
            )
        src = p.input[0]
    if src != input_name:
        raise UnsupportedTreeError("the ONNX tree ensemble does not read the model's feature input")
    return node


_PATH_OPS = ("Identity", "Cast", "Mul", "Div", "Sigmoid", "ZipMap")


def _scalar_const(proto: Any, name: str) -> float | None:
    from onnx import numpy_helper

    for init in proto.graph.initializer:
        if init.name == name:
            arr = numpy_helper.to_array(init)
            return float(arr.ravel()[0]) if arr.size == 1 else None
    for n in proto.graph.node:
        if n.op_type == "Constant" and name in n.output:
            for a in n.attribute:
                if a.name == "value":
                    arr = numpy_helper.to_array(a.t)
                    return float(arr.ravel()[0]) if arr.size == 1 else None
                if a.name == "value_float":
                    return float(a.f)
    return None


def _output_path_scale(proto: Any, node: Any, output_name: str) -> float:
    """Product of scalar Mul/Div factors between the tree op's score output and ``output_name``
    (e.g. skl2onnx's exponential-loss GBC: scores -> Mul(2) -> Sigmoid). 1.0 if none/unknown."""
    consumers: dict[str, list[Any]] = {}
    for n in proto.graph.node:
        for i in n.input:
            consumers.setdefault(i, []).append(n)
    starts = list(node.output[1:] if node.op_type == "TreeEnsembleClassifier" else node.output[:1])
    stack: list[tuple[str, float]] = [(o, 1.0) for o in starts]
    seen: set[str] = set()
    while stack:
        name, scale = stack.pop()
        if name == output_name:
            return scale
        if name in seen:
            continue
        seen.add(name)
        for c in consumers.get(name, []):
            if c.op_type not in _PATH_OPS:
                continue
            k = scale
            if c.op_type in ("Mul", "Div"):
                other = [i for i in c.input if i != name]
                v = _scalar_const(proto, other[0]) if len(other) == 1 else None
                if v is None or v == 0.0:
                    continue
                k = scale * v if c.op_type == "Mul" else scale / v
            stack.extend((o, k) for o in c.output)
    return 1.0


def _parse_structure(attrs: dict[str, Any], double_compare: bool) -> dict[str, Any]:
    tree_ids = np.asarray(attrs.get("nodes_treeids", []), dtype=np.int64)
    node_ids = np.asarray(attrs.get("nodes_nodeids", []), dtype=np.int64)
    feats = np.asarray(attrs.get("nodes_featureids", []), dtype=np.int64)
    modes = [_s(m) for m in attrs.get("nodes_modes", [])]
    true_ids = np.asarray(attrs.get("nodes_truenodeids", []), dtype=np.int64)
    false_ids = np.asarray(attrs.get("nodes_falsenodeids", []), dtype=np.int64)
    values, values_double = _floats(attrs, "nodes_values")
    n = tree_ids.shape[0]
    if n == 0 or not (node_ids.shape[0] == feats.shape[0] == len(modes) == true_ids.shape[0] == false_ids.shape[0] == n):
        raise UnsupportedTreeError("ONNX tree ensemble has inconsistent nodes_* attributes")
    if values.shape[0] != n:
        raise UnsupportedTreeError("ONNX tree ensemble nodes_values length does not match the node count")
    tracks = np.asarray(attrs.get("nodes_missing_value_tracks_true", []) or np.zeros(n), dtype=np.int64)
    if tracks.shape[0] != n:
        tracks = np.zeros(n, dtype=np.int64)
    bad = sorted({m for m in modes if m not in ("LEAF", "BRANCH_LEQ", "BRANCH_LT", "BRANCH_GTE", "BRANCH_GT")})
    if bad:
        raise UnsupportedTreeError(f"ONNX tree split modes {bad} are not supported")
    double = values_double or double_compare

    def lt_to_le(v: np.ndarray) -> np.ndarray:
        return np.nextafter(v, -np.inf) if double else strict_lt_to_le(v)

    trees: dict[int, dict[str, Any]] = {}
    for tid in np.unique(tree_ids):
        idx = np.flatnonzero(tree_ids == tid)
        local = {int(node_ids[i]): k for k, i in enumerate(idx)}
        m = idx.shape[0]
        left = np.full(m, -1, dtype=np.int64)
        right = np.full(m, -1, dtype=np.int64)
        thr = np.zeros(m)
        dleft = np.zeros(m, dtype=bool)
        referenced: set[int] = set()
        for k, i in enumerate(idx):
            mode = modes[i]
            if mode == "LEAF":
                continue
            try:
                t_node, f_node = local[int(true_ids[i])], local[int(false_ids[i])]
            except KeyError as e:
                raise UnsupportedTreeError(f"ONNX tree {tid} references a missing node id {e}") from e
            referenced.update((t_node, f_node))
            v = values[i]
            if mode in ("BRANCH_LEQ", "BRANCH_LT"):  # true branch is x <= t / x < t
                left[k], right[k] = t_node, f_node
                thr[k] = v if mode == "BRANCH_LEQ" else float(lt_to_le(np.asarray([v]))[0])
            else:  # BRANCH_GT: x > v true <=> x <= v false;  BRANCH_GTE: x >= v true <=> x < v false
                left[k], right[k] = f_node, t_node
                thr[k] = v if mode == "BRANCH_GT" else float(lt_to_le(np.asarray([v]))[0])
            default_node = t_node if tracks[i] else f_node
            dleft[k] = default_node == left[k]
        roots = [k for k in range(m) if k not in referenced]
        if len(roots) != 1:
            raise UnsupportedTreeError(f"ONNX tree {tid} does not have a single root")
        trees[int(tid)] = {
            "local": local,
            "left": left,
            "right": right,
            "feature": feats[idx],
            "threshold": thr,
            "default_left": dleft,
            "root": roots[0],
            "raw_values": values[idx],
            "is_leaf": np.asarray([modes[i] == "LEAF" for i in idx]),
        }
    return trees


def _leaf_weights(
    attrs: dict[str, Any], trees: dict[int, dict[str, Any]], prefix: str, n_out: int
) -> list[dict[int, np.ndarray]]:
    """Per output id: {tree id: per-node leaf weight array}."""
    tids = np.asarray(attrs.get(f"{prefix}_treeids", []), dtype=np.int64)
    nids = np.asarray(attrs.get(f"{prefix}_nodeids", []), dtype=np.int64)
    oids = np.asarray(attrs.get(f"{prefix}_ids", []), dtype=np.int64)
    w, _ = _floats(attrs, f"{prefix}_weights")
    if not (tids.shape == nids.shape == oids.shape == w.shape):
        raise UnsupportedTreeError(f"ONNX tree ensemble has inconsistent {prefix}_* attributes")
    if oids.size and (oids.min() < 0 or oids.max() >= n_out):
        raise UnsupportedTreeError(f"ONNX tree ensemble has {prefix} ids outside [0, {n_out})")
    out = [{tid: np.zeros(t["left"].shape[0]) for tid, t in trees.items()} for _ in range(n_out)]
    for tid, nid, oid, wt in zip(tids, nids, oids, w):
        t = trees.get(int(tid))
        if t is None or int(nid) not in t["local"]:
            raise UnsupportedTreeError(f"ONNX leaf weight references unknown node {int(tid)}/{int(nid)}")
        out[int(oid)][int(tid)][t["local"][int(nid)]] += wt
    return out


def _make_trees(trees: dict[int, dict[str, Any]], leaf: dict[int, np.ndarray]) -> list[Tree]:
    return [
        build_tree(
            left=t["left"],
            right=t["right"],
            feature=t["feature"],
            threshold=t["threshold"],
            default_left=t["default_left"],
            leaf_value=leaf[tid],
            cover=np.ones(t["left"].shape[0]),
            root=t["root"],
            what=f"ONNX tree {tid}",
        )
        for tid, t in sorted(trees.items())
    ]


def onnx_tree_ensemble(model: ONNXModel) -> TreeEnsemble:
    """Normalize the graph's tree-ensemble operator (verified against onnxruntime's own output)."""
    try:
        proto = model.model_proto()
    except ImportError as e:
        raise UnsupportedTreeError("ONNX tree access needs the 'onnx' package (from the MalValid source folder: pip install -e '.[onnx]')") from e
    node = _find_tree_node(proto, model.input_name)
    attrs = _attrs(node)
    post = _s(attrs.get("post_transform", b"NONE"))
    if post not in ("NONE", "LOGISTIC", "SOFTMAX"):
        raise UnsupportedTreeError(f"ONNX tree ensemble post_transform {post!r} is not supported")
    trees = _parse_structure(attrs, double_compare=model.input_dtype == np.float64)
    base, _ = _floats(attrs, "base_values")
    average = False
    if node.op_type == "TreeEnsembleClassifier":
        labels = attrs.get("classlabels_int64s") or attrs.get("classlabels_strings") or []
        if len(labels) != 2:
            raise UnsupportedTreeError(f"ONNX TreeEnsembleClassifier has {len(labels)} classes; only binary is supported")
        W = _leaf_weights(attrs, trees, "class", 2)
        b = [float(base[0]) if base.size > 0 else 0.0, float(base[1]) if base.size > 1 else 0.0]
        # (c0, c1, const, transform): raw = c0*s0 + c1*s1 + const, s_k = class-k weights + base_k.
        candidates = [
            (0.0, 1.0, 0.0, "logistic"),
            (0.0, 1.0, 0.0, "identity"),
            (1.0, 0.0, 0.0, "logistic"),
            (1.0, 0.0, 0.0, "identity"),
            (-1.0, 0.0, 0.0, "logistic"),
            (-1.0, 0.0, 1.0, "identity"),
            (-1.0, 1.0, 0.0, "logistic"),  # 2-class softmax
        ]
    else:
        n_targets = int(attrs.get("n_targets", 1))
        if n_targets != 1:
            raise UnsupportedTreeError(f"ONNX TreeEnsembleRegressor with {n_targets} targets is not supported")
        agg = _s(attrs.get("aggregate_function", b"SUM"))
        if agg not in ("SUM", "AVERAGE"):
            raise UnsupportedTreeError(f"ONNX TreeEnsembleRegressor aggregate_function {agg!r} is not supported")
        average = agg == "AVERAGE"
        W = _leaf_weights(attrs, trees, "target", 1) + [
            {tid: np.zeros(t["left"].shape[0]) for tid, t in trees.items()}
        ]
        b = [float(base[0]) if base.size > 0 else 0.0, 0.0]
        candidates = [
            (1.0, 0.0, 0.0, "logistic"),
            (1.0, 0.0, 0.0, "identity"),
            (-1.0, 0.0, 0.0, "logistic"),
            (-1.0, 0.0, 1.0, "identity"),
        ]
    if post == "SOFTMAX":
        candidates = [c for c in candidates if c[:2] == (-1.0, 1.0)] + candidates
    path_scale = _output_path_scale(proto, node, model.output_name)
    scales = [1.0] if path_scale == 1.0 else [path_scale, 1.0]

    n_features = int(max(int(t["feature"][~t["is_leaf"]].max(initial=-1)) for t in trees.values()) + 1)
    if model.n_features is not None:
        n_features = max(n_features, model.n_features)
    thresholds: dict[int, list[np.ndarray]] = {}
    for t in trees.values():
        internal = ~t["is_leaf"]
        for f in np.unique(t["feature"][internal]):
            thresholds.setdefault(int(f), []).append(t["raw_values"][internal & (t["feature"] == f)])
    thr = {f: np.unique(np.concatenate(v)) for f, v in thresholds.items()}
    # Two independent probe sets: the fallback fit uses the first, every candidate is checked on both.
    X_fit = probe_matrix(thr, n_features, n_rows=384, seed=7)
    X = np.vstack([X_fit, probe_matrix(thr, n_features, n_rows=384, seed=8)])
    try:
        target = model.predict_proba(X)
    except Exception as e:  # noqa: BLE001 - output isn't a probability etc.
        raise UnsupportedTreeError(f"cannot verify ONNX trees against the model output: {e}") from e

    per_class = [
        TreeEnsemble(trees=_make_trees(trees, W[k]), n_features=n_features, output_transform="identity",
                     average_output=average)
        for k in (0, 1)
    ]
    raw = [pc.predict_raw(X) for pc in per_class]  # identity, base 0: plain (averaged) leaf sums

    # (a0, a1, k, transform, method): raw margin = a0 * s0 + a1 * s1 + k, s_c = class-c leaf sums.
    combos: list[tuple[float, float, float, str, str]] = []
    for scale in scales:
        for c0, c1, const, transform in candidates:
            combos.append((scale * c0, scale * c1, scale * (c0 * b[0] + c1 * b[1]) + const, transform, "attributes"))
    n_fit = X_fit.shape[0]
    combos.extend(_fit_combinations(raw[0][:n_fit], raw[1][:n_fit], target[:n_fit]))

    tol = 1e-5 + 1e-6 * len(trees)
    scored: list[tuple[float, tuple[float, float, float, str, str]]] = []
    for combo in combos:
        a0, a1, k, transform, _ = combo
        r = a0 * raw[0] + a1 * raw[1] + k
        p = 1.0 / (1.0 + np.exp(-r)) if transform == "logistic" else np.clip(r, 0.0, 1.0)
        scored.append((float(np.max(np.abs(p - target))), combo))
    # An interpretation read from the graph's attributes wins whenever it reproduces the output;
    # the least-squares fit is only a fallback (it otherwise just absorbs float32 noise).
    exact = [sc for sc in scored if sc[1][4] == "attributes" and sc[0] <= tol]
    err, (a0, a1, k, transform, method) = min(exact or scored, key=lambda sc: sc[0])
    if err > tol:
        raise UnsupportedTreeError(
            f"the ONNX graph's tree ensemble does not reproduce its output (best max error {err:.3g}); "
            "the graph probably post-processes the scores — tree-based analyses fall back to black-box methods"
        )
    if a0 == 0.0 and a1 == 0.0:
        log.warning(
            "ONNX model output %r does not depend on its tree ensemble (onnxruntime ignores the leaf "
            "weights for this class_ids/base_values layout); tree attributions will all be zero",
            model.output_name,
        )
    elif method == "fitted":
        log.info("ONNX tree score combination identified by fitting: %.6g*s0 + %.6g*s1 + %.6g", a0, a1, k)
    leaf = {tid: a0 * W[0][tid] + a1 * W[1][tid] for tid in trees}
    return TreeEnsemble(
        trees=_make_trees(trees, leaf),
        n_features=n_features,
        base_score=float(k),
        output_transform=transform,
        sigmoid_scale=1.0,
        average_output=average,
        model_kind="onnx",
        feature_names=None,
        meta={
            "loader": "onnx",
            "source": f"ai.onnx.ml.{node.op_type}",
            "post_transform": post,
            "score_combination": {"class0": a0, "class1": a1, "const": k, "method": method},
            "verify_max_abs_diff": err,
            "cover": "uniform",
            "onnx_output": model.output_name,
        },
    )


def _snap(c: float) -> float:
    """Round a fitted coefficient to a multiple of 1/1024 when it is within float noise of one."""
    r = round(c * 1024.0) / 1024.0
    return r if abs(r - c) <= 2e-6 * max(1.0, abs(c)) else c


def _fit_combinations(
    s0: np.ndarray, s1: np.ndarray, p: np.ndarray
) -> list[tuple[float, float, float, str, str]]:
    """Least-squares ``a0*s0 + a1*s1 + k`` in logit space (logistic) and in probability space (identity).

    onnxruntime's binary ``TreeEnsembleClassifier`` combines per-class leaf sums and ``base_values``
    in layout-dependent ways; every one of them is linear in the two sums, so fitting the three
    coefficients on probe rows (then verifying on independent rows) covers layouts the explicit
    candidates miss. Weights ``p(1-p)`` keep float32 rounding near 0/1 from dominating the logit fit.
    """
    out: list[tuple[float, float, float, str, str]] = []
    A = np.column_stack([s0, s1, np.ones_like(s0)])
    for transform in ("logistic", "identity"):
        if transform == "logistic":
            m = (p > 1e-6) & (p < 1.0 - 1e-6)
            if m.sum() < 8:
                continue
            y = np.log(p[m] / (1.0 - p[m]))
            w = p[m] * (1.0 - p[m])
        else:
            m = (p > 1e-9) & (p < 1.0 - 1e-9)
            if m.sum() < 8:
                continue
            y, w = p[m], np.ones(int(m.sum()))
        try:
            coef, *_ = np.linalg.lstsq(A[m] * w[:, None], y * w, rcond=None)
        except np.linalg.LinAlgError:  # pragma: no cover - degenerate design
            continue
        a0, a1 = _snap(float(coef[0])), _snap(float(coef[1]))
        k = float(np.median(y - a0 * s0[m] - a1 * s1[m]))
        out.append((a0, a1, k, transform, "fitted"))
    return out


class ONNXLoader(ModelLoader):
    kind: ClassVar[str] = "onnx"
    extensions: ClassVar[tuple[str, ...]] = (".onnx",)
    safe_extensions: ClassVar[tuple[str, ...]] = (".onnx",)
    description: ClassVar[str] = (
        "ONNX graph scored with onnxruntime (safe format); tree access for ai.onnx.ml TreeEnsemble ops"
    )

    def can_load(self, path: Path) -> bool:
        p = Path(path)
        return p.suffix.lower() in self.extensions and not (p.is_file() and self.is_pickle(p))

    def load(self, path: Path, *, allow_pickle: bool = False) -> ONNXModel:
        p = Path(path)
        if not p.is_file():
            raise FileNotFoundError(f"ONNX model file not found: {p}")
        self.check_pickle_policy(p, allow_pickle)
        if self.is_pickle(p):  # allowed pickles are still not ONNX; never unpickle here
            raise ValueError(f"{p} is a pickle, not an ONNX model; the onnx loader never unpickles")
        try:
            model = ONNXModel(p)
        except ImportError as e:
            raise ImportError("ONNX models need onnxruntime: install the [onnx] extra (from the MalValid source "
                              "folder: pip install -e '.[onnx]')") from e
        log.debug("loaded ONNX model %s: %s", p, model.info())
        return model

    def supports_native(self, obj: Any) -> bool:
        if isinstance(obj, ONNXModel):
            return True
        if isinstance(obj, (str, Path)):
            return Path(obj).suffix.lower() == ".onnx"
        mod = type(obj).__module__.split(".")[0]
        if mod == "onnxruntime":
            return callable(getattr(obj, "run", None)) and callable(getattr(obj, "get_inputs", None))
        return mod == "onnx" and type(obj).__name__ == "ModelProto"

    def predict_proba(self, native: Any, X: np.ndarray) -> np.ndarray:
        return as_onnx_model(native).predict_proba(X)

    def tree_ensemble(self, native: Any) -> TreeEnsemble:
        return onnx_tree_ensemble(as_onnx_model(native))

    def library_version(self) -> str | None:
        try:
            import onnxruntime

            return str(onnxruntime.__version__)
        except ImportError:  # pragma: no cover
            return None
