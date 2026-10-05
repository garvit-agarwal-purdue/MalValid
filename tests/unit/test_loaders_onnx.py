"""ONNX loader: output selection (ZipMap / tensor), ai.onnx.ml tree extraction verified against
onnxruntime (skl2onnx exports and hand-built graphs with every split mode, NaN tracking, ties)."""

from __future__ import annotations

import zlib
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from malvalid.core import UnsupportedTreeError
from malvalid.loaders.onnx_loader import ONNXLoader, ONNXModel, onnx_tree_ensemble
from tests.unit.test_loaders import THREADS, assert_faithful, make_binary_data

onnx = pytest.importorskip("onnx")
ort = pytest.importorskip("onnxruntime")
skl2onnx = pytest.importorskip("skl2onnx")

from onnx import TensorProto, helper  # noqa: E402
from sklearn.ensemble import (  # noqa: E402
    ExtraTreesClassifier,
    GradientBoostingClassifier,
    RandomForestClassifier,
)

OPSET = {"": 17, "ai.onnx.ml": 3}


def _export(est: Any, X: np.ndarray, path: Path, **options: Any) -> Path:
    kw = {"options": {id(est): options}} if options else {}
    onx = skl2onnx.to_onnx(est, X[:1], target_opset=OPSET, **kw)
    path.write_bytes(onx.SerializeToString())
    return path


# --------------------------------------------------------------------------------------------------
# skl2onnx exports
# --------------------------------------------------------------------------------------------------

SKL_CASES = {
    "gbc_zipmap": (lambda: GradientBoostingClassifier(n_estimators=20, max_depth=3, random_state=0), {}),
    "gbc_tensor": (lambda: GradientBoostingClassifier(n_estimators=20, max_depth=3, random_state=0), {"zipmap": False}),
    "gbc_exponential": (
        lambda: GradientBoostingClassifier(n_estimators=15, loss="exponential", max_depth=3, random_state=0),
        {"zipmap": False},
    ),
    "random_forest": (
        lambda: RandomForestClassifier(n_estimators=10, max_depth=5, random_state=0, n_jobs=THREADS),
        {},
    ),
    "extra_trees": (
        lambda: ExtraTreesClassifier(n_estimators=6, max_depth=5, random_state=0, n_jobs=THREADS),
        {"zipmap": False},
    ),
}


@pytest.mark.parametrize("case", sorted(SKL_CASES))
def test_skl2onnx_tree_models(tmp_path: Path, case: str) -> None:
    make, options = SKL_CASES[case]
    X, y = make_binary_data(600, 8, seed=1)
    est = make().fit(X, y)
    path = _export(est, X, tmp_path / f"{case}.onnx", **options)
    loader = ONNXLoader()
    assert loader.can_load(path)
    model = loader.load(path)
    assert isinstance(model, ONNXModel) and model.n_features == 8
    # the ONNX graph scores like the sklearn model (float32 arithmetic inside onnxruntime)
    np.testing.assert_allclose(loader.predict_proba(model, X), est.predict_proba(X)[:, 1], rtol=0, atol=2e-5)
    te = assert_faithful(loader, model, X, nan=True)  # NaN follows the graph's missing-value tracking
    assert te.model_kind == "onnx" and te.n_trees == len(getattr(est, "estimators_"))
    assert te.meta["cover"] == "uniform"
    if case.startswith("gbc"):
        assert te.output_transform == "logistic"
        # the normalized margins equal sklearn's (up to float32 rounding in the graph)
        factor = 2.0 if "exponential" in case else 1.0
        np.testing.assert_allclose(te.predict_raw(X), factor * est.decision_function(X), rtol=0, atol=1e-4)


def test_double_precision_input(tmp_path: Path) -> None:
    # onnxruntime 1.2x/1.30 rejects double-output TreeEnsembleClassifier/Regressor at ai.onnx.ml
    # opset 3 (skl2onnx's double GBC export fails to load), but a double-input random forest
    # (float probabilities) loads; its splits compare the float64 input directly.
    X, y = make_binary_data(500, 6, seed=2)
    Xd = X.astype(np.float64)
    est = RandomForestClassifier(n_estimators=6, max_depth=5, random_state=0, n_jobs=THREADS).fit(Xd, y)
    onx = skl2onnx.to_onnx(est, Xd[:1], target_opset=OPSET, options={id(est): {"zipmap": False}})
    path = tmp_path / "double.onnx"
    path.write_bytes(onx.SerializeToString())
    loader = ONNXLoader()
    model = loader.load(path)
    assert model.input_dtype == np.float64
    np.testing.assert_allclose(loader.predict_proba(model, Xd), est.predict_proba(Xd)[:, 1], rtol=0, atol=1e-6)
    assert_faithful(loader, model, Xd, nan=True)


def test_string_class_labels_pick_malicious(tmp_path: Path) -> None:
    X, y = make_binary_data(400, 6, seed=3)
    labels = np.where(y == 1, "malicious", "benign")
    est = GradientBoostingClassifier(n_estimators=8, random_state=0).fit(X, labels)
    path = _export(est, X, tmp_path / "labels.onnx")
    loader = ONNXLoader()
    model = loader.load(path)
    assert model.output_type.startswith("seq(map(string")
    np.testing.assert_allclose(loader.predict_proba(model, X), est.predict_proba(X)[:, 1], rtol=0, atol=2e-5)
    assert_faithful(loader, model, X)


@pytest.mark.parametrize("zipmap", [True, False])
def test_unrecognised_class_labels_follow_sklearn_order_with_one_warning(
    tmp_path: Path, zipmap: bool, caplog: pytest.LogCaptureFixture
) -> None:
    X, y = make_binary_data(300, 6, seed=6)
    labels = np.where(y == 1, "goodware", "badware")  # neither name says "malicious"
    est = GradientBoostingClassifier(n_estimators=5, random_state=0).fit(X, labels)
    path = _export(est, X, tmp_path / "odd.onnx", **({} if zipmap else {"zipmap": False}))
    model = ONNXModel(path)
    with caplog.at_level("WARNING", logger="malvalid.loaders"):
        p = model.predict_proba(X)
        model.predict_proba(X[:10])
    # the last sorted class, i.e. sklearn's classes_[1] / predict_proba column 1
    assert list(est.classes_) == ["badware", "goodware"]
    np.testing.assert_allclose(p, est.predict_proba(X)[:, 1], rtol=0, atol=2e-5)
    msgs = [r.getMessage() for r in caplog.records if "recognisably malicious" in r.getMessage()]
    assert len(msgs) == 1 and "goodware" in msgs[0]
    np.testing.assert_allclose(ONNXModel(path, positive_class="badware").predict_proba(X), 1.0 - p, atol=1e-6)
    with pytest.raises(ValueError, match="no class"):
        ONNXModel(path, positive_class="nope").predict_proba(X)


def test_positive_class_and_output_overrides(tmp_path: Path) -> None:
    X, y = make_binary_data(300, 6, seed=4)
    est = GradientBoostingClassifier(n_estimators=5, random_state=0).fit(X, y)
    path = _export(est, X, tmp_path / "gbc.onnx", zipmap=False)
    p1 = ONNXModel(path).predict_proba(X)
    np.testing.assert_allclose(ONNXModel(path, positive_class=0).predict_proba(X), 1.0 - p1, rtol=0, atol=1e-6)
    np.testing.assert_allclose(ONNXModel(path, output="probabilities").predict_proba(X), p1)
    with pytest.raises(ValueError, match="no output 'nope'"):
        ONNXModel(path, output="nope")
    with pytest.raises(ValueError, match="features"):
        ONNXModel(path).predict_proba(X[:, :3])


def test_native_forms_are_equivalent(tmp_path: Path) -> None:
    X, y = make_binary_data(300, 6, seed=5)
    est = GradientBoostingClassifier(n_estimators=6, random_state=0).fit(X, y)
    path = _export(est, X, tmp_path / "gbc.onnx")
    loader = ONNXLoader()
    ref = loader.predict_proba(loader.load(path), X)
    sess = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    proto = onnx.load_model(str(path))
    for native in (sess, proto, str(path), path, ONNXModel(path.read_bytes())):
        assert loader.supports_native(native)
        np.testing.assert_array_equal(loader.predict_proba(native, X), ref)
        assert loader.tree_ensemble(native).n_trees == 6


def test_non_tree_model_scores_but_has_no_trees(tmp_path: Path) -> None:
    from sklearn.linear_model import LogisticRegression

    X, y = make_binary_data(300, 6, seed=6)
    est = LogisticRegression().fit(X, y)
    path = _export(est, X, tmp_path / "lr.onnx", zipmap=False)
    loader = ONNXLoader()
    model = loader.load(path)
    np.testing.assert_allclose(loader.predict_proba(model, X), est.predict_proba(X)[:, 1], rtol=0, atol=1e-5)
    with pytest.raises(UnsupportedTreeError, match="no TreeEnsembleClassifier"):
        loader.tree_ensemble(model)


def test_trees_behind_preprocessing_are_unsupported(tmp_path: Path) -> None:
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    X, y = make_binary_data(300, 6, seed=7)
    pipe = make_pipeline(StandardScaler(), GradientBoostingClassifier(n_estimators=5, random_state=0)).fit(X, y)
    path = tmp_path / "pipe.onnx"
    path.write_bytes(skl2onnx.to_onnx(pipe, X[:1], target_opset=OPSET).SerializeToString())
    loader = ONNXLoader()
    model = loader.load(path)
    np.testing.assert_allclose(loader.predict_proba(model, X), pipe.predict_proba(X)[:, 1], rtol=0, atol=2e-5)
    with pytest.raises(UnsupportedTreeError, match="Scaler"):
        loader.tree_ensemble(model)


def test_multiclass_graph_is_rejected(tmp_path: Path) -> None:
    X, _ = make_binary_data(300, 6, seed=8)
    y3 = np.random.default_rng(0).integers(0, 3, X.shape[0])
    est = GradientBoostingClassifier(n_estimators=3, random_state=0).fit(X, y3)
    path = _export(est, X, tmp_path / "multi.onnx", zipmap=False)
    loader = ONNXLoader()
    model = loader.load(path)
    with pytest.raises(ValueError, match="3 classes"):
        loader.predict_proba(model, X)
    with pytest.raises(UnsupportedTreeError, match="binary"):
        loader.tree_ensemble(model)


# --------------------------------------------------------------------------------------------------
# hand-built ai.onnx.ml graphs: every split mode, missing-value tracking, shuffled node ids
# --------------------------------------------------------------------------------------------------

MODES = ("BRANCH_LEQ", "BRANCH_LT", "BRANCH_GT", "BRANCH_GTE")


def random_tree_nodes(rng: np.random.Generator, n_trees: int, d: int, max_depth: int = 4) -> list[tuple]:
    """Rows ``(tree, node, feature, threshold, mode, true_id, false_id, nan_tracks_true, leaf_value)``.

    Thresholds come from a coarse float32 grid so probe rows hit exact ties; node ids are a shuffled
    permutation of ``0..m-1`` (the root is listed first, as onnxruntime requires)."""
    grid = np.linspace(-1.5, 1.5, 13).astype(np.float32)
    rows: list[tuple] = []
    for t in range(n_trees):
        nodes: list[list[Any]] = []

        def build(depth: int) -> int:
            nid = len(nodes)
            if depth >= max_depth or (depth > 0 and rng.random() < 0.25):
                nodes.append([t, nid, 0, 0.0, "LEAF", 0, 0, 0, float(rng.normal())])
                return nid
            nodes.append([])
            a, b = build(depth + 1), build(depth + 1)
            nodes[nid] = [t, nid, int(rng.integers(d)), float(rng.choice(grid)), str(rng.choice(MODES)), a, b,
                          int(rng.integers(2)), 0.0]
            return nid

        build(0)
        perm = rng.permutation(len(nodes))
        for r in nodes:
            r[1] = int(perm[r[1]])
            if r[4] != "LEAF":
                r[5], r[6] = int(perm[r[5]]), int(perm[r[6]])
        order = [0] + [int(i) + 1 for i in rng.permutation(len(nodes) - 1)]
        rows.extend(tuple(nodes[i]) for i in order)
    return rows


def tree_graph(
    rows: list[tuple],
    d: int,
    *,
    op: str = "TreeEnsembleClassifier",
    post: str = "LOGISTIC",
    base: tuple[float, ...] = (),
    class_mode: str = "class0",
    aggregate: str = "SUM",
    sigmoid_after: bool = False,
    labels: tuple[Any, ...] = (0, 1),
    zipmap: bool = False,
    double: bool = False,
) -> Any:
    in_type = TensorProto.DOUBLE if double else TensorProto.FLOAT
    attrs: dict[str, Any] = dict(
        nodes_treeids=[r[0] for r in rows],
        nodes_nodeids=[r[1] for r in rows],
        nodes_featureids=[r[2] for r in rows],
        nodes_values=[r[3] for r in rows],
        nodes_modes=[r[4] for r in rows],
        nodes_truenodeids=[r[5] for r in rows],
        nodes_falsenodeids=[r[6] for r in rows],
        nodes_missing_value_tracks_true=[r[7] for r in rows],
        post_transform=post,
    )
    if base:
        attrs["base_values"] = list(base)
    leaves = [r for r in rows if r[4] == "LEAF"]
    X = helper.make_tensor_value_info("X", in_type, [None, d])
    nodes = []
    if op == "TreeEnsembleRegressor":
        attrs.update(
            n_targets=1,
            target_ids=[0] * len(leaves),
            target_treeids=[r[0] for r in leaves],
            target_nodeids=[r[1] for r in leaves],
            target_weights=[r[8] for r in leaves],
            aggregate_function=aggregate,
        )
        score = "scores" if sigmoid_after else "variable"
        nodes.append(helper.make_node(op, ["X"], [score], domain="ai.onnx.ml", **attrs))
        if sigmoid_after:
            nodes.append(helper.make_node("Sigmoid", ["scores"], ["variable"]))
        outs = [helper.make_tensor_value_info("variable", TensorProto.FLOAT, [None, 1])]
    else:
        tids, nids, w = [r[0] for r in leaves], [r[1] for r in leaves], [r[8] for r in leaves]
        if class_mode == "both":
            # independent per-class leaf weights (collinear ones would make any fit "exact")
            w1 = np.random.default_rng(len(leaves)).normal(size=len(leaves)).tolist()
            ids = [0] * len(leaves) + [1] * len(leaves)
            tids, nids, w = tids * 2, nids * 2, [-0.5 * v for v in w] + w1
        else:
            ids = [0 if class_mode == "class0" else 1] * len(leaves)
        attrs.update(class_ids=ids, class_treeids=tids, class_nodeids=nids, class_weights=w)
        string_labels = isinstance(labels[0], str)
        attrs["classlabels_strings" if string_labels else "classlabels_int64s"] = list(labels)
        nodes.append(helper.make_node(op, ["X"], ["label", "probabilities"], domain="ai.onnx.ml", **attrs))
        label_type = TensorProto.STRING if string_labels else TensorProto.INT64
        outs = [
            helper.make_tensor_value_info("label", label_type, [None]),
            helper.make_tensor_value_info("probabilities", TensorProto.FLOAT, [None, 2]),
        ]
        if zipmap:
            key = "classlabels_strings" if string_labels else "classlabels_int64s"
            nodes.append(
                helper.make_node("ZipMap", ["probabilities"], ["output_probability"], domain="ai.onnx.ml",
                                 **{key: list(labels)})
            )
            mt = helper.make_map_type_proto(label_type, helper.make_tensor_type_proto(TensorProto.FLOAT, []))
            outs = [outs[0], helper.make_value_info("output_probability", helper.make_sequence_type_proto(mt))]
    graph = helper.make_graph(nodes, "trees", [X], outs)
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid(k, v) for k, v in OPSET.items()])
    model.ir_version = 8
    onnx.checker.check_model(model)
    return model


GRAPH_CASES = {
    # the layout onnxmltools uses for LightGBM / XGBoost binary models
    "class0_logistic": dict(class_mode="class0", base=(0.2,)),
    "class0_logistic_nobase": dict(class_mode="class0"),
    "class1_logistic": dict(class_mode="class1", base=(0.3,)),
    "both_softmax": dict(class_mode="both", post="SOFTMAX", base=(0.1, -0.1)),
    "both_logistic": dict(class_mode="both", post="LOGISTIC", base=(0.1, -0.1)),
    "zipmap_string_labels": dict(class_mode="class0", base=(0.2,), labels=("benign", "malicious"), zipmap=True),
    "double_input": dict(class_mode="class0", base=(-0.4,), double=True),
    "regressor_sigmoid": dict(op="TreeEnsembleRegressor", post="NONE", base=(0.25,), sigmoid_after=True),
    "regressor_avg_sigmoid": dict(op="TreeEnsembleRegressor", post="NONE", aggregate="AVERAGE", sigmoid_after=True),
}


@pytest.mark.parametrize("case", sorted(GRAPH_CASES))
def test_hand_built_tree_graphs(tmp_path: Path, case: str) -> None:
    rng = np.random.default_rng(zlib.crc32(case.encode()))  # stable across processes
    d = 6
    rows = random_tree_nodes(rng, 12, d)
    path = tmp_path / f"{case}.onnx"
    path.write_bytes(tree_graph(rows, d, **GRAPH_CASES[case]).SerializeToString())
    X = rng.normal(size=(300, d)).astype(np.float32)
    X[rng.random(X.shape) < 0.1] = np.nan
    loader = ONNXLoader()
    model = loader.load(path)
    te = assert_faithful(loader, model, X, nan=True)
    assert te.n_trees == 12
    assert te.average_output == (GRAPH_CASES[case].get("aggregate") == "AVERAGE")
    # Standard single-class layouts resolve from the attributes. With weights on *both* classes,
    # onnxruntime 1.30's probability column 1 is sigmoid(s0 + b1) (class-1 weights ignored), which
    # only the verified least-squares fallback recovers; either way the result is exact (above).
    method = te.meta["score_combination"]["method"]
    if GRAPH_CASES[case].get("class_mode") != "both":
        assert method == "attributes"
    else:
        assert method in ("attributes", "fitted")


def test_all_split_modes_and_nan_tracking_are_exercised() -> None:
    rng = np.random.default_rng(11)
    rows = random_tree_nodes(rng, 20, 5)
    modes = {r[4] for r in rows}
    assert set(MODES) <= modes
    assert {r[7] for r in rows if r[4] != "LEAF"} == {0, 1}
    model = ONNXModel(tree_graph(rows, 5, base=(0.1,)).SerializeToString())
    te = onnx_tree_ensemble(model)
    # A row exactly on every threshold of the root split, for each tree, with and without NaN.
    Z = np.zeros((40, 5), dtype=np.float32)
    roots = [r for r in rows if r[1] == [q for q in rows if q[0] == r[0]][0][1]]
    for i, r in enumerate(roots):
        Z[i, r[2]] = r[3]
        Z[20 + i, r[2]] = np.nan
    np.testing.assert_allclose(te.predict_proba(Z), model.predict_proba(Z), rtol=0, atol=1e-6)


def test_raw_score_output_is_refused() -> None:
    rng = np.random.default_rng(12)
    rows = random_tree_nodes(rng, 6, 4)
    model = ONNXModel(tree_graph(rows, 4, op="TreeEnsembleRegressor", post="NONE", base=(0.5,)).SerializeToString())
    X = rng.normal(size=(200, 4)).astype(np.float32)
    with pytest.raises(ValueError, match="not a probability"):
        model.predict_proba(X)
    with pytest.raises(UnsupportedTreeError, match="cannot verify"):
        onnx_tree_ensemble(model)


def test_unsupported_split_mode_is_rejected() -> None:
    rng = np.random.default_rng(13)
    rows = [tuple("BRANCH_EQ" if (i == 0) else v for i, v in enumerate(r[4:5])) and r for r in random_tree_nodes(rng, 2, 3)]
    rows = [r[:4] + ("BRANCH_EQ",) + r[5:] if r[4] != "LEAF" else r for r in rows]
    model = ONNXModel(tree_graph(rows, 3, base=(0.1,)).SerializeToString())
    with pytest.raises(UnsupportedTreeError, match="BRANCH_EQ"):
        onnx_tree_ensemble(model)


def test_graph_with_two_inputs_is_rejected() -> None:
    a = helper.make_tensor_value_info("a", TensorProto.FLOAT, [None, 2])
    b = helper.make_tensor_value_info("b", TensorProto.FLOAT, [None, 2])
    out = helper.make_tensor_value_info("c", TensorProto.FLOAT, [None, 2])
    g = helper.make_graph([helper.make_node("Add", ["a", "b"], ["c"])], "two", [a, b], [out])
    m = helper.make_model(g, opset_imports=[helper.make_opsetid("", 17)])
    m.ir_version = 8
    with pytest.raises(ValueError, match="2 inputs"):
        ONNXModel(m.SerializeToString())


# --------------------------------------------------------------------------------------------------
# float64 tail resolution (onnxruntime applies LOGISTIC in float32: margins > ~16.6 all give 1.0)
# --------------------------------------------------------------------------------------------------


def test_xgboost_onnx_saturated_scores_do_not_tie() -> None:
    xgb = pytest.importorskip("xgboost")
    onnxmltools = pytest.importorskip("onnxmltools")
    from onnxmltools.convert.common.data_types import FloatTensorType
    from scipy.special import expit

    from malvalid.loaders.onnx_loader import ONNXModel

    rng = np.random.default_rng(0)
    X = rng.normal(size=(2000, 5)).astype(np.float32)
    y = (X[:, 0] + X[:, 1] > 0).astype(int)
    clf = xgb.XGBClassifier(n_estimators=50, max_depth=4, learning_rate=1.0, reg_lambda=0, n_jobs=THREADS).fit(X, y)
    Xe = X.copy()
    Xe[:, :2] *= 8.0
    onx = onnxmltools.convert.convert_xgboost(
        clf, initial_types=[("input", FloatTensorType([None, 5]))], target_opset=15
    )
    margin = clf.predict(Xe, output_margin=True).astype(np.float64)
    assert (margin > 17).sum() > 20
    model = ONNXModel(onx.SerializeToString(), threads=THREADS)
    graph_p = model.session.run(None, {"input": Xe})[1][:, 1]
    assert np.unique(graph_p[margin > 17]).size == 1  # the artifact is real in the raw graph output
    p = model.predict_proba(Xe)
    assert model._margin is not False
    assert np.unique(p[margin > 17]).size > 10
    np.testing.assert_allclose(p, expit(margin), rtol=0, atol=1e-6)
    ok = np.abs(margin) < 10
    np.testing.assert_allclose(p[ok], graph_p[ok], rtol=0, atol=1e-6)


@pytest.mark.parametrize("zipmap", [False, True])
def test_gbc_onnx_uses_float64_margin_path_and_matches_graph(tmp_path: Path, zipmap: bool) -> None:
    from malvalid.loaders.onnx_loader import ONNXModel

    rng = np.random.default_rng(3)
    X = rng.normal(size=(800, 6)).astype(np.float32)
    y = (X[:, 0] - X[:, 2] > 0).astype(int)
    est = GradientBoostingClassifier(n_estimators=25, max_depth=3, learning_rate=1.0, random_state=0).fit(X, y)
    path = _export(est, X, tmp_path / "gbc.onnx", **({} if zipmap else {"zipmap": False}))
    model = ONNXModel(path, threads=THREADS)
    p = model.predict_proba(X)
    assert model._margin is not False  # skl2onnx GBC: TreeEnsembleClassifier(LOGISTIC) [-> Identity | ZipMap]
    out = model.session.run([model.output_name], {model.input_name: X})[0]
    ref = np.array([r[1] for r in out]) if zipmap else out[:, 1]
    np.testing.assert_allclose(p, ref, rtol=0, atol=1e-5)
    np.testing.assert_allclose(p, est.predict_proba(X)[:, 1], rtol=0, atol=1e-4)
