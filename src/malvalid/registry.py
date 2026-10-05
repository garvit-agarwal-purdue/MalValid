"""Plugin registry for Modules, FeatureSchemas, CorpusProviders and ModelLoaders.

Third parties register plugins via entry points (groups ``malvalid.modules``,
``malvalid.feature_schemas``, ``malvalid.corpora``, ``malvalid.model_loaders``) — no fork needed.
Built-ins are also listed here so a source checkout works without installation. A plugin that
fails to import is recorded in :func:`unavailable` with the reason instead of crashing the tool.
"""

from __future__ import annotations

import importlib
import logging
from importlib.metadata import entry_points
from typing import Any

log = logging.getLogger(__name__)

GROUPS = {
    "modules": "malvalid.modules",
    "feature_schemas": "malvalid.feature_schemas",
    "corpora": "malvalid.corpora",
    "model_loaders": "malvalid.model_loaders",
}

_BUILTINS: dict[str, dict[str, str]] = {
    "modules": {
        "dummy": "malvalid.modules.dummy:DummyModule",
        "file_safety": "malvalid.modules.file_safety:FileSafetyModule",
        "performance": "malvalid.modules.performance:PerformanceModule",
        "drift": "malvalid.modules.drift:DriftModule",
        "membership_inf": "malvalid.modules.membership:MembershipInferenceModule",
        "backdoor_screen": "malvalid.modules.backdoor:BackdoorScreenModule",
        "extraction": "malvalid.modules.extraction:ExtractionModule",
        "explanation": "malvalid.modules.explanation:ExplanationModule",
    },
    "feature_schemas": {
        "ember_v2": "malvalid.schemas.ember_v2:EmberV2Schema",
        "ember_v3": "malvalid.schemas.ember_v3:EmberV3Schema",
    },
    "corpora": {
        "ember_v2_2018": "malvalid.corpora.ember2018:Ember2018Provider",
        "ember_v3_2024": "malvalid.corpora.ember2024:Ember2024Provider",
        "synthetic_v2": "malvalid.corpora.synthetic:SyntheticEmberV2Provider",
        "synthetic_v3": "malvalid.corpora.synthetic:SyntheticEmberV3Provider",
    },
    "model_loaders": {
        "lightgbm": "malvalid.loaders.lightgbm_loader:LightGBMLoader",
        "xgboost": "malvalid.loaders.xgboost_loader:XGBoostLoader",
        "sklearn_gbdt": "malvalid.loaders.sklearn_loader:SklearnLoader",
        "onnx": "malvalid.loaders.onnx_loader:ONNXLoader",
    },
}

_cache: dict[str, dict[str, Any]] = {}
_unavailable: dict[str, dict[str, str]] = {k: {} for k in GROUPS}
_extra: dict[str, dict[str, Any]] = {k: {} for k in GROUPS}  # programmatic registrations
_schema_instances: dict[str, Any] = {}


def _import_ref(ref: str) -> Any:
    mod, _, attr = ref.partition(":")
    obj = importlib.import_module(mod)
    for part in attr.split("."):
        obj = getattr(obj, part)
    return obj


def _load(kind: str) -> dict[str, Any]:
    if kind in _cache:
        return _cache[kind]
    refs: dict[str, str] = dict(_BUILTINS[kind])
    try:
        for ep in entry_points(group=GROUPS[kind]):
            refs.setdefault(ep.name, ep.value)
            if refs[ep.name] != ep.value:  # third-party override of a builtin name
                log.warning("entry point %s=%s overrides builtin %s", ep.name, ep.value, refs[ep.name])
                refs[ep.name] = ep.value
    except Exception as e:  # pragma: no cover - metadata problems
        log.warning("could not read entry points for %s: %s", kind, e)
    out: dict[str, Any] = {}
    for name, ref in refs.items():
        try:
            out[name] = _import_ref(ref)
        except Exception as e:
            _unavailable[kind][name] = f"{type(e).__name__}: {e}"
            log.debug("plugin %s:%s unavailable: %s", kind, name, e)
    out.update(_extra[kind])
    _cache[kind] = out
    return out


def register(kind: str, name: str, obj: Any) -> None:
    """Register a plugin programmatically (tests, notebooks)."""
    if kind not in GROUPS:
        raise KeyError(kind)
    _extra[kind][name] = obj
    _cache.pop(kind, None)
    _schema_instances.pop(name, None)


def unregister(kind: str, name: str) -> None:
    _extra[kind].pop(name, None)
    _cache.pop(kind, None)
    _schema_instances.pop(name, None)


def reset() -> None:
    _cache.clear()
    _schema_instances.clear()
    for k in GROUPS:
        _unavailable[k].clear()


def unavailable(kind: str | None = None) -> dict[str, Any]:
    if kind is None:
        for k in GROUPS:
            _load(k)
        return {k: dict(v) for k, v in _unavailable.items()}
    _load(kind)
    return dict(_unavailable[kind])


# ---- typed accessors -----------------------------------------------------------------------------


def modules() -> dict[str, Any]:
    """{id: Module subclass}, ordered by scorecard code (M0, M1, ...)."""
    mods = _load("modules")
    return dict(sorted(mods.items(), key=lambda kv: (getattr(kv[1], "code", "Z"), kv[0])))


def get_module(module_id: str) -> Any:
    mods = modules()
    if module_id not in mods:
        why = _unavailable["modules"].get(module_id)
        raise KeyError(
            f"unknown module {module_id!r}" + (f" (failed to import: {why})" if why else "")
            + f"; known: {sorted(mods)}"
        )
    return mods[module_id]


def feature_schemas() -> dict[str, Any]:
    return dict(_load("feature_schemas"))


def get_schema(name: str) -> Any:
    """A (cached) FeatureSchema instance."""
    if name not in _schema_instances:
        schemas = feature_schemas()
        if name not in schemas:
            why = _unavailable["feature_schemas"].get(name)
            raise KeyError(
                f"unknown feature schema {name!r}" + (f" (failed to import: {why})" if why else "")
                + f"; known: {sorted(schemas)}"
            )
        _schema_instances[name] = schemas[name]()
    return _schema_instances[name]


def corpora() -> dict[str, Any]:
    return dict(_load("corpora"))


def get_corpus_provider(name: str) -> Any:
    provs = corpora()
    if name not in provs:
        why = _unavailable["corpora"].get(name)
        raise KeyError(
            f"unknown corpus {name!r}" + (f" (failed to import: {why})" if why else "")
            + f"; known: {sorted(provs)}"
        )
    return provs[name]()


def model_loaders() -> dict[str, Any]:
    return dict(_load("model_loaders"))


def get_loader(kind: str) -> Any:
    lds = model_loaders()
    if kind not in lds:
        why = _unavailable["model_loaders"].get(kind)
        raise KeyError(
            f"no model loader {kind!r}" + (f" (failed to import: {why})" if why else "")
            + f"; known: {sorted(lds)}"
        )
    return lds[kind]()
