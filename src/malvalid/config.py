"""Gate configuration: YAML loading, defaults, and validation."""

from __future__ import annotations

import copy
from importlib import resources
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator

from malvalid.core import ConfigError, GateMode

# Module config keys that are not module parameters.
_RESERVED = {"enabled", "gate"}


class ModuleConfig(BaseModel):
    model_config = ConfigDict(extra="allow")

    enabled: bool = True
    gate: GateMode = GateMode.WARN

    def params(self) -> dict[str, Any]:
        return dict(self.model_extra or {})


class RuntimeConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    seed: int = 0
    sandbox: bool = True
    sandbox_backend: Literal["auto", "bwrap", "unshare", "subprocess"] = "auto"
    # When `auto` finds no OS sandbox (Windows, macOS, Linux without user namespaces): False = refuse to
    # run; True = run the model in a plain worker process (reduced isolation, recorded as
    # isolation: process_only in report.json). `malvalid run/serve --allow-reduced-isolation`.
    allow_reduced_isolation: bool = False
    sandbox_memory_mb: int = Field(32768, ge=256)
    threads: int = Field(8, ge=1)
    max_seconds_per_module: float = Field(1800, gt=0)
    allow_pickle: bool = False
    skipped_hard_gate_fails: bool = True
    chunk_rows: int = Field(20000, ge=1)
    # "cached": trust a corpus file whose size/mtime/ctime/inode match the last verified hash;
    # "full": re-hash every corpus file on load (`malvalid run --verify-corpus`).
    corpus_verification: Literal["cached", "full"] = "cached"
    # Which benign eval rows an auto-calibrated threshold (--calibrate-fpr) is fit on:
    # "earliest" = the earliest 10% by timestamp (a held threshold, fit before the scored period);
    # "uniform" = a 10% sha256 slice across the whole eval period (sees the scored period).
    calibration_period: Literal["earliest", "uniform"] = "earliest"


class ReportConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    title: str | None = None
    html: bool = True


class VerdictConfig(BaseModel):
    """Policy for the headline production-readiness verdict (see malvalid.verdict)."""

    model_config = ConfigDict(extra="forbid")

    ready_min: float = Field(80, ge=0, le=100)
    conditional_min: float = Field(60, ge=0, le=100)
    min_coverage_ready: float = Field(0.75, ge=0, le=1)
    blocked_score_cap: float = Field(49, ge=0, le=100)
    default_weight: float = Field(1.0, ge=0)
    required_for_ready: list[str] = Field(default_factory=lambda: ["performance"])
    weights: dict[str, float] = Field(default_factory=dict)

    @field_validator("weights")
    @classmethod
    def _nonneg(cls, v: dict[str, float]) -> dict[str, float]:
        for k, w in v.items():
            if w < 0:
                raise ValueError(f"verdict weight for {k} must be >= 0")
        return v

    def model_post_init(self, __context: Any) -> None:
        if self.conditional_min > self.ready_min:
            raise ValueError("verdict.conditional_min must be <= verdict.ready_min")


class GateConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    corpus: str = "ember_v2_2018"
    corpus_dir: str | None = None
    sample_dir: str | None = None
    modules: dict[str, ModuleConfig] = Field(default_factory=dict)
    runtime: RuntimeConfig = Field(default_factory=RuntimeConfig)
    report: ReportConfig = Field(default_factory=ReportConfig)
    verdict: VerdictConfig = Field(default_factory=VerdictConfig)
    source_path: str | None = None  # where this config was loaded from (informational)

    def module(self, module_id: str) -> ModuleConfig:
        return self.modules.get(module_id) or ModuleConfig(enabled=False)

    def to_dict(self) -> dict[str, Any]:
        d = self.model_dump(mode="json")
        for mid, mc in self.modules.items():
            d["modules"][mid] = {"enabled": mc.enabled, "gate": mc.gate.value, **mc.params()}
        return d


def default_config_text() -> str:
    return resources.files("malvalid").joinpath("default_gate.yaml").read_text()


def _deep_merge(base: dict[str, Any], over: dict[str, Any]) -> dict[str, Any]:
    out = copy.deepcopy(base)
    for k, v in (over or {}).items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = copy.deepcopy(v)
    return out


def load_config(path: str | Path | None = None, overrides: dict[str, Any] | None = None) -> GateConfig:
    """Load the packaged defaults, deep-merge the user's YAML and ``overrides`` on top, validate."""
    base = yaml.safe_load(default_config_text()) or {}
    user: dict[str, Any] = {}
    if path is not None:
        p = Path(path)
        if not p.exists():
            raise ConfigError(f"config file not found: {p}")
        try:
            user = yaml.safe_load(p.read_text()) or {}
        except yaml.YAMLError as e:
            raise ConfigError(f"{p}: invalid YAML: {e}") from e
        if not isinstance(user, dict):
            raise ConfigError(f"{p}: top level must be a mapping")
    merged = _deep_merge(base, user)
    if overrides:
        merged = _deep_merge(merged, overrides)
    if path is not None:
        merged["source_path"] = str(Path(path).resolve())
    try:
        cfg = GateConfig.model_validate(merged)
    except ValidationError as e:
        raise ConfigError(f"invalid gate config: {e}") from e
    return cfg


def validate_against_registry(cfg: GateConfig) -> list[str]:
    """Check module ids and parameter names against registered modules.

    Unknown module ids and unknown parameter keys are errors (a typo in a gate silently changing
    policy is worse than a loud failure). Returns non-fatal warnings.
    """
    from malvalid import registry

    mods = registry.modules()
    warnings: list[str] = []
    errors: list[str] = []
    for mid, mc in cfg.modules.items():
        if mid not in mods:
            why = registry.unavailable("modules").get(mid)
            if why:
                if mc.enabled:
                    errors.append(f"module {mid!r} is enabled but failed to import: {why}")
                else:
                    warnings.append(f"module {mid!r} unavailable ({why})")
                continue
            errors.append(f"unknown module {mid!r} in config; known: {sorted(mods)}")
            continue
        allowed = set(mods[mid].default_params)
        unknown = set(mc.params()) - allowed
        if unknown:
            errors.append(
                f"module {mid!r}: unknown parameter(s) {sorted(unknown)}; allowed: {sorted(allowed)}"
            )
            continue
        # Parameter *values* too, so a bad value fails here instead of after the model is loaded.
        check = getattr(mods[mid], "validate_params", None)
        if callable(check):
            try:
                check(module_params(cfg, mods[mid]))
            except ConfigError as e:
                errors.append(str(e))
            except (TypeError, ValueError, KeyError) as e:
                errors.append(f"module {mid!r}: invalid parameter value ({type(e).__name__}: {e})")
    vc = cfg.verdict
    if vc.blocked_score_cap >= vc.conditional_min:
        warnings.append(
            f"verdict.blocked_score_cap ({vc.blocked_score_cap:g}) is not below verdict.conditional_min "
            f"({vc.conditional_min:g}): a BLOCKED run can then show a score inside the CONDITIONAL/READY "
            "bands; keep the cap below conditional_min"
        )
    for k in list(cfg.verdict.weights) + list(cfg.verdict.required_for_ready):
        if k not in mods:
            warnings.append(f"verdict config references unknown module {k!r}")
    for k in cfg.verdict.required_for_ready:
        if k in cfg.modules and not cfg.modules[k].enabled:
            warnings.append(f"verdict.required_for_ready lists {k!r} but it is disabled: READY is unreachable")
    if errors:
        raise ConfigError("; ".join(errors))
    return warnings


def module_params(cfg: GateConfig, module_cls: Any) -> dict[str, Any]:
    """Defaults from the module class overlaid with the config's values for that module."""
    params = copy.deepcopy(dict(module_cls.default_params))
    params.update(copy.deepcopy(cfg.module(module_cls.id).params()))
    return params
