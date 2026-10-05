"""A trivial module proving the pipeline runs end to end (milestone 1). Disabled by default."""

from __future__ import annotations

import numpy as np

from malvalid.core import GateCheck, Module, ModuleResult, Requirement


class DummyModule(Module):
    id = "dummy"
    code = "X0"
    title = "Pipeline smoke test"
    description = "Scores a handful of random vectors and checks the adapter returns valid scores."
    requires = (Requirement.QUERY_ONLY,)
    default_params = {"n": 16}

    def run(self, ctx) -> ModuleResult:
        n = int(ctx.params["n"])
        X = ctx.rng.random((n, ctx.schema.dim), dtype=np.float32)
        p = ctx.score(X)
        return self.result(
            ctx,
            finding=f"Scored {n} random vectors; scores in [{p.min():.3f}, {p.max():.3f}].",
            checks=[GateCheck.evaluate("valid_scores", float(np.isfinite(p).all()), "==", 1.0)],
            metrics={"n": n, "mean_score": float(p.mean())},
        )
