"""Candidate configuration construction for the optimization engine.

A *candidate* is a flat mapping ``{param_key: value}`` produced by an
optimization algorithm. This module translates a candidate into a fully-formed,
isolated :class:`~core.config.backtest_schema.BacktestParameters` instance --
without mutating the user's original configuration.

It also handles the cross-parameter relationships that the flat schema encodes:

* **Sum-groups** (cap weights, scoring weights, quality pillars): when a group
  is being optimized it is auto-normalised so the underlying frozen dataclasses'
  ``__post_init__`` sum-checks pass, while still letting the optimizer search the
  relative proportions.
* **Cap allocation -> portfolio counts**: ``cap_segment`` weights are mirrored
  into ``portfolio``'s ``large/mid/small_size`` counts based on total size, so a
  candidate remains internally consistent.

The engine never hardcodes any module logic here; it only knows ``(block, field)``
paths supplied by :mod:`core.optimization.spec`.
"""

from __future__ import annotations

from dataclasses import replace
from typing import Any

from core.config.backtest_schema import BacktestParameters
from core.optimization.spec import get_spec, get_specs, OptimizerParamSpec


def _group_members(specs: list[OptimizerParamSpec]) -> dict[str, list[OptimizerParamSpec]]:
    groups: dict[str, list[OptimizerParamSpec]] = {}
    for spec in specs:
        if spec.group is not None:
            groups.setdefault(spec.group, []).append(spec)
    return groups


def normalize_candidate(
    base: BacktestParameters,
    values: dict[str, Any],
    specs: list[OptimizerParamSpec],
) -> dict[str, Any]:
    """Return a copy of ``values`` with sum-groups normalised to their target.

    A sum-group is normalised across **all** its members -- both the optimized
    ones (from ``values``) and the non-optimized ones (read from ``base``) -- so
    the resulting config always satisfies the schema's ``sum == target`` check,
    regardless of how many members are being optimized. Within the optimized
    subset the *relative proportions* chosen by the algorithm are preserved.

    Group membership is taken from the **full** spec registry so that every
    member of a validated group (e.g. all three scoring weights) is included in
    the normalization even when only a subset is being optimized.
    """
    out = dict(values)
    all_specs = get_specs()
    for group, members in _group_members(all_specs).items():
        target = members[0].group_target
        # Only touch groups that have at least one optimized member.
        if not any(m.key in out for m in members):
            continue
        # Base values for every member of the group.
        base_vals: dict[str, float] = {}
        for m in members:
            block = getattr(base, m.block)
            base_vals[m.key] = float(getattr(block, m.field, m.current))
        # Blend optimized values over base values.
        blended = {m.key: float(out[m.key]) if m.key in out else base_vals[m.key] for m in members}
        opt_keys = [m.key for m in members if m.key in out]
        opt_total = sum(blended[k] for k in opt_keys)
        if opt_total <= 0:
            share = target / len(opt_keys)
            for k in opt_keys:
                blended[k] = share
        # Scale the whole group so it sums to target (preserves relative
        # proportions among the optimized members).
        scale = target / sum(blended.values())
        for k in blended:
            blended[k] = blended[k] * scale
        for k, v in blended.items():
            out[k] = v
    return out


def build_candidate(
    base: BacktestParameters,
    values: dict[str, Any],
    specs: list[OptimizerParamSpec],
) -> BacktestParameters:
    """Construct an isolated candidate configuration from ``base`` + ``values``.

    ``base`` is never mutated. Parameters present in ``values`` overwrite the
    corresponding ``(block, field)``; everything else is inherited from ``base``.
    Sum-groups are normalised first (using the full spec registry) so the
    resulting config passes schema validation. Non-optimized members of a group
    that has optimized members are still injected (scaled) to keep the group
    consistent.
    """
    values = normalize_candidate(base, values, specs)
    # Ensure every (possibly non-selected) normalized group member is applied.
    spec_by_key = {s.key: s for s in get_specs()}
    for key, val in list(values.items()):
        if key not in {s.key for s in specs} and key in spec_by_key:
            specs = specs + [spec_by_key[key]]

    # Group incoming values by block.
    by_block: dict[str, dict[str, Any]] = {}
    for spec in specs:
        if spec.key in values:
            by_block.setdefault(spec.block, {})[spec.field] = values[spec.key]

    # Build replacement blocks via dataclasses.replace (frozen-safe).
    general = replace(base.general, **by_block.get("general", {}))
    regime = replace(base.regime, **by_block.get("regime", {}))
    universe = replace(base.universe, **by_block.get("universe", {}))
    cap_segment = replace(base.cap_segment, **by_block.get("cap_segment", {}))
    momentum = replace(base.momentum, **by_block.get("momentum", {}))
    stability = replace(base.stability, **by_block.get("stability", {}))
    persistence = replace(base.persistence, **by_block.get("persistence", {}))
    quality = replace(base.quality, **by_block.get("quality", {}))
    scoring = replace(base.scoring, **by_block.get("scoring", {}))
    portfolio = replace(base.portfolio, **by_block.get("portfolio", {}))

    # Derive cap counts from the (possibly optimized) cap weights.
    if any(k in values for k in ("large_cap_weight", "mid_cap_weight", "small_cap_weight")):
        total = portfolio.total_size
        lc = max(0, round(cap_segment.large_cap_weight * total))
        mc = max(0, round(cap_segment.mid_cap_weight * total))
        sc = max(0, total - lc - mc)
        portfolio = replace(portfolio, large_size=lc, mid_size=mc, small_size=sc)

    return BacktestParameters(
        general=general,
        regime=regime,
        universe=universe,
        cap_segment=cap_segment,
        momentum=momentum,
        stability=stability,
        persistence=persistence,
        quality=quality,
        scoring=scoring,
        portfolio=portfolio,
        management=base.management,
        pipeline=base.pipeline,
    )
