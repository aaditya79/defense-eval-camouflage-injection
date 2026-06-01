"""
Statistical significance testing for defense evaluation results.

For each (model, defense) pair runs:
  1. McNemar's test — paired comparison of static vs camouflage ASR
     (are camouflage injections significantly more effective than static?)
  2. Chi-square test — defense camouflage ASR vs baseline camouflage ASR
     (does this defense provide a significant reduction?)
  3. Cohen's h — effect size for the defense vs baseline comparison
  4. Minimum-n analysis — trials needed for 80% power at observed effect size

Usage:
  python experiments/statistical_tests.py
  python experiments/statistical_tests.py --live-only
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import os
from pathlib import Path
from typing import Optional

import numpy as np
from scipy.stats import chi2_contingency

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from experiments.exp5_defense_eval import (
    DEFENSES,
    MODEL_CONFIGS,
    OUTPUT_FILE,
    _HAIKU_FALLBACK,
)


# ---------------------------------------------------------------------------
# McNemar's exact test (manual — scipy's version requires newer API)
# ---------------------------------------------------------------------------

def mcnemar_exact_p(b: int, c: int) -> float:
    """
    Two-sided exact McNemar p-value.

    b = discordant pairs where camouflage succeeded but static did not.
    c = discordant pairs where static succeeded but camouflage did not.
    """
    n = b + c
    if n == 0:
        return 1.0
    # Exact binomial: probability of getting at least as extreme as observed
    # under H0: P(success) = 0.5
    from scipy.stats import binom
    k_obs = max(b, c)
    p = 2 * binom.sf(k_obs - 1, n, 0.5)
    return min(p, 1.0)


# ---------------------------------------------------------------------------
# Cohen's h (effect size for two proportions)
# ---------------------------------------------------------------------------

def cohens_h(p1: float, p2: float) -> float:
    """Cohen's h for the difference between two proportions."""
    phi1 = 2 * math.asin(math.sqrt(max(0.0, min(1.0, p1))))
    phi2 = 2 * math.asin(math.sqrt(max(0.0, min(1.0, p2))))
    return abs(phi1 - phi2)


# ---------------------------------------------------------------------------
# Minimum-n for 80% power (chi-square, two-sided, alpha=0.05)
# ---------------------------------------------------------------------------

def min_n_for_power(h: float, alpha: float = 0.05, power: float = 0.80) -> int:
    """
    Approximate minimum n per group for a two-sample proportions test
    at the observed Cohen's h effect size.

    Uses the formula: n = (z_alpha/2 + z_beta)^2 / h^2
    """
    if h < 1e-6:
        return 9999  # immeasurably small effect — no finite n suffices
    from scipy.stats import norm
    z_a = norm.ppf(1 - alpha / 2)
    z_b = norm.ppf(power)
    return math.ceil((z_a + z_b) ** 2 / h ** 2)


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_trials(live_only: bool) -> list[dict]:
    if not OUTPUT_FILE.exists():
        print(f"No trial file found at {OUTPUT_FILE}", file=sys.stderr)
        sys.exit(1)
    records = []
    with open(OUTPUT_FILE) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                r = json.loads(line)
                if live_only and r.get("is_dry_run", False):
                    continue
                if not r.get("model"):
                    r["model"] = _HAIKU_FALLBACK
                records.append(r)
            except json.JSONDecodeError:
                continue
    return records


# ---------------------------------------------------------------------------
# Core analysis
# ---------------------------------------------------------------------------

def run_tests(trials: list[dict], models: list[str]) -> None:
    label_map = {v: k for k, v in MODEL_CONFIGS.items()}

    # ── Table header ────────────────────────────────────────────────────
    col = 12
    hdr = (
        f"{'Model':<8} | {'Defense':<22} | "
        f"{'McNemar_p':>{col}} | {'ChiSq_vs_base_p':>{col}} | "
        f"{'Cohen_h':>{col}} | {'min_n':>6} | {'sig?':<12}"
    )
    print("\n=== Statistical Tests (α=0.05) ===\n")
    print(hdr)
    print("-" * len(hdr))

    for model_id in models:
        mlabel = label_map.get(model_id, model_id[:8])
        m_trials = [t for t in trials if t["model"] == model_id]

        # Baseline camouflage ASR for chi-square comparisons
        base_cam = [t for t in m_trials
                    if t.get("defense") == "baseline"
                    and t.get("payload_type") == "camouflage"]
        base_n     = len(base_cam)
        base_succ  = sum(1 for t in base_cam if t.get("followed_injection"))
        base_asr   = base_succ / base_n if base_n else 0.0

        for defense in DEFENSES:
            d_trials  = [t for t in m_trials if t.get("defense") == defense]
            s_trials  = [t for t in d_trials if t.get("payload_type") == "static"]
            c_trials  = [t for t in d_trials if t.get("payload_type") == "camouflage"]

            # ── McNemar: pair trials by task_id ─────────────────────────
            s_by_task = {t["task_id"]: t.get("followed_injection", False)
                         for t in s_trials}
            c_by_task = {t["task_id"]: t.get("followed_injection", False)
                         for t in c_trials}
            shared    = set(s_by_task) & set(c_by_task)
            b = sum(1 for tid in shared
                    if c_by_task[tid] and not s_by_task[tid])  # cam only
            c_cnt = sum(1 for tid in shared
                        if s_by_task[tid] and not c_by_task[tid])  # static only
            mcnemar_p = mcnemar_exact_p(b, c_cnt) if shared else float("nan")

            # ── Chi-square: defense camouflage vs baseline camouflage ───
            d_succ = sum(1 for t in c_trials if t.get("followed_injection"))
            d_n    = len(c_trials)
            if defense == "baseline" or base_n == 0 or d_n == 0:
                chisq_p = float("nan")
                h       = float("nan")
                n_req   = 0
            else:
                d_asr = d_succ / d_n
                h     = cohens_h(base_asr, d_asr)
                contingency = [
                    [base_succ, base_n - base_succ],
                    [d_succ,    d_n    - d_succ],
                ]
                # Use Fisher's exact if any expected cell < 5
                min_expected = min(
                    (base_n * base_succ) / (base_n + d_n),
                    (d_n    * base_succ) / (base_n + d_n),
                    (base_n * d_succ)    / (base_n + d_n),
                    (d_n    * d_succ)    / (base_n + d_n),
                ) if (base_succ + d_succ) > 0 else 0
                if min_expected < 5:
                    from scipy.stats import fisher_exact
                    _, chisq_p = fisher_exact(contingency)
                else:
                    _, chisq_p, _, _ = chi2_contingency(contingency, correction=False)
                n_req = min_n_for_power(h) if not math.isnan(h) and h > 0 else 0

            # ── Significance label ──────────────────────────────────────
            def _sig(p: float) -> str:
                if math.isnan(p): return "—"
                return "YES" if p <= 0.05 else "NO (p>{:.2f})".format(p)

            def _fmt_p(p: float) -> str:
                if math.isnan(p): return f"{'—':>{col}}"
                return f"{p:>{col}.4f}"

            def _fmt_h(h: float) -> str:
                if math.isnan(h): return f"{'—':>{col}}"
                return f"{h:>{col}.3f}"

            sig_str = _sig(mcnemar_p)
            if not math.isnan(chisq_p):
                sig_str = _sig(chisq_p)

            print(
                f"{mlabel:<8} | {defense:<22} | "
                f"{_fmt_p(mcnemar_p)} | {_fmt_p(chisq_p)} | "
                f"{_fmt_h(h)} | {n_req if n_req else '—':>6} | {sig_str:<12}"
            )
        print()

    # ── Interpretation note ──────────────────────────────────────────────
    print("Notes:")
    print("  McNemar_p   : paired test — is camouflage ASR significantly > static ASR?")
    print("  ChiSq_vs_base_p : is this defense significantly better than baseline?")
    print("                    (Fisher exact used when expected cell counts < 5)")
    print("  Cohen_h     : effect size of defense vs baseline (0.2=small, 0.5=medium, 0.8=large)")
    print("  min_n       : trials per group needed for 80% power at observed Cohen's h")
    print("  n=45 per cell — most comparisons are underpowered; treat as exploratory.")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--live-only", action="store_true")
    parser.add_argument("--model", choices=list(MODEL_CONFIGS), default=None)
    args = parser.parse_args()

    trials = load_trials(live_only=args.live_only)
    print(f"Loaded {len(trials)} trial records")

    if args.model:
        models = [MODEL_CONFIGS[args.model]]
    else:
        present = {t["model"] for t in trials}
        models = [m for m in MODEL_CONFIGS.values() if m in present]

    run_tests(trials, models)


if __name__ == "__main__":
    main()
