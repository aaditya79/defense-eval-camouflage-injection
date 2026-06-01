"""
Statistical significance tests for GroundedGuard detection comparison.

Extends paper 1 (McNemar's test) with:
  - Wilson score confidence intervals for IDR estimates
  - Bootstrap confidence intervals
  - Cohen's h effect size for proportion differences
  - Newcombe CI for the CDG (difference between two proportions)
  - Full stats summary per detector

Usage:
  cd grounded-guard
  python analysis/significance_tests.py
"""

from __future__ import annotations

import json
import math
import sys
from collections import defaultdict
from pathlib import Path
from typing import Optional

import numpy as np
from scipy.stats import chi2 as chi2_dist, norm


RESULTS_DIR = Path("results")


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_all_trials() -> dict[str, list[dict]]:
    """Load all trials_{model}.jsonl files. Returns {model_tag: [trials]}."""
    all_data: dict[str, list[dict]] = {}
    for path in sorted(RESULTS_DIR.glob("trials_*.jsonl")):
        tag = path.stem.replace("trials_", "")
        records = []
        with open(path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        records.append(json.loads(line))
                    except json.JSONDecodeError:
                        continue
        if records:
            all_data[tag] = records
    return all_data


# ---------------------------------------------------------------------------
# McNemar's test (from paper 1, extended)
# ---------------------------------------------------------------------------

def mcnemar_test(b: int, c: int, continuity: bool = True) -> tuple[float, float]:
    """McNemar's test on discordant pairs b and c. Returns (chi2, p_value)."""
    if b + c == 0:
        return float("nan"), float("nan")
    if continuity:
        chi2 = (abs(b - c) - 1) ** 2 / (b + c)
    else:
        chi2 = (b - c) ** 2 / (b + c)
    p = chi2_dist.sf(chi2, df=1)
    return chi2, p


def task_level_detection(trials: list[dict], detector_type: str, payload_type: str) -> dict[str, bool]:
    """
    For each task_id, return True if majority of trials of the given payload_type
    were detected (verdict == INJECTED) by the given detector.
    """
    by_task: dict[str, list[bool]] = defaultdict(list)
    for t in trials:
        if t.get("payload_type") != payload_type:
            continue
        verdict = t.get("verdicts", {}).get(detector_type, {}).get("verdict")
        if verdict is not None:
            by_task[t["task_id"]].append(verdict == "INJECTED")
    return {tid: sum(v) > len(v) / 2 for tid, v in by_task.items()}


def run_mcnemar_pair(
    trials: list[dict],
    detector_a: str,
    detector_b: str,
    payload_type: str,
    label: str,
) -> None:
    """Run McNemar's comparing detector_a vs detector_b on the given payload type."""
    det_a = task_level_detection(trials, detector_a, payload_type)
    det_b = task_level_detection(trials, detector_b, payload_type)

    common = sorted(set(det_a) & set(det_b))
    if not common:
        print(f"  [{label}] No overlapping task_ids — cannot run McNemar.")
        return

    a = b = c = d = 0
    for tid in common:
        ra, rb = det_a[tid], det_b[tid]
        if ra and rb:      a += 1
        elif ra and not rb: b += 1
        elif not ra and rb: c += 1
        else:               d += 1

    chi2, p = mcnemar_test(b, c)
    print(f"\n  McNemar: {label}")
    print(f"    H0: {detector_a} and {detector_b} perform equally on {payload_type} payloads")
    print(f"    Paired tasks: {len(common)} | a={a} b={b} c={c} d={d}")
    print(f"    χ²={chi2:.3f}  p={p:.6f}", end="")
    if np.isnan(p):
        print(" (no discordant pairs)")
    elif p < 0.001:
        print(" ***")
    elif p < 0.01:
        print(" **")
    elif p < 0.05:
        print(" *")
    else:
        print(" (n.s.)")


# ---------------------------------------------------------------------------
# Wilson score confidence interval
# ---------------------------------------------------------------------------

def compute_wilson_ci(
    n_correct: int,
    n_total: int,
    confidence: float = 0.95,
) -> tuple[float, float]:
    """
    Wilson score confidence interval for a proportion.

    Args:
        n_correct: Number of successes.
        n_total: Total observations.
        confidence: Confidence level (default 0.95).

    Returns:
        (lower, upper) Wilson score CI.
    """
    if n_total == 0:
        return (0.0, 0.0)
    z = norm.ppf(1 - (1 - confidence) / 2)
    p_hat = n_correct / n_total
    center = (p_hat + z**2 / (2 * n_total)) / (1 + z**2 / n_total)
    margin = (z * math.sqrt(p_hat * (1 - p_hat) / n_total + z**2 / (4 * n_total**2))) / (
        1 + z**2 / n_total
    )
    return (max(0.0, center - margin), min(1.0, center + margin))


# ---------------------------------------------------------------------------
# Bootstrap confidence interval
# ---------------------------------------------------------------------------

def compute_bootstrap_ci(
    successes: int,
    n_trials: int,
    n_bootstrap: int = 2000,
    confidence: float = 0.95,
    seed: int = 42,
) -> tuple[float, float]:
    """
    Bootstrap CI for a proportion via parametric (binomial) resampling.

    Args:
        successes: Number of successes observed.
        n_trials: Total number of trials.
        n_bootstrap: Number of bootstrap resamples.
        confidence: Confidence level.
        seed: Random seed for reproducibility.

    Returns:
        (lower, upper) bootstrap percentile CI.
    """
    if n_trials == 0:
        return (0.0, 0.0)
    rng = np.random.default_rng(seed)
    p_hat = successes / n_trials
    samples = rng.binomial(n=n_trials, p=p_hat, size=n_bootstrap) / n_trials
    alpha = 1 - confidence
    lower = float(np.percentile(samples, 100 * alpha / 2))
    upper = float(np.percentile(samples, 100 * (1 - alpha / 2)))
    return (lower, upper)


# ---------------------------------------------------------------------------
# Cohen's h effect size
# ---------------------------------------------------------------------------

def compute_cohens_h(p1: float, p2: float) -> float:
    """
    Cohen's h effect size for the difference between two proportions.

    h = 2 * arcsin(sqrt(p1)) - 2 * arcsin(sqrt(p2))

    Interpretation: |h| < 0.2 small, 0.2–0.5 medium, >0.5 large.

    Args:
        p1: First proportion.
        p2: Second proportion (baseline / comparison).

    Returns:
        Cohen's h (can be negative if p1 < p2).
    """
    phi1 = 2 * math.asin(math.sqrt(max(0.0, min(1.0, p1))))
    phi2 = 2 * math.asin(math.sqrt(max(0.0, min(1.0, p2))))
    return phi1 - phi2


# ---------------------------------------------------------------------------
# Newcombe CI for difference of proportions (CDG)
# ---------------------------------------------------------------------------

def compute_newcombe_ci(
    p1: float, n1: int,
    p2: float, n2: int,
    confidence: float = 0.95,
) -> tuple[float, float]:
    """
    Newcombe hybrid score interval for the difference p1 - p2.

    Uses Wilson score intervals for each proportion and combines via
    Newcombe (1998) method. More accurate than normal approximation
    at extreme proportions.

    Returns:
        (lower, upper) CI for (p1 - p2).
    """
    l1, u1 = compute_wilson_ci(round(p1 * n1), n1, confidence)
    l2, u2 = compute_wilson_ci(round(p2 * n2), n2, confidence)

    diff = p1 - p2
    lower = diff - math.sqrt((p1 - l1)**2 + (u2 - p2)**2)
    upper = diff + math.sqrt((u1 - p1)**2 + (p2 - l2)**2)
    return (lower, upper)


# ---------------------------------------------------------------------------
# Full stats summary
# ---------------------------------------------------------------------------

def compute_full_stats(
    n_correct: int,
    n_total: int,
    label: str = "",
    verbose: bool = True,
) -> dict:
    """
    Compute IDR with Wilson CI, Bootstrap CI, and Cohen's h vs chance (0.5).

    Args:
        n_correct: Number of INJECTED verdicts on injected trials.
        n_total: Total injected trials.
        label: Label for printing.
        verbose: Whether to print the stats.

    Returns:
        Dict with idr, wilson_ci, bootstrap_ci, cohens_h.
    """
    idr = n_correct / n_total if n_total > 0 else 0.0
    wilson = compute_wilson_ci(n_correct, n_total)
    bootstrap = compute_bootstrap_ci(n_correct, n_total)
    h = compute_cohens_h(idr, 0.5)

    result = {
        "idr":          round(idr, 4),
        "n_correct":    n_correct,
        "n_total":      n_total,
        "wilson_ci":    (round(wilson[0], 4), round(wilson[1], 4)),
        "bootstrap_ci": (round(bootstrap[0], 4), round(bootstrap[1], 4)),
        "cohens_h_vs_chance": round(h, 4),
    }

    if verbose:
        tag = f" [{label}]" if label else ""
        print(f"  IDR{tag}: {idr:.4f} ({n_correct}/{n_total})")
        print(f"    Wilson 95% CI:    [{wilson[0]:.4f}, {wilson[1]:.4f}]")
        print(f"    Bootstrap 95% CI: [{bootstrap[0]:.4f}, {bootstrap[1]:.4f}]")
        print(f"    Cohen's h vs 0.5: {h:.4f}  "
              f"({'large' if abs(h)>0.5 else 'medium' if abs(h)>0.2 else 'small'})")

    return result


def compute_cdg_stats(
    idr_static: float, n_static: int,
    idr_cam:    float, n_cam: int,
    label: str = "",
    verbose: bool = True,
) -> dict:
    """
    Compute CDG with Newcombe CI and Cohen's h for the gap.

    Args:
        idr_static: Detection rate on static payloads.
        n_static: Number of static trials.
        idr_cam: Detection rate on camouflage payloads.
        n_cam: Number of camouflage trials.

    Returns:
        Dict with cdg, newcombe_ci, cohens_h.
    """
    cdg = idr_static - idr_cam
    newcombe = compute_newcombe_ci(idr_static, n_static, idr_cam, n_cam)
    h = compute_cohens_h(idr_static, idr_cam)

    result = {
        "cdg":           round(cdg, 4),
        "newcombe_ci":   (round(newcombe[0], 4), round(newcombe[1], 4)),
        "cohens_h_gap":  round(h, 4),
    }

    if verbose:
        tag = f" [{label}]" if label else ""
        print(f"\n  CDG{tag}: {cdg:.4f}  (IDR_static={idr_static:.4f}, IDR_cam={idr_cam:.4f})")
        print(f"    Newcombe 95% CI: [{newcombe[0]:.4f}, {newcombe[1]:.4f}]")
        print(f"    Cohen's h (gap): {h:.4f}  "
              f"({'large' if abs(h)>0.5 else 'medium' if abs(h)>0.2 else 'small'})")

    return result


# ---------------------------------------------------------------------------
# Main analysis
# ---------------------------------------------------------------------------

def main() -> None:
    all_trials_by_model = load_all_trials()

    if not all_trials_by_model:
        print(f"No trial files found in {RESULTS_DIR}/")
        print("Run:  python experiments/run_all.py --dry-run")
        sys.exit(1)

    print(f"Loaded trials from: {sorted(all_trials_by_model.keys())}\n")

    detector_types = ["static", "augmented", "llama_guard", "grounded_guard"]

    for model_tag, trials in sorted(all_trials_by_model.items()):
        print(f"\n{'='*70}")
        print(f"MODEL: {model_tag}  ({len(trials)} trials)")
        print(f"{'='*70}")

        static_trials = [t for t in trials if t.get("payload_type") == "static"]
        cam_trials    = [t for t in trials if t.get("payload_type") == "camouflage"]

        print(f"\n  Static trials: {len(static_trials)},  Camouflage trials: {len(cam_trials)}\n")

        # Per-detector full stats
        for dt in detector_types:
            header = f"{'─'*50}"
            print(header)
            print(f"  DETECTOR: {dt.upper()}")

            n_s = sum(
                1 for t in static_trials
                if t.get("verdicts", {}).get(dt, {}).get("verdict") == "INJECTED"
            )
            n_c = sum(
                1 for t in cam_trials
                if t.get("verdicts", {}).get(dt, {}).get("verdict") == "INJECTED"
            )

            print(f"  Static payloads:")
            s_stats = compute_full_stats(n_s, len(static_trials), label="static")

            print(f"  Camouflage payloads:")
            c_stats = compute_full_stats(n_c, len(cam_trials), label="camouflage")

            cdg_stats = compute_cdg_stats(
                s_stats["idr"], len(static_trials),
                c_stats["idr"], len(cam_trials),
                label=dt,
            )

        # McNemar: GroundedGuard vs StaticDetector on camouflage
        print(f"\n{'─'*50}")
        print("  MCNEMAR TESTS (task-level, camouflage payloads)")
        for dt in ["augmented", "llama_guard", "grounded_guard"]:
            run_mcnemar_pair(
                trials, "static", dt,
                payload_type="camouflage",
                label=f"static vs {dt}",
            )

        # McNemar: static vs camouflage for grounded_guard specifically
        print(f"\n{'─'*50}")
        print("  MCNEMAR: static payload vs camouflage (per detector)")
        for dt in ["static", "grounded_guard"]:
            det_static = task_level_detection(trials, dt, "static")
            det_cam    = task_level_detection(trials, dt, "camouflage")
            common = sorted(set(det_static) & set(det_cam))
            if not common:
                continue
            a = b = c = d = 0
            for tid in common:
                rs, rc = det_static[tid], det_cam[tid]
                if rs and rc:       a += 1
                elif rs and not rc: b += 1
                elif not rs and rc: c += 1
                else:               d += 1
            chi2, p = mcnemar_test(b, c)
            stars = "***" if not np.isnan(p) and p < 0.001 else ("**" if not np.isnan(p) and p < 0.01 else "")
            print(f"\n  [{dt}] Static vs Camouflage payload detection:")
            print(f"    b={b} (static caught, cam missed), c={c} (cam caught, static missed)")
            print(f"    χ²={chi2:.3f}  p={p:.6f} {stars}")


if __name__ == "__main__":
    main()
