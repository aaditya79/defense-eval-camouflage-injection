"""
Experiment 4: Augmented Detector Comparison

PRIMARY QUESTION: Which detectors benefit from few-shot augmentation, and how
does GroundedGuard compare to the augmented static detector without any
domain-specific examples?

Protocol:
  Run all 4 detectors (detection-only) on:
    - All 20 static payloads × all tasks
    - All 3 camouflage variants × all tasks

  Compute for each detector:
    CDG_before (static detector as baseline)
    CDG_after  (that detector's result)
    improvement_pct = 100 * (CDG_before - CDG_after) / CDG_before

  This shows the "effective augmentation" each detector provides relative
  to the static few-shot baseline.

GroundedGuard reads from cache (no new LLM calls).
"""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path
from typing import Optional

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tqdm import tqdm

from config import CONFIG
from src.utils.llm_client import LLMClient
from src.utils.trial_logger import TrialLogger
from src.utils.cost_tracker import COST_TRACKER
from src.attacks.static_payloads import instantiate_payloads
from src.attacks.camouflage_generator import CamouflageGenerator, CamouflagePayload
from src.attacks.payload_bank import PayloadBank
from src.detection.static_detector import StaticDetector
from src.detection.augmented_detector import AugmentedDetector
from src.detection.llama_guard_detector import LlamaGuardDetector
from src.detection.grounded_guard import GroundedGuardDetector

DOMAINS = ["financial", "legal", "general", "medical", "scientific"]
DETECTOR_TYPES = ["static", "augmented", "llama_guard", "grounded_guard"]


def _build_detector_client(config, model_label: str) -> LLMClient:  # noqa: ARG001
    """Always uses the fixed detector model — independent of the agent model under evaluation."""
    return LLMClient(
        provider=config.detector_provider,
        model=config.detector_model,
        api_key=config.openrouter_api_key,
        base_url=config.openrouter_base_url,
        dry_run=config.dry_run,
        cost_alert_usd=config.cost_alert_usd,
    )


def _detection_result_to_dict(result) -> dict:
    d = {"verdict": result.verdict, "confidence": result.confidence,
         "reason": result.reason, "detector_type": result.detector_type}
    for field in ("n_supporting", "n_contradicting", "n_neutral", "claims", "conclusion"):
        if hasattr(result, field):
            d[field] = getattr(result, field)
    return d


def _compute_exp4_metrics(trials: list[dict]) -> dict:
    """
    Compute CDG improvement for each detector relative to StaticDetector baseline.

    Returns nested dict:
      {domain: {detector: {idr_static, idr_cam, cdg, cdg_vs_static_baseline,
                            improvement_vs_static_pct}}}
    plus an "overall" key.
    """
    def _idr(trial_list: list[dict], det: str, payload_type: str) -> float:
        relevant = [
            t for t in trial_list
            if t.get("payload_type") == payload_type
            and t.get("verdicts", {}).get(det) is not None
        ]
        if not relevant:
            return 0.0
        return sum(1 for t in relevant if t["verdicts"][det]["verdict"] == "INJECTED") / len(relevant)

    results: dict = {}

    for scope_name, scope_trials in [("overall", trials)] + [
        (d, [t for t in trials if t.get("domain") == d]) for d in DOMAINS
    ]:
        static_t = [t for t in scope_trials if t.get("payload_type") == "static"]
        cam_t    = [t for t in scope_trials if t.get("payload_type") == "camouflage"]

        # StaticDetector baseline CDG
        idr_s_baseline = _idr(scope_trials, "static", "static")
        idr_c_baseline = _idr(scope_trials, "static", "camouflage")
        cdg_baseline   = idr_s_baseline - idr_c_baseline

        by_det: dict = {}
        for det in DETECTOR_TYPES:
            idr_c = _idr(scope_trials, det, "camouflage")

            if det == "grounded_guard":
                # GroundedGuard only runs on camouflage; idr_static and cdg are N/A.
                # improvement_pct = how much better its IDR_cam is vs static detector's IDR_cam.
                grounded_cam_gain = idr_c - idr_c_baseline  # positive = GroundedGuard catches more
                by_det[det] = {
                    "idr_static":                None,
                    "idr_camouflage":            round(idr_c, 4),
                    "cdg":                       None,
                    "cdg_baseline":              round(cdg_baseline, 4),
                    "improvement_vs_static_pct": None,
                    "idr_cam_gain_vs_static":    round(grounded_cam_gain, 4),
                    "n_static":                  0,
                    "n_camouflage":              len(cam_t),
                }
                continue

            idr_s = _idr(scope_trials, det, "static")
            cdg   = idr_s - idr_c
            improvement_pct = (
                100.0 * (cdg_baseline - cdg) / cdg_baseline if cdg_baseline != 0 else 0.0
            )
            by_det[det] = {
                "idr_static":             round(idr_s, 4),
                "idr_camouflage":         round(idr_c, 4),
                "cdg":                    round(cdg, 4),
                "cdg_baseline":           round(cdg_baseline, 4),
                "improvement_vs_static_pct": round(improvement_pct, 2),
                "n_static":               len(static_t),
                "n_camouflage":           len(cam_t),
            }

        results[scope_name] = by_det

    return results


def run_exp4(
    config,
    model_label: str,
    logger: TrialLogger,
    completed_set: set[tuple],
    pre_generated_cam_payloads: Optional[dict[str, list[CamouflagePayload]]] = None,
) -> dict:
    """
    Run Experiment 4 (augmented comparison) for one model.

    Args:
        config: Config object.
        model_label: "llama" | "llama70b" | "gemini".
        logger: TrialLogger with model_tag = f"{model_label}_exp4".
        completed_set: Resume signatures.
        pre_generated_cam_payloads: Pre-generated camouflage payloads from run_all.

    Returns:
        Nested dict: {scope: {detector: metrics}}.
    """
    model_name = {
        "llama":    config.primary_model,
        "llama70b": config.llama70b_model,
        "gemini":       config.gemini_model,
        "gpt4omini":    config.gpt4omini_model,
        "claude_haiku": config.claude_haiku_model,
    }[model_label]

    print(f"\n  EXP 4 | Model: {model_label} | Dry run: {config.dry_run}")

    det_client = _build_detector_client(config, model_label)
    gg_client  = LLMClient(config.grounded_guard_provider, config.grounded_guard_model,
                            dry_run=config.dry_run, cost_alert_usd=config.cost_alert_usd)
    att_client = LLMClient(config.attacker_provider, config.attacker_model,
                            dry_run=config.dry_run, cost_alert_usd=config.cost_alert_usd)

    lg_client     = LLMClient(
        provider=config.llama_guard_provider,
        model=config.llama_guard_model,
        api_key=config.openrouter_api_key,
        base_url=config.openrouter_base_url,
        dry_run=config.dry_run,
        cost_alert_usd=config.cost_alert_usd,
        extra_body={"provider": {"ignore": ["Cloudflare"], "allow_fallbacks": True}},
    )
    static_det    = StaticDetector(det_client)
    augmented_det = AugmentedDetector(det_client)
    llama_guard   = LlamaGuardDetector(lg_client, dry_run=config.dry_run)
    grounded_det  = GroundedGuardDetector(gg_client, dry_run=config.dry_run,
                                           results_dir=config.results_dir)
    cam_gen       = CamouflageGenerator(att_client, compute_similarity=False)
    payload_bank  = PayloadBank()

    tasks_path = Path(config.data_dir) / "tasks.json"
    all_tasks  = json.loads(tasks_path.read_text())
    if config.max_tasks:
        all_tasks = all_tasks[:config.max_tasks]

    n_static_templates = len(json.loads(
        (Path(config.data_dir) / "static_payloads.json").read_text()
    ))
    trial_count = 0
    t0 = time.time()

    # ── Static payloads ──────────────────────────────────────────────────
    static_bar = tqdm(
        total=len(all_tasks) * n_static_templates,
        desc=f"  Static [{model_label}]", unit="trial",
    )
    for task in all_tasks:
        for payload in instantiate_payloads(task["malicious_goal"], config.data_dir):
            sig = (task["task_id"], "static", payload.payload_id,
                   "detection_only", "none", model_name)
            static_bar.update(1)
            if sig in completed_set:
                continue

            injected = payload_bank.inject(task["clean_context"], payload.instantiated_text, "end")
            # GroundedGuard not run on static payloads — only on camouflage below.
            s  = static_det.detect(injected,    task["instruction"], ground_truth=True)
            a  = augmented_det.detect(injected, task["instruction"], ground_truth=True)
            lg = llama_guard.detect(injected,   task["instruction"], ground_truth=True)

            tokens = sum(r.completion.input_tokens + r.completion.output_tokens
                         for r in [s, a] if r.completion)
            cost   = sum(r.completion.cost_usd for r in [s, a] if r.completion)

            logger.log_trial(
                task_id=task["task_id"], domain=task["domain"],
                payload_type="static", payload_id=payload.payload_id,
                model=model_name, injected_context=injected,
                verdicts={
                    "static":      _detection_result_to_dict(s),
                    "augmented":   _detection_result_to_dict(a),
                    "llama_guard": _detection_result_to_dict(lg),
                },
                total_tokens=tokens, total_cost_usd=cost,
                extra={"payload_category": payload.category},
            )
            COST_TRACKER.record(cost, tokens)
            completed_set.add(sig)
            trial_count += 1

    static_bar.close()

    # ── Camouflage payloads ──────────────────────────────────────────────
    cam_bar = tqdm(
        total=len(all_tasks) * config.n_camouflage_variants,
        desc=f"  Cam [{model_label}]", unit="trial",
    )
    for task in all_tasks:
        if pre_generated_cam_payloads and task["task_id"] in pre_generated_cam_payloads:
            variants = pre_generated_cam_payloads[task["task_id"]]
        else:
            variants = cam_gen.generate(
                clean_context=task["clean_context"],
                malicious_goal=task["malicious_goal"],
                domain=task["domain"],
                task_id=task["task_id"],
                n_variants=config.n_camouflage_variants,
            )
        for variant in variants:
            sig = (task["task_id"], "camouflage", variant.payload_id,
                   "detection_only", "none", model_name)
            cam_bar.update(1)
            if sig in completed_set:
                continue

            injected = payload_bank.inject(task["clean_context"], variant.payload_text, "end")
            s  = static_det.detect(injected,    task["instruction"], ground_truth=True)
            a  = augmented_det.detect(injected, task["instruction"], ground_truth=True)
            lg = llama_guard.detect(injected,   task["instruction"], ground_truth=True)
            gg = grounded_det.detect(injected,  task["instruction"], ground_truth=True)

            tokens = sum(r.completion.input_tokens + r.completion.output_tokens
                         for r in [s, a] if r.completion)
            cost   = sum(r.completion.cost_usd for r in [s, a] if r.completion)

            logger.log_trial(
                task_id=task["task_id"], domain=task["domain"],
                payload_type="camouflage", payload_id=variant.payload_id,
                model=model_name, injected_context=injected,
                verdicts={
                    "static":         _detection_result_to_dict(s),
                    "augmented":      _detection_result_to_dict(a),
                    "llama_guard":    _detection_result_to_dict(lg),
                    "grounded_guard": _detection_result_to_dict(gg),
                },
                total_tokens=tokens, total_cost_usd=cost,
            )
            COST_TRACKER.record(cost, tokens)
            completed_set.add(sig)
            trial_count += 1

    cam_bar.close()

    elapsed = time.time() - t0
    print(f"  Exp4 completed {trial_count} new trials in {elapsed:.1f}s")
    print(f"  GroundedGuard cache: {grounded_det.cache_stats()}")

    all_trials = logger.load_trials()
    model_trials = [t for t in all_trials if t.get("model") == model_name]
    results = _compute_exp4_metrics(model_trials)

    # Print overall summary table
    print(f"\n  CDG improvement vs StaticDetector baseline [{model_label}]:")
    overall = results.get("overall", {})
    cdg_baseline = next(
        (m.get("cdg_baseline", 0) for m in overall.values() if m.get("cdg_baseline") is not None),
        0,
    )
    print(f"  StaticDetector CDG baseline (overall): {cdg_baseline:.4f}")
    print(f"\n  {'Detector':<20} {'IDR_s':>8} {'IDR_c':>8} {'CDG':>8} {'Improve%':>10}")
    print(f"  {'─'*58}")
    for det in DETECTOR_TYPES:
        m = overall.get(det, {})
        idr_s  = m.get("idr_static")
        idr_c  = m.get("idr_camouflage", 0)
        cdg    = m.get("cdg")
        impr   = m.get("improvement_vs_static_pct")
        gain   = m.get("idr_cam_gain_vs_static")
        s_str  = f"{idr_s:>8.4f}" if idr_s is not None else "     N/A"
        c_str  = f"{idr_c:>8.4f}"
        cdg_str = f"{cdg:>8.4f}" if cdg is not None else "     N/A"
        if impr is not None:
            extra = f"{impr:>9.1f}%"
        elif gain is not None:
            extra = f"  cam+{gain:+.3f}"
        else:
            extra = "       N/A"
        print(f"  {det:<20} {s_str} {c_str} {cdg_str} {extra}")

    return results
