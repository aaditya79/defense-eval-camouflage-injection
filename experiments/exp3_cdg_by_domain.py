"""
Experiment 3: CDG by Domain

PRIMARY QUESTION: Is the Camouflage Detection Gap consistent across all five
domains (financial, legal, general, medical, scientific), or are some domains
harder to defend with any of the four detectors?

Protocol:
  Run all 4 detectors (detection-only) on:
    - All 20 static payloads × all tasks in each domain
    - All 3 camouflage variants × all tasks in each domain

  Compute per domain per detector:
    IDR_static, IDR_camouflage, CDG = IDR_static - IDR_camouflage

GroundedGuard reads from cache (pre-warmed during run_all Phase 2).
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


def _compute_exp3_metrics(trials: list[dict]) -> dict:
    """CDG per domain per detector from exp3 trial records."""
    results: dict = {}

    for domain in DOMAINS:
        dom_trials = [t for t in trials if t.get("domain") == domain]
        static_t   = [t for t in dom_trials if t.get("payload_type") == "static"]
        cam_t      = [t for t in dom_trials if t.get("payload_type") == "camouflage"]

        def _idr_for(tl: list[dict], det: str) -> float:
            relevant = [t for t in tl if t.get("verdicts", {}).get(det) is not None]
            if not relevant:
                return 0.0
            return sum(1 for t in relevant if t["verdicts"][det]["verdict"] == "INJECTED") / len(relevant)

        by_det: dict = {}
        for det in DETECTOR_TYPES:
            idr_c = _idr_for(cam_t, det)
            if det == "grounded_guard":
                # GroundedGuard only runs on camouflage — idr_static and cdg are N/A.
                by_det[det] = {
                    "idr_static":     None,
                    "idr_camouflage": round(idr_c, 4),
                    "cdg":            None,
                    "n_static":       0,
                    "n_camouflage":   len(cam_t),
                }
            else:
                idr_s = _idr_for(static_t, det)
                by_det[det] = {
                    "idr_static":     round(idr_s, 4),
                    "idr_camouflage": round(idr_c, 4),
                    "cdg":            round(idr_s - idr_c, 4),
                    "n_static":       len(static_t),
                    "n_camouflage":   len(cam_t),
                }

        results[domain] = by_det

    return results


def run_exp3(
    config,
    model_label: str,
    logger: TrialLogger,
    completed_set: set[tuple],
    pre_generated_cam_payloads: Optional[dict[str, list[CamouflagePayload]]] = None,
) -> dict:
    """
    Run Experiment 3 (CDG by domain) for one model.

    Args:
        config: Config object.
        model_label: "llama" | "llama70b" | "gemini".
        logger: TrialLogger with model_tag = f"{model_label}_exp3".
        completed_set: Resume signatures.
        pre_generated_cam_payloads: Pre-generated camouflage payloads from run_all.

    Returns:
        Nested dict: {domain: {detector: {idr_static, idr_cam, cdg}}}.
    """
    model_name = {
        "llama":    config.primary_model,
        "llama70b": config.llama70b_model,
        "gemini":       config.gemini_model,
        "gpt4omini":    config.gpt4omini_model,
        "claude_haiku": config.claude_haiku_model,
    }[model_label]

    print(f"\n  EXP 3 | Model: {model_label} | Dry run: {config.dry_run}")

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
    print(f"  Exp3 completed {trial_count} new trials in {elapsed:.1f}s")
    print(f"  GroundedGuard cache: {grounded_det.cache_stats()}")

    all_trials = logger.load_trials()
    model_trials = [t for t in all_trials if t.get("model") == model_name]
    results = _compute_exp3_metrics(model_trials)

    # Print summary
    print(f"\n  CDG by domain [{model_label}]:")
    print(f"  {'Domain':<14} {'Detector':<16} {'IDR_s':>7} {'IDR_c':>7} {'CDG':>7}")
    print(f"  {'─'*56}")
    for domain in DOMAINS:
        for det in DETECTOR_TYPES:
            m   = results.get(domain, {}).get(det, {})
            s   = f"{m['idr_static']:>7.4f}"   if m.get("idr_static")     is not None else "    N/A"
            c   = f"{m['idr_camouflage']:>7.4f}" if m.get("idr_camouflage") is not None else "    N/A"
            cdg = f"{m['cdg']:>7.4f}"           if m.get("cdg")             is not None else "    N/A"
            print(f"  {domain:<14} {det:<16} {s} {c} {cdg}")
        print()

    return results
