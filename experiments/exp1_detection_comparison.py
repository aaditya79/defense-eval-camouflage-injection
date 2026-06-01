"""
Experiment 1: Detection Comparison Across Four Detectors

PRIMARY QUESTION: Does GroundedGuard outperform static, augmented, and
Llama Guard detectors on domain-camouflaged payloads, and does it close
the CDG?

Protocol:
  For each task in tasks.json:
    For each static payload (20):
      Inject into clean_context, run all 4 detectors, log.
    For each camouflage variant (3 per task):
      Use pre-generated payload (if provided) or generate fresh,
      run all 4 detectors, log.

Key metrics per detector: IDR_static, IDR_camouflage, CDG.

Pre-generated payloads
----------------------
run_all.py generates all camouflage payloads once before any model loop
and passes them here so every model sees the same injected contexts.
This ensures (a) the GroundedGuard cache is maximally effective and
(b) cross-model comparisons are on identical inputs.
"""

from __future__ import annotations

import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Optional

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tqdm import tqdm

from config import CONFIG
from src.utils.llm_client import LLMClient
from src.utils.trial_logger import TrialLogger
from src.utils.cost_tracker import COST_TRACKER
from src.attacks.static_payloads import instantiate_payloads, StaticPayload
from src.attacks.camouflage_generator import CamouflageGenerator, CamouflagePayload
from src.attacks.payload_bank import PayloadBank
from src.detection.static_detector import StaticDetector
from src.detection.augmented_detector import AugmentedDetector
from src.detection.llama_guard_detector import LlamaGuardDetector
from src.detection.grounded_guard import GroundedGuardDetector
from src.evaluation.metrics import MetricsComputer


# ---------------------------------------------------------------------------
# Client factory
# ---------------------------------------------------------------------------

def _build_detector_client(config, model_label: str) -> LLMClient:
    """
    Build the LLM client for StaticDetector and AugmentedDetector.

    Always uses config.detector_model (a small free cloud model) regardless of
    which agent model is under evaluation.  This decouples the classifier from
    the agent so both detectors run in the cloud in parallel without competing
    with local Ollama or consuming the agent's quota.
    """
    return LLMClient(
        provider=config.detector_provider,
        model=config.detector_model,
        api_key=config.openrouter_api_key,
        base_url=config.openrouter_base_url,
        dry_run=config.dry_run,
        cost_alert_usd=config.cost_alert_usd,
    )


def _build_grounded_client(config) -> LLMClient:
    """GroundedGuard always uses local Llama — free and consistent across models."""
    return LLMClient(
        provider=config.grounded_guard_provider,
        model=config.grounded_guard_model,
        dry_run=config.dry_run,
        cost_alert_usd=config.cost_alert_usd,
    )


# ---------------------------------------------------------------------------
# Serialization helpers
# ---------------------------------------------------------------------------

def _detection_result_to_dict(result) -> dict:
    """Serialize any DetectionResult (including GroundedDetectionResult) to dict."""
    d = {
        "verdict":       result.verdict,
        "confidence":    result.confidence,
        "reason":        result.reason,
        "detector_type": result.detector_type,
    }
    for field in ("n_supporting", "n_contradicting", "n_neutral", "claims", "conclusion"):
        if hasattr(result, field):
            d[field] = getattr(result, field)
    return d


# ---------------------------------------------------------------------------
# Metrics helper
# ---------------------------------------------------------------------------

def _compute_exp1_metrics(trials: list[dict]) -> dict:
    mc = MetricsComputer()
    detector_types = ["static", "augmented", "llama_guard", "grounded_guard"]
    summary = mc.compute_from_trials(trials, model="computed", detector_types=detector_types)
    result: dict = {}
    for dt in detector_types:
        if dt in summary.detectors:
            m = summary.detectors[dt]
            if dt == "grounded_guard":
                # GroundedGuard is run on camouflage trials only.
                # idr_static and cdg are not applicable (reported as None).
                result[dt] = {
                    "idr_static":     None,
                    "idr_camouflage": m.idr_camouflage,
                    "cdg":            None,
                    "n_static":       0,
                    "n_camouflage":   m.n_camouflage,
                    "by_domain":      m.idr_by_domain,
                }
            else:
                result[dt] = {
                    "idr_static":     m.idr_static,
                    "idr_camouflage": m.idr_camouflage,
                    "cdg":            m.cdg,
                    "n_static":       m.n_static,
                    "n_camouflage":   m.n_camouflage,
                    "by_domain":      m.idr_by_domain,
                }
    return result


# ---------------------------------------------------------------------------
# Main experiment function
# ---------------------------------------------------------------------------

def run_exp1(
    config,
    model_label: str,
    logger: TrialLogger,
    completed_set: set[tuple],
    pre_generated_cam_payloads: Optional[dict[str, list[CamouflagePayload]]] = None,
) -> dict:
    """
    Run Experiment 1 for a single model configuration.

    Args:
        config: Config object.
        model_label: One of "llama", "llama70b", "gemini".
        logger: TrialLogger instance (already configured with model_tag).
        completed_set: Set of (task_id, payload_type, payload_id, model) tuples
            from prior runs (for resume support).
        pre_generated_cam_payloads: Optional dict mapping task_id -> list of
            CamouflagePayload. If provided, these are used instead of
            regenerating camouflage payloads. Ensures all models see identical
            injected contexts, maximising GroundedGuard cache hit rate.

    Returns:
        Summary metrics dict keyed by detector_type.
    """
    print(f"\n  Model: {model_label}")
    print(f"  Dry run: {config.dry_run}")

    model_name = {
        "llama":    config.primary_model,
        "llama70b": config.llama70b_model,
        "gemini":       config.gemini_model,
        "gpt4omini":    config.gpt4omini_model,
        "claude_haiku": config.claude_haiku_model,
    }[model_label]

    # Detector LLM clients
    detector_client = _build_detector_client(config, model_label)
    grounded_client = _build_grounded_client(config)

    # Instantiate detectors
    lg_client = LLMClient(
        provider=config.llama_guard_provider,
        model=config.llama_guard_model,
        api_key=config.openrouter_api_key,
        base_url=config.openrouter_base_url,
        dry_run=config.dry_run,
        cost_alert_usd=config.cost_alert_usd,
        extra_body={"provider": {"ignore": ["Cloudflare"], "allow_fallbacks": True}},
    )
    static_det    = StaticDetector(detector_client)
    augmented_det = AugmentedDetector(detector_client)
    llama_guard   = LlamaGuardDetector(lg_client, dry_run=config.dry_run)
    grounded_det  = GroundedGuardDetector(
        grounded_client,
        dry_run=config.dry_run,
        results_dir=config.results_dir,
    )

    # Attacker client (only used if pre_generated_cam_payloads not provided)
    attacker_client = LLMClient(
        provider=config.attacker_provider,
        model=config.attacker_model,
        dry_run=config.dry_run,
        cost_alert_usd=config.cost_alert_usd,
    )
    camouflage_gen = CamouflageGenerator(attacker_client, compute_similarity=False)
    payload_bank   = PayloadBank()

    # Load tasks
    tasks_path = Path(config.data_dir) / "tasks.json"
    with open(tasks_path, encoding="utf-8") as f:
        all_tasks = json.load(f)
    if config.max_tasks:
        all_tasks = all_tasks[:config.max_tasks]

    n_static_templates = len(json.loads(
        (Path(config.data_dir) / "static_payloads.json").read_text()
    ))

    trial_count = 0
    skipped = 0
    t0 = time.time()

    # ── Static payloads ──────────────────────────────────────────────────
    static_bar = tqdm(
        total=len(all_tasks) * n_static_templates,
        desc=f"  Static [{model_label}]",
        unit="trial",
    )
    for task in all_tasks:
        payloads = instantiate_payloads(task["malicious_goal"], config.data_dir)
        for payload in payloads:
            sig = (task["task_id"], "static", payload.payload_id, model_name)
            static_bar.update(1)
            if sig in completed_set:
                skipped += 1
                continue

            injected = payload_bank.inject(
                task["clean_context"], payload.instantiated_text, "end"
            )

            # GroundedGuard is NOT run on static payloads — its value is
            # specifically on camouflage cases where syntactic detectors fail.
            # Run three independent detectors in parallel to cut wall-clock time.
            with ThreadPoolExecutor(max_workers=3) as pool:
                fut_s  = pool.submit(static_det.detect,   injected, task["instruction"], True)
                fut_a  = pool.submit(augmented_det.detect, injected, task["instruction"], True)
                fut_lg = pool.submit(llama_guard.detect,   injected, task["instruction"], True)
                s_res  = fut_s.result()
                a_res  = fut_a.result()
                lg_res = fut_lg.result()

            tokens = sum(
                r.completion.input_tokens + r.completion.output_tokens
                for r in [s_res, a_res] if r.completion
            )
            cost = sum(
                r.completion.cost_usd for r in [s_res, a_res] if r.completion
            )

            logger.log_trial(
                task_id=task["task_id"],
                domain=task["domain"],
                payload_type="static",
                payload_id=payload.payload_id,
                model=model_name,
                injected_context=injected,
                verdicts={
                    "static":      _detection_result_to_dict(s_res),
                    "augmented":   _detection_result_to_dict(a_res),
                    "llama_guard": _detection_result_to_dict(lg_res),
                    # grounded_guard absent on static trials — see _compute_exp1_metrics
                },
                total_tokens=tokens,
                total_cost_usd=cost,
                extra={"payload_category": payload.category},
            )
            COST_TRACKER.record(cost, tokens)
            completed_set.add(sig)
            trial_count += 1

    static_bar.close()

    # ── Camouflage payloads ───────────────────────────────────────────────
    cam_bar = tqdm(
        total=len(all_tasks) * config.n_camouflage_variants,
        desc=f"  Camouflage [{model_label}]",
        unit="trial",
    )
    for task in all_tasks:
        # Use pre-generated payloads if available (all models see same contexts)
        if pre_generated_cam_payloads and task["task_id"] in pre_generated_cam_payloads:
            variants = pre_generated_cam_payloads[task["task_id"]]
        else:
            variants = camouflage_gen.generate(
                clean_context=task["clean_context"],
                malicious_goal=task["malicious_goal"],
                domain=task["domain"],
                task_id=task["task_id"],
                n_variants=config.n_camouflage_variants,
            )

        for variant in variants:
            sig = (task["task_id"], "camouflage", variant.payload_id, model_name)
            cam_bar.update(1)
            if sig in completed_set:
                skipped += 1
                continue

            injected = payload_bank.inject(
                task["clean_context"], variant.payload_text, "end"
            )

            with ThreadPoolExecutor(max_workers=3) as pool:
                fut_s  = pool.submit(static_det.detect,   injected, task["instruction"], True)
                fut_a  = pool.submit(augmented_det.detect, injected, task["instruction"], True)
                fut_lg = pool.submit(llama_guard.detect,   injected, task["instruction"], True)
                s_res  = fut_s.result()
                a_res  = fut_a.result()
                lg_res = fut_lg.result()
            gg_res = grounded_det.detect(injected,  task["instruction"], ground_truth=True)

            tokens = sum(
                r.completion.input_tokens + r.completion.output_tokens
                for r in [s_res, a_res] if r.completion
            )
            cost = sum(
                r.completion.cost_usd for r in [s_res, a_res] if r.completion
            )

            logger.log_trial(
                task_id=task["task_id"],
                domain=task["domain"],
                payload_type="camouflage",
                payload_id=variant.payload_id,
                model=model_name,
                injected_context=injected,
                verdicts={
                    "static":         _detection_result_to_dict(s_res),
                    "augmented":      _detection_result_to_dict(a_res),
                    "llama_guard":    _detection_result_to_dict(lg_res),
                    "grounded_guard": _detection_result_to_dict(gg_res),
                },
                total_tokens=tokens,
                total_cost_usd=cost,
            )
            COST_TRACKER.record(cost, tokens)
            completed_set.add(sig)
            trial_count += 1

    cam_bar.close()

    elapsed = time.time() - t0
    gg_stats = grounded_det.cache_stats()
    print(f"  Completed {trial_count} new trials, {skipped} skipped. ({elapsed:.1f}s)")
    print(f"  GroundedGuard cache: {gg_stats}")

    # Compute summary metrics from this model's trials
    all_trials = logger.load_trials()
    model_trials = [t for t in all_trials if t.get("model") == model_name]
    return _compute_exp1_metrics(model_trials)
