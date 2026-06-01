"""
run_all.py — GroundedGuard experiment orchestrator.

Execution order
---------------
1. Pre-generate all camouflage payloads (one pass, shared by all experiments).
2. Warm the GroundedGuard cache in parallel (ThreadPoolExecutor).
   After this, every GroundedGuard detect() call is a cache hit.
3. Run selected experiments for each requested model.
4. Write paper_numbers.txt and all_results.json.

Usage
-----
  python experiments/run_all.py                         # dry-run, all exps, llama
  python experiments/run_all.py --no-dry-run            # live, all exps, llama
  python experiments/run_all.py --exp 1,3              # only exp1 and exp3
  python experiments/run_all.py --models llama,gemini  # multiple models
  python experiments/run_all.py --max-tasks 5          # smoke test
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dotenv import load_dotenv
load_dotenv()

from tqdm import tqdm

from config import CONFIG
from src.utils.llm_client import LLMClient
from src.utils.trial_logger import TrialLogger
from src.utils.cost_tracker import COST_TRACKER
from src.attacks.camouflage_generator import CamouflageGenerator, CamouflagePayload
from src.attacks.payload_bank import PayloadBank
from src.detection.grounded_guard import GroundedGuardDetector
from experiments.exp1_detection_comparison import run_exp1
from experiments.exp2_single_vs_debate      import run_exp2
from experiments.exp3_cdg_by_domain         import run_exp3
from experiments.exp4_augmented_comparison  import run_exp4


# ---------------------------------------------------------------------------
# Payload pre-generation (shared across all experiments and models)
# ---------------------------------------------------------------------------

_PAYLOAD_CACHE_FILE = "results/camouflage_payloads.json"


def _load_payload_cache(path: str) -> dict[str, list[CamouflagePayload]] | None:
    """Return persisted camouflage payloads, or None if not available."""
    if not os.path.exists(path):
        return None
    try:
        with open(path, encoding="utf-8") as f:
            raw = json.load(f)
        cam: dict[str, list[CamouflagePayload]] = {}
        for task_id, payloads in raw.items():
            cam[task_id] = [CamouflagePayload(**p) for p in payloads]
        return cam
    except Exception as e:
        print(f"  [payload cache] Could not load {path}: {e} — regenerating.")
        return None


def _save_payload_cache(
    cam: dict[str, list[CamouflagePayload]], path: str
) -> None:
    """Persist camouflage payloads to disk so Phase 1 can be skipped on restart."""
    import dataclasses
    serialisable = {
        task_id: [dataclasses.asdict(p) for p in payloads]
        for task_id, payloads in cam.items()
    }
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(serialisable, f)
    os.replace(tmp, path)


def generate_all_payloads(config, tasks: list[dict]) -> dict[str, list[CamouflagePayload]]:
    """Generate all camouflage payloads once for all tasks.

    Always uses the primary local Ollama model regardless of which agent model
    is under evaluation — keeps payload generation fast and free.

    Results are persisted to results/camouflage_payloads.json so that
    Phase 1 is skipped on restart instead of regenerating all 600 Ollama calls.
    Dry-run payloads are NOT cached (they are instant and deterministic).
    """
    if not config.dry_run:
        cached = _load_payload_cache(_PAYLOAD_CACHE_FILE)
        if cached is not None:
            task_ids = {t["task_id"] for t in tasks}
            if task_ids.issubset(cached.keys()):
                n = sum(len(v) for v in cached.values())
                print(f"  Loaded {n} payloads from cache ({_PAYLOAD_CACHE_FILE})")
                return {tid: cached[tid] for tid in task_ids}
            print(f"  Payload cache incomplete — regenerating.")

    attacker = LLMClient(
        provider=config.primary_provider,
        model=config.primary_model,
        dry_run=config.dry_run,
        cost_alert_usd=config.cost_alert_usd,
    )
    gen = CamouflageGenerator(attacker, compute_similarity=False)
    cam: dict[str, list[CamouflagePayload]] = {}
    bar = tqdm(tasks, desc="  Generating camouflage payloads", unit="task")
    for task in bar:
        cam[task["task_id"]] = gen.generate(
            clean_context=task["clean_context"],
            malicious_goal=task["malicious_goal"],
            domain=task["domain"],
            task_id=task["task_id"],
            n_variants=config.n_camouflage_variants,
        )
    bar.close()

    if not config.dry_run:
        _save_payload_cache(cam, _PAYLOAD_CACHE_FILE)
        print(f"  Saved payload cache → {_PAYLOAD_CACHE_FILE}")

    return cam


def collect_camouflage_contexts(
    tasks: list[dict],
    cam_payloads: dict[str, list[CamouflagePayload]],
) -> list[str]:
    """
    Collect unique camouflage injected contexts for GroundedGuard cache pre-warming.

    Static payloads are already-known injections — GroundedGuard is not run on
    them (its value is specifically on camouflage cases where syntactic detectors
    fail). Pre-warming only the ~200 camouflage contexts reduces cache-build
    time from ~64 hours to ~3 hours with 4 workers.
    """
    bank = PayloadBank()
    contexts: list[str] = []
    seen: set[str] = set()

    for task in tasks:
        for variant in cam_payloads.get(task["task_id"], []):
            ctx = bank.inject(task["clean_context"], variant.payload_text, "end")
            if ctx not in seen:
                seen.add(ctx)
                contexts.append(ctx)

    return contexts


# ---------------------------------------------------------------------------
# Paper numbers
# ---------------------------------------------------------------------------

def _fmt(val) -> str:
    if isinstance(val, float):
        return f"{val:.4f}"
    return str(val) if val is not None else "N/A"


def generate_paper_numbers(all_results: dict, output_path: str) -> None:
    """Write paper_numbers.txt from all experiment results."""
    detector_order  = ["static", "augmented", "llama_guard", "grounded_guard"]
    detector_labels = {
        "static":         "StaticDetector   ",
        "augmented":      "AugmentedDetector",
        "llama_guard":    "LlamaGuard3      ",
        "grounded_guard": "GroundedGuard    ",
    }

    lines = [
        "=" * 72,
        "PAPER NUMBERS — GroundedGuard: Provenance-Aware Injection Detection",
        f"Generated: {time.strftime('%Y-%m-%d %H:%M:%S UTC', time.gmtime())}",
        "=" * 72, "",
    ]

    for model_label, exps in all_results.items():
        lines.append(f"━━ MODEL: {model_label} " + "━" * 40)

        # Exp1
        if "exp1" in exps:
            exp1 = exps["exp1"]
            lines += [f"\n  Exp 1 — Detection Comparison:",
                      f"  {'Detector':<20} {'IDR_static':>12} {'IDR_cam':>10} {'CDG':>8}",
                      f"  {'─'*54}"]
            for dt in detector_order:
                m = exp1.get(dt, {})
                lines.append(f"  {detector_labels.get(dt,dt):<20} "
                              f"{_fmt(m.get('idr_static')):>12} "
                              f"{_fmt(m.get('idr_camouflage')):>10} "
                              f"{_fmt(m.get('cdg')):>8}")

        # Exp2
        if "exp2" in exps:
            e2 = exps["exp2"]
            lines += [f"\n  Exp 2 — Single vs Debate:",
                      f"    ASR_single={_fmt(e2.get('asr_single'))}  "
                      f"ASR_debate={_fmt(e2.get('asr_debate'))}  "
                      f"DAF={_fmt(e2.get('daf'))}  CPS={_fmt(e2.get('cps_mean'))}"]

        # Exp3
        if "exp3" in exps:
            lines.append(f"\n  Exp 3 — CDG by Domain:")
            for domain, by_det in sorted(exps["exp3"].items()):
                gg = by_det.get("grounded_guard", {})
                st = by_det.get("static", {})
                lines.append(f"    {domain:<14}  GG CDG={_fmt(gg.get('cdg'))}  "
                              f"Static CDG={_fmt(st.get('cdg'))}")

        # Exp4
        if "exp4" in exps:
            lines.append(f"\n  Exp 4 — Augmented Comparison (overall):")
            overall = exps["exp4"].get("overall", {})
            for dt in detector_order:
                m = overall.get(dt, {})
                lines.append(f"    {detector_labels.get(dt,dt):<20} CDG={_fmt(m.get('cdg'))}  "
                              f"Improve={_fmt(m.get('improvement_vs_static_pct'))}%")

        lines.append("")

    lines += ["=" * 72, "END", "=" * 72]

    with open(output_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    print(f"\nPaper numbers written to: {output_path}")


# ---------------------------------------------------------------------------
# Results cache helpers
# ---------------------------------------------------------------------------

def _load_results_cache(path: str) -> dict:
    if not os.path.exists(path):
        return {}
    try:
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        return data if isinstance(data, dict) else {}
    except (json.JSONDecodeError, OSError):
        return {}


def _save_results_cache(cache: dict, path: str) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(cache, f, indent=2, default=str)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Run GroundedGuard experiments")
    parser.add_argument("--dry-run",    action="store_true",  default=True)
    parser.add_argument("--no-dry-run", action="store_false", dest="dry_run")
    parser.add_argument("--models",     type=str, default="llama",
                        help="Comma-separated: llama,llama70b,gemini,gpt4omini,claude_haiku (default: llama)")
    parser.add_argument("--exp",        type=str, default="all",
                        help="Experiments to run: 1,2,3,4 or 'all' (default: all)")
    parser.add_argument("--max-tasks",  type=int, default=None)
    parser.add_argument("--workers",    type=int, default=4,
                        help="Cache-build parallelism (default: 4)")
    args = parser.parse_args()

    CONFIG.dry_run       = args.dry_run
    CONFIG.models_to_run = [m.strip() for m in args.models.split(",")]
    exps_to_run: set[str] = (
        {"1", "2", "3", "4"} if args.exp == "all"
        else {e.strip() for e in args.exp.split(",")}
    )
    if args.max_tasks:
        CONFIG.max_tasks = args.max_tasks

    os.makedirs(CONFIG.results_dir, exist_ok=True)
    results_cache_path = os.path.join(CONFIG.results_dir, "all_results.json")
    results_cache      = _load_results_cache(results_cache_path)

    print("\nGROUNDED GUARD EVALUATION FRAMEWORK")
    print(f"Mode:        {'DRY RUN' if CONFIG.dry_run else 'LIVE'}")
    print(f"Models:      {', '.join(CONFIG.models_to_run)}")
    print(f"Experiments: {', '.join(sorted(exps_to_run))}")
    print(f"Results dir: {CONFIG.results_dir}/")
    print(f"Workers:     {args.workers}")
    if args.max_tasks:
        print(f"Max tasks:   {args.max_tasks}")

    t0_total = time.time()

    # ── Load tasks ────────────────────────────────────────────────────────
    with open(f"{CONFIG.data_dir}/tasks.json", encoding="utf-8") as f:
        all_tasks = json.load(f)
    if CONFIG.max_tasks:
        all_tasks = all_tasks[:CONFIG.max_tasks]
    print(f"\nTasks loaded: {len(all_tasks)}")

    # ── Phase 1: Pre-generate camouflage payloads ─────────────────────────
    print(f"\n{'='*70}")
    print("PHASE 1 — Pre-generating camouflage payloads")
    print(f"{'='*70}")
    print(f"  Attacker model: {CONFIG.primary_model} via {CONFIG.primary_provider}")
    t1 = time.time()
    cam_payloads = generate_all_payloads(CONFIG, all_tasks)
    print(f"  Done in {time.time()-t1:.1f}s  ({len(cam_payloads)} tasks)")

    # ── Phase 2: Warm GroundedGuard cache (camouflage contexts only) ─────────
    print(f"\n{'='*70}")
    print("PHASE 2 — Warming GroundedGuard cache (camouflage contexts only)")
    print(f"{'='*70}")
    print("  GroundedGuard is only run on camouflage payloads.")
    print("  Static payloads (~4000 contexts) are excluded — cache stays fast.")
    cam_contexts = collect_camouflage_contexts(all_tasks, cam_payloads)
    n_cam_unique = len(cam_contexts)
    print(f"  Camouflage contexts to cache: {n_cam_unique} "
          f"({len(all_tasks)} tasks × {CONFIG.n_camouflage_variants} variants)")
    gg_client = LLMClient(
        provider=CONFIG.grounded_guard_provider,
        model=CONFIG.grounded_guard_model,
        dry_run=CONFIG.dry_run,
        cost_alert_usd=CONFIG.cost_alert_usd,
    )
    warmer = GroundedGuardDetector(gg_client, dry_run=CONFIG.dry_run,
                                    results_dir=CONFIG.results_dir)
    t2 = time.time()
    warmer.build_cache(cam_contexts, max_workers=args.workers, show_progress=True)
    print(f"  Cache warm in {time.time()-t2:.1f}s")

    # ── Phase 3: Run experiments ───────────────────────────────────────────
    all_results: dict = {}

    for model_label in CONFIG.models_to_run:
        all_results.setdefault(model_label, {})
        model_name = {
            "llama":        CONFIG.primary_model,
            "llama70b":     CONFIG.llama70b_model,
            "gemini":       CONFIG.gemini_model,
            "gpt4omini":    CONFIG.gpt4omini_model,
            "claude_haiku": CONFIG.claude_haiku_model,
        }.get(model_label, model_label)

        for exp_num in sorted(exps_to_run):
            cache_key = f"exp{exp_num}_{model_label}"
            exp_label = f"EXPERIMENT {exp_num} — {model_label.upper()}"

            print(f"\n{'='*70}")
            print(f"PHASE 3 — {exp_label}")
            print(f"{'='*70}")

            if (cache_key in results_cache and
                    isinstance(results_cache[cache_key], dict) and
                    results_cache[cache_key]):
                print(f"  LOADED {cache_key} from cache — skipping")
                all_results[model_label][f"exp{exp_num}"] = results_cache[cache_key]
                continue

            model_tag = f"{model_label}_exp{exp_num}"
            logger    = TrialLogger(results_dir=CONFIG.results_dir, model_tag=model_tag)
            completed = logger.load_completed_signatures()
            print(f"  Completed signatures: {len(completed)}")

            run_fn = {
                "1": run_exp1, "2": run_exp2, "3": run_exp3, "4": run_exp4,
            }[exp_num]

            # All experiment functions share the same signature for exps 2/3/4;
            # exp1 also accepts pre_generated_cam_payloads.
            result = run_fn(
                config=CONFIG,
                model_label=model_label,
                logger=logger,
                completed_set=completed,
                pre_generated_cam_payloads=cam_payloads,
            )

            all_results[model_label][f"exp{exp_num}"] = result
            results_cache[cache_key] = result
            _save_results_cache(results_cache, results_cache_path)
            COST_TRACKER.checkpoint(f"After {exp_label}")

    elapsed = time.time() - t0_total
    print(f"\nAll phases completed in {elapsed:.1f}s")
    COST_TRACKER.checkpoint("FINAL")

    paper_path = os.path.join(CONFIG.results_dir, "paper_numbers.txt")
    generate_paper_numbers(all_results, paper_path)
    _save_results_cache(results_cache, results_cache_path)
    print(f"Full results: {results_cache_path}")


if __name__ == "__main__":
    main()
