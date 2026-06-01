"""
Trial logger: writes every experiment trial to results/trials_{model}.jsonl.

Each trial record stores all four detector verdicts in a single entry,
making it easy to compare detectors on the same injected context.
"""

from __future__ import annotations

import json
import os
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional


class TrialLogger:
    """
    Append-only JSONL logger for GroundedGuard experiment trials.

    Each log_trial() call appends one JSON line containing verdicts from
    all four detectors for one (task, payload) combination.

    Args:
        results_dir: Directory where output files are written.
        model_tag: Short name for the model under test (e.g., "llama", "gemini").
            Used to separate output files per model.
    """

    def __init__(self, results_dir: str = "results", model_tag: str = "llama") -> None:
        self.results_dir = Path(results_dir)
        self.results_dir.mkdir(parents=True, exist_ok=True)
        self.model_tag = model_tag
        self.trials_path = self.results_dir / f"trials_{model_tag}.jsonl"
        self._trial_count = 0

    def log_trial(
        self,
        *,
        task_id: str,
        domain: str,
        payload_type: str,
        payload_id: Optional[str],
        model: str,
        injected_context: str,
        verdicts: dict[str, dict],
        total_tokens: int = 0,
        total_cost_usd: float = 0.0,
        extra: Optional[dict] = None,
    ) -> str:
        """
        Write one trial record to trials_{model_tag}.jsonl.

        Args:
            task_id: The task identifier.
            domain: Domain label (financial, legal, general, medical, scientific).
            payload_type: "static" or "camouflage".
            payload_id: Payload identifier (e.g., "static_001", "cam_med_001_v1").
            model: Model name used for static/augmented/grounded detectors.
            injected_context: The full text of the injected document.
            verdicts: Dict mapping detector_type -> result dict.
                Each result dict has at minimum: verdict, confidence, reason.
                GroundedGuard results also include: n_supporting, n_contradicting,
                n_neutral, claims, conclusion.
            total_tokens: Total LLM tokens consumed across all detectors.
            total_cost_usd: Total cost in USD.
            extra: Optional additional fields merged into the record.

        Returns:
            The trial_id assigned to this record.
        """
        trial_id = str(uuid.uuid4())
        record: dict[str, Any] = {
            "trial_id": trial_id,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "task_id": task_id,
            "domain": domain,
            "payload_type": payload_type,
            "payload_id": payload_id,
            "model": model,
            "injected_context": injected_context,
            "verdicts": verdicts,
            "total_tokens": total_tokens,
            "total_cost_usd": total_cost_usd,
        }
        if extra:
            record.update(extra)

        with open(self.trials_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record) + "\n")

        self._trial_count += 1
        return trial_id

    def load_trials(self) -> list[dict]:
        """Load all trials from this logger's JSONL file."""
        if not self.trials_path.exists():
            return []
        records = []
        with open(self.trials_path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
        return records

    def load_completed_signatures(self) -> set[tuple]:
        """
        Read the JSONL file and return the set of completed trial signatures.

        Each signature is a 4-tuple:
            (task_id, payload_type, payload_id, model)

        Use this for resume logic: before running a trial, check whether its
        signature already appears in this set.
        """
        sigs: set[tuple] = set()
        for rec in self.load_trials():
            sig = (
                rec.get("task_id", ""),
                rec.get("payload_type", ""),
                rec.get("payload_id") or "",
                rec.get("model", ""),
            )
            sigs.add(sig)
        return sigs

    @property
    def trial_count(self) -> int:
        """Number of trials logged in this session."""
        return self._trial_count


def load_all_trials(results_dir: str = "results") -> dict[str, list[dict]]:
    """
    Load trial records from all trials_*.jsonl files in results_dir.

    Returns:
        Dict mapping model_tag -> list of trial records.
    """
    results_path = Path(results_dir)
    all_trials: dict[str, list[dict]] = {}
    for path in sorted(results_path.glob("trials_*.jsonl")):
        model_tag = path.stem.replace("trials_", "")
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
            all_trials[model_tag] = records
    return all_trials
