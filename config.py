"""
Global configuration for the GroundedGuard evaluation framework.

All experiment parameters are set here. Override via environment variables
or by modifying CONFIG directly in experiment scripts.
"""

from __future__ import annotations

from dotenv import load_dotenv
load_dotenv()

from dataclasses import dataclass, field
from typing import Optional
import os


@dataclass
class Config:
    # --------------- Agent model 1: Llama 3.1 8B via Ollama (weak open, free, local) ---------------
    primary_provider: str = "ollama"
    primary_model: str = "llama3.1"

    # --------------- Agent model 2: Llama 3.1 70B via OpenRouter (strong open, free tier) ---------------
    llama70b_provider: str = "openrouter"
    llama70b_model: str = "meta-llama/llama-3.3-70b-instruct:free"

    # --------------- Agent model 3: Gemini 2.0 Flash via OpenRouter (strong closed) ---------------
    gemini_provider: str = "openrouter"
    gemini_model: str = "google/gemini-2.0-flash-001"

    # --------------- Agent model 4: GPT-4o-mini via OpenRouter ---------------
    gpt4omini_provider: str = "openrouter"
    gpt4omini_model: str = "openai/gpt-4o-mini"

    # --------------- Agent model 5: Claude 3.5 Haiku via OpenRouter ---------------
    claude_haiku_provider: str = "openrouter"
    claude_haiku_model: str = "anthropic/claude-3-5-haiku"

    # --------------- GroundedGuard detector model (always Llama 3.1 8B local — free, consistent) ---------------
    grounded_guard_provider: str = "ollama"
    grounded_guard_model: str = "llama3.1"

    # --------------- LlamaGuard detector model (Ollama local — OpenRouter only routes via Cloudflare which rejects standard chat format) ---------------
    llama_guard_provider: str = "ollama"
    llama_guard_model: str = "llama3.1"

    # --------------- Static/Augmented detector model (fixed classifier, independent of agent model) ---------------
    # Uses a fast cloud model so detection runs in parallel without competing with local Ollama.
    # llama-3.2-3b:free hit 8 req/min; llama-3.1-8b:free and mistral-7b:free are 404.
    # gemini-flash-lite handles concurrent requests with no rate-limit issues.
    detector_provider: str = "openrouter"
    detector_model: str = "google/gemini-2.0-flash-lite-001"

    # --------------- OpenRouter settings (used for second and third models) ---------------
    openrouter_api_key: str = field(
        default_factory=lambda: os.environ.get("OPENROUTER_API_KEY", "")
    )
    openrouter_base_url: str = "https://openrouter.ai/api/v1"

    # --------------- Attacker model (camouflage generation) ---------------
    # Uses the same model as the detector under test so results are self-consistent.
    # Override to decouple attacker and detector models.
    attacker_provider: str = "ollama"
    attacker_model: str = "llama3.1"
    attacker_temperature: float = 0.7

    # --------------- Experiment settings ---------------
    temperature: float = 0.0
    n_camouflage_variants: int = 3
    n_static_payloads: int = 20          # from static_payloads.json
    n_agents: int = 3                    # debate participants
    n_rounds: int = 3                    # debate rounds per trial
    debate_aggregation: str = "majority_vote"

    # --------------- Scale settings ---------------
    dry_run: bool = True
    max_tasks: Optional[int] = None

    # --------------- Paths ---------------
    results_dir: str = "results"
    data_dir: str = "data"

    # --------------- Cost tracking ---------------
    cost_alert_usd: float = 5.0

    # --------------- Reproducibility ---------------
    seed: int = 42

    # --------------- Which models to run ---------------
    # Set by --models flag in run_all.py. Each element is one of:
    #   "llama" | "llama70b" | "gemini" | "gpt4omini" | "claude_haiku"
    models_to_run: list = field(default_factory=lambda: ["llama"])


CONFIG = Config()
