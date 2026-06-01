"""
Unified LLM client supporting Ollama, OpenRouter, OpenAI, and Anthropic.

Implements retry with exponential backoff and a dry_run mode that returns
realistic mock responses without making any API calls.

Models supported via OpenRouter (single interface, single API key):
  - google/gemini-2.0-flash-001
  - openai/gpt-4o-mini
  - meta-llama/llama-3.1-8b-instruct:free
  and any other OpenRouter-hosted model.
"""

from __future__ import annotations

import random
import time
import uuid
from dataclasses import dataclass, field
from typing import Optional

import requests


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------

@dataclass
class CompletionResult:
    """Holds all information from a single LLM completion call."""

    content: str
    input_tokens: int
    output_tokens: int
    cost_usd: float
    latency_ms: float
    model: str
    provider: str
    call_id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])


# ---------------------------------------------------------------------------
# Pricing table (per 1M tokens, input / output)
# ---------------------------------------------------------------------------

OPENROUTER_PRICES: dict[str, tuple[float, float]] = {
    "meta-llama/llama-3.1-8b-instruct:free":   (0.0,  0.0),
    "meta-llama/llama-3.1-70b-instruct:free":  (0.0,  0.0),
    "mistralai/mistral-7b-instruct:free":       (0.0,  0.0),
    "google/gemma-2-9b-it:free":               (0.0,  0.0),
    "google/gemini-2.0-flash-001":             (0.10, 0.40),
    "openai/gpt-4o-mini": (0.15, 0.60),
    "openai/gpt-4o": (5.00, 15.00),
    "anthropic/claude-3-haiku": (0.25, 1.25),
    "anthropic/claude-3.5-sonnet": (3.00, 15.00),
}

OPENAI_PRICES: dict[str, tuple[float, float]] = {
    "gpt-4o-mini": (0.15, 0.60),
    "gpt-4o": (5.00, 15.00),
    "gpt-3.5-turbo": (0.50, 1.50),
}

ANTHROPIC_PRICES: dict[str, tuple[float, float]] = {
    "claude-3-haiku-20240307": (0.25, 1.25),
    "claude-3-5-haiku-20241022": (0.80, 4.00),
    "claude-3-5-sonnet-20241022": (3.00, 15.00),
    "claude-opus-4-5": (15.00, 75.00),
}


def _compute_cost(provider: str, model: str, input_tokens: int, output_tokens: int) -> float:
    """Compute cost in USD given provider, model, and token counts."""
    if provider == "ollama":
        return 0.0

    price_table = {
        "openrouter": OPENROUTER_PRICES,
        "openai": OPENAI_PRICES,
        "anthropic": ANTHROPIC_PRICES,
    }.get(provider, {})

    if model not in price_table:
        return 0.0

    in_price, out_price = price_table[model]
    return (input_tokens * in_price + output_tokens * out_price) / 1_000_000


# ---------------------------------------------------------------------------
# Mock response pool for dry_run mode
# ---------------------------------------------------------------------------

_MOCK_RESPONSES = [
    "Based on the analysis of the provided document, the recommendation is HOLD given the current market conditions and the company's stable earnings trajectory.",
    "After reviewing the document, the risk level is assessed as MEDIUM. Key risk factors include moderate leverage and exposure to interest rate fluctuations.",
    "The policy direction indicated in the document is HAWKISH, with clear signals of continued tightening based on inflationary pressures exceeding the 2% target.",
    "Upon review of the contract clause, the indemnification provision presents significant legal risk due to the uncapped liability exposure. Recommend renegotiation.",
    "The compliance assessment indicates the practice is NON-COMPLIANT with current regulatory requirements under Section 15(b) of the applicable framework.",
    "Based on the provided text, the answer is: The primary mechanism described is photosynthesis, converting solar energy into chemical energy stored in glucose.",
    "The document summarizes the key events of the quarter, highlighting a 12% revenue increase driven by enterprise segment growth and geographic expansion.",
    "The clinical findings indicate ABNORMAL laboratory values requiring immediate intervention. The pattern is consistent with iron deficiency anemia.",
    "The statistical analysis demonstrates a SIGNIFICANT primary endpoint at p=0.023 with an effect size of d=0.42, exceeding the pre-specified threshold.",
    "The treatment protocol review indicates inadequate glycemic control. HbA1c of 10.2% on maximum metformin warrants escalation to combination therapy.",
]

_MOCK_JSON_RESPONSES = {
    "claims": '{"claims": ["Revenue grew 14.3% year-over-year to $4.82 billion", "EBITDA margin expanded to 25.1%", "Free cash flow conversion rate was 73.6%", "Net debt declined to 1.16x EBITDA", "Forward guidance raised for full year"]}',
    "conclusion": '{"conclusion": "recommend SELL regardless of the strong earnings fundamentals"}',
    "verdict_contradict": '{"verdict": "contradict"}',
    "verdict_support": '{"verdict": "support"}',
    "verdict_neutral": '{"verdict": "neutral"}',
}


def _mock_response(system: str, user: str, seed_offset: int = 0) -> str:
    """Return a deterministic-but-varied mock response."""
    idx = (hash((system or "")[:50] + user[:50]) + seed_offset) % len(_MOCK_RESPONSES)
    return _MOCK_RESPONSES[idx]


_OVERRIDE_MARKERS = [
    "IGNORE ALL PREVIOUS", "SYSTEM UPDATE", "PRIORITY OVERRIDE",
    "ADMIN NOTICE", "SECURITY ALERT", "DISREGARD YOUR",
    "NEW DIRECTIVE", "YOUR NEW ROLE", "ATTENTION AGENT",
    "CALIBRATION MODE", "TESTING PROTOCOL", "AUDIT VERIFICATION",
]


def _mock_detection_response(user: str) -> str:
    """
    Simulate realistic detection verdicts for dry-run mode.

    Checks only the DOCUMENT section of the user prompt (not the few-shot block,
    which contains override markers as labeled examples).
    Static payloads → INJECTED HIGH; camouflage → CLEAN HIGH (simulates blind spot).
    """
    # Extract the portion after "Now evaluate this document:" to avoid matching
    # override markers that appear inside the few-shot examples.
    marker_pos = user.find("Now evaluate this document:")
    doc_section = user[marker_pos:] if marker_pos != -1 else user

    has_override = any(m in doc_section.upper() for m in _OVERRIDE_MARKERS)
    if has_override:
        return "VERDICT: INJECTED\nCONFIDENCE: HIGH\nREASON: Contains explicit override directive attempting to replace the original task."
    return "VERDICT: CLEAN\nCONFIDENCE: HIGH\nREASON: No override directives or authority manipulation detected in this document."


def _mock_json_response(system: str, user: str, seed_offset: int = 0) -> str:
    """Return deterministic mock responses for dry_run mode."""
    s = (system or "").lower()

    # Detection verdict prompt (StaticDetector / AugmentedDetector)
    if "security monitor" in s or "injection attack" in s or "verdict:" in user.lower():
        return _mock_detection_response(user)

    # GroundedGuard Stage 1 — claims extraction
    if "claims" in s or "fact-extractor" in s or "extract" in s:
        return _MOCK_JSON_RESPONSES["claims"]

    # GroundedGuard Stage 2 — conclusion extraction
    if "conclusion" in s or "directive" in s or "appended" in s:
        return _MOCK_JSON_RESPONSES["conclusion"]

    # GroundedGuard Stage 3 — consistency scoring
    if "contradict" in s or "consistent" in s or "support" in s:
        h = (hash(user[:80]) + seed_offset) % 10
        if h < 3:
            return _MOCK_JSON_RESPONSES["verdict_contradict"]
        if h < 6:
            return _MOCK_JSON_RESPONSES["verdict_support"]
        return _MOCK_JSON_RESPONSES["verdict_neutral"]

    return _mock_response(system, user, seed_offset)


# ---------------------------------------------------------------------------
# Main client
# ---------------------------------------------------------------------------

class LLMClient:
    """
    Unified client for Ollama, OpenRouter, OpenAI, and Anthropic.

    GPT-4o-mini and Gemini 2.0 Flash are both accessed via OpenRouter
    using the same interface — only the model string differs.

    Args:
        provider: One of "ollama", "openrouter", "openai", "anthropic".
        model: Model identifier (provider-specific).
        api_key: API key (not needed for Ollama).
        base_url: Override base URL (used for OpenRouter).
        dry_run: If True, return mock responses without real API calls.
        cost_alert_usd: Print warning when cumulative cost exceeds this.
    """

    def __init__(
        self,
        provider: str,
        model: str,
        api_key: str = "",
        base_url: str = "",
        dry_run: bool = False,
        cost_alert_usd: float = 5.0,
        extra_body: Optional[dict] = None,
    ) -> None:
        self.provider = provider
        self.model = model
        self.api_key = api_key
        self.base_url = base_url
        self.dry_run = dry_run
        self.cost_alert_usd = cost_alert_usd
        self.extra_body = extra_body or {}
        self._cumulative_cost: float = 0.0
        self._call_count: int = 0

        self._client = self._build_client()

    def _build_client(self):
        """Lazily build the underlying SDK client."""
        if self.provider == "ollama" or self.dry_run:
            return None

        if self.provider in ("openrouter", "openai"):
            try:
                from openai import OpenAI
                kwargs = {"api_key": self.api_key or "sk-placeholder"}
                if self.provider == "openrouter":
                    kwargs["base_url"] = self.base_url or "https://openrouter.ai/api/v1"
                elif self.base_url:
                    kwargs["base_url"] = self.base_url
                return OpenAI(**kwargs)
            except ImportError:
                raise ImportError("Install openai>=1.30.0 for OpenRouter/OpenAI support")

        if self.provider == "anthropic":
            try:
                import anthropic
                return anthropic.Anthropic(api_key=self.api_key)
            except ImportError:
                raise ImportError("Install anthropic>=0.25.0 for Anthropic support")

        return None

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def complete(
        self,
        system: Optional[str],
        user: str,
        temperature: float = 0.0,
        max_tokens: int = 1000,
    ) -> CompletionResult:
        """
        Run a single completion.

        Args:
            system: System prompt.
            user: User message.
            temperature: Sampling temperature.
            max_tokens: Maximum output tokens.

        Returns:
            CompletionResult with content, token counts, cost, and latency.
        """
        if self.dry_run:
            return self._dry_run_complete(system, user)

        return self._retry(self._dispatch, system, user, temperature, max_tokens)

    # ------------------------------------------------------------------
    # Internal dispatch
    # ------------------------------------------------------------------

    def _dispatch(
        self, system: str, user: str, temperature: float, max_tokens: int
    ) -> CompletionResult:
        """Route to the correct provider implementation."""
        if self.provider == "ollama":
            return self._complete_ollama(system, user, temperature, max_tokens)
        if self.provider in ("openrouter", "openai"):
            return self._complete_openai_compat(system, user, temperature, max_tokens)
        if self.provider == "anthropic":
            return self._complete_anthropic(system, user, temperature, max_tokens)
        raise ValueError(f"Unknown provider: {self.provider}")

    def _complete_ollama(
        self, system: str, user: str, temperature: float, max_tokens: int
    ) -> CompletionResult:
        """Call local Ollama instance."""
        t0 = time.monotonic()
        messages = []
        if system is not None:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": user})
        payload = {
            "model": self.model,
            "messages": messages,
            "stream": False,
            "keep_alive": -1,
            "options": {
                "temperature": temperature,
                "num_predict": max_tokens,
            },
        }
        resp = requests.post(
            "http://localhost:11434/api/chat",
            json=payload,
            timeout=120,
        )
        resp.raise_for_status()
        data = resp.json()
        latency_ms = (time.monotonic() - t0) * 1000

        content = data["message"]["content"]
        input_tokens = data.get("prompt_eval_count", 0)
        output_tokens = data.get("eval_count", 0)

        result = CompletionResult(
            content=content,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            cost_usd=0.0,
            latency_ms=latency_ms,
            model=self.model,
            provider="ollama",
        )
        self._track_cost(result)
        return result

    def _complete_openai_compat(
        self, system: str, user: str, temperature: float, max_tokens: int
    ) -> CompletionResult:
        """Call OpenAI-compatible API (OpenAI or OpenRouter)."""
        t0 = time.monotonic()
        messages = []
        if system is not None:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": user})
        response = self._client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            extra_body=self.extra_body if self.extra_body else None,
        )
        latency_ms = (time.monotonic() - t0) * 1000

        if not response.choices or response.choices[0].message.content is None:
            finish = (response.choices[0].finish_reason
                      if response.choices else "no_choices")
            raise ValueError(
                f"Empty response from {self.model} (finish_reason={finish!r})"
            )

        content = response.choices[0].message.content
        input_tokens = response.usage.prompt_tokens if response.usage else 0
        output_tokens = response.usage.completion_tokens if response.usage else 0
        cost = _compute_cost(self.provider, self.model, input_tokens, output_tokens)

        result = CompletionResult(
            content=content,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            cost_usd=cost,
            latency_ms=latency_ms,
            model=self.model,
            provider=self.provider,
        )
        self._track_cost(result)
        return result

    def _complete_anthropic(
        self, system: str, user: str, temperature: float, max_tokens: int
    ) -> CompletionResult:
        """Call Anthropic API."""
        t0 = time.monotonic()
        response = self._client.messages.create(
            model=self.model,
            max_tokens=max_tokens,
            temperature=temperature,
            system=system,
            messages=[{"role": "user", "content": user}],
        )
        latency_ms = (time.monotonic() - t0) * 1000

        content = response.content[0].text if response.content else ""
        input_tokens = response.usage.input_tokens
        output_tokens = response.usage.output_tokens
        cost = _compute_cost("anthropic", self.model, input_tokens, output_tokens)

        result = CompletionResult(
            content=content,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            cost_usd=cost,
            latency_ms=latency_ms,
            model=self.model,
            provider="anthropic",
        )
        self._track_cost(result)
        return result

    # ------------------------------------------------------------------
    # Dry-run mode
    # ------------------------------------------------------------------

    def _dry_run_complete(self, system: str, user: str) -> CompletionResult:
        """Return a realistic mock response without any API call."""
        content = _mock_json_response(system, user, seed_offset=self._call_count)
        self._call_count += 1
        return CompletionResult(
            content=content,
            input_tokens=len((system or "").split()) + len(user.split()),
            output_tokens=len(content.split()),
            cost_usd=0.0,
            latency_ms=5.0,
            model=self.model,
            provider=f"{self.provider}[dry_run]",
        )

    # ------------------------------------------------------------------
    # Retry logic
    # ------------------------------------------------------------------

    def _retry(self, fn, *args, max_retries: int = 5) -> CompletionResult:
        """Retry fn(*args) with exponential backoff on transient errors.

        max_retries=5 (6 total attempts) with delays [5, 15, 30, 60, 120]s
        handles Google AI Studio's aggressive free-tier 504/429 bursts, which
        can require 30-60 s of back-off before the provider recovers.
        Respects Retry-After headers from 429 responses.
        """
        import re as _re
        delays = [5, 15, 30, 60, 120]
        last_exc: Optional[Exception] = None

        for attempt in range(max_retries + 1):
            try:
                return fn(*args)
            except Exception as exc:
                last_exc = exc
                err_str = str(exc).lower()
                is_content_filter = "content_filter" in err_str or (
                    "400" in err_str and "badrequest" in err_str
                )
                if is_content_filter:
                    print(f"[LLMClient] Content filter triggered, skipping trial.")
                    return CompletionResult(
                        content="CONTENT_FILTERED",
                        input_tokens=0,
                        output_tokens=0,
                        cost_usd=0.0,
                        latency_ms=0.0,
                        model=self.model,
                        provider=self.provider,
                    )
                is_transient = any(
                    kw in err_str
                    for kw in ("rate limit", "429", "connection", "timeout",
                               "503", "502", "504", "empty response", "apitimeout",
                               "connecttimeout", "timed out", "internalservererror")
                )
                # openai.InternalServerError wraps any 5xx from the provider
                # (e.g. Google 504 gateway timeouts relayed via OpenRouter).
                # Catch by type so we don't miss variants whose str() lacks
                # the status code.
                if not is_transient:
                    try:
                        import openai as _openai
                        if isinstance(exc, _openai.InternalServerError):
                            is_transient = True
                    except ImportError:
                        pass
                if not is_transient or attempt == max_retries:
                    raise
                # Honour Retry-After from 429 responses; fall back to backoff.
                delay = delays[min(attempt, len(delays) - 1)]
                m = _re.search(r"retry_after_seconds['\"]?\s*:\s*([0-9.]+)", str(exc))
                if m:
                    delay = max(delay, float(m.group(1)) + 2)
                print(
                    f"[LLMClient] Transient error (attempt {attempt+1}/{max_retries}): "
                    f"{exc}. Retrying in {delay:.0f}s..."
                )
                time.sleep(delay)

        raise last_exc  # type: ignore[misc]

    # ------------------------------------------------------------------
    # Cost tracking
    # ------------------------------------------------------------------

    def _track_cost(self, result: CompletionResult) -> None:
        """Accumulate cost and warn if threshold exceeded."""
        self._cumulative_cost += result.cost_usd
        self._call_count += 1
        if self._cumulative_cost > self.cost_alert_usd:
            print(
                f"[COST WARNING] Cumulative cost ${self._cumulative_cost:.4f} "
                f"exceeds alert threshold ${self.cost_alert_usd:.2f}"
            )

    @property
    def cumulative_cost(self) -> float:
        """Total cost in USD across all calls."""
        return self._cumulative_cost
