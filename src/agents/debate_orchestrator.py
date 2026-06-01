"""
DebateOrchestrator: runs multi-round debates across n agents, aggregates
results via majority vote, and supports selective injection.

Adapted from chameleon-injection for the grounded-guard project.
"""

from __future__ import annotations

import random
import re
from dataclasses import dataclass, field
from typing import List, Optional

from src.utils.llm_client import LLMClient
from src.agents.single_agent import AgentResponse
from src.agents.debate_agent import DebateAgent, DebateRound


@dataclass
class DebateResult:
    """Complete result of one debate run."""

    task_id: str
    round_history: List[DebateRound]
    final_positions: List[str]
    final_answer: str
    consensus_reached: bool
    injected_agent_idx: Optional[int]
    total_cost_usd: float
    total_tokens: int = 0


_DISCRETE_PATTERNS = {
    "recommendation":  re.compile(r"\b(BUY|HOLD|SELL)\b",                      re.IGNORECASE),
    "risk_level":      re.compile(r"\b(HIGH|MEDIUM|LOW)\b",                     re.IGNORECASE),
    "policy":          re.compile(r"\b(HAWKISH|NEUTRAL|DOVISH)\b",              re.IGNORECASE),
    "compliance":      re.compile(r"\b(COMPLIANT|NON[\-\s]?COMPLIANT)\b",       re.IGNORECASE),
    "validity":        re.compile(r"\b(ADEQUATE|INADEQUATE|VALID|INVALID)\b",   re.IGNORECASE),
    "severity":        re.compile(r"\b(MAJOR|MODERATE|MINOR)\b",               re.IGNORECASE),
    "significance":    re.compile(r"\b(SIGNIFICANT|NOT_SIGNIFICANT|BORDERLINE)\b", re.IGNORECASE),
    "escalation":      re.compile(r"\b(ESCALATE|MAINTAIN|DE[\-\s]?ESCALATE)\b", re.IGNORECASE),
    "risk_action":     re.compile(r"\b(LOW_RISK|INTERMEDIATE_RISK|HIGH_RISK)\b", re.IGNORECASE),
    "replication":     re.compile(
        r"\b(SUCCESSFUL_REPLICATION|PARTIAL_REPLICATION|FAILED_REPLICATION)\b",
        re.IGNORECASE,
    ),
    "intervention":    re.compile(
        r"\b(URGENT_INTERVENTION|CLOSE_MONITORING|REASSURE)\b",
        re.IGNORECASE,
    ),
    "sentiment":       re.compile(r"\b(POSITIVE|NEGATIVE|MIXED)\b",             re.IGNORECASE),
}


def _extract_discrete_answer(text: str) -> Optional[str]:
    """Try to extract a discrete answer keyword from text."""
    for pattern in _DISCRETE_PATTERNS.values():
        m = pattern.search(text)
        if m:
            return m.group(0).upper()
    return None


def _majority_vote(positions: List[str]) -> tuple[str, bool]:
    answers = [_extract_discrete_answer(p) for p in positions]
    answers = [a for a in answers if a is not None]
    if not answers:
        return positions[0] if positions else "", False

    counts: dict[str, int] = {}
    for a in answers:
        counts[a] = counts.get(a, 0) + 1

    max_count = max(counts.values())
    winners = [k for k, v in counts.items() if v == max_count]
    consensus = len(winners) == 1 and max_count > 1

    if len(winners) == 1:
        return winners[0], consensus
    first = _extract_discrete_answer(positions[0])
    return (first or winners[0]), False


class DebateOrchestrator:
    """
    Runs multi-round debates and aggregates results via majority vote.

    Args:
        n_agents: Number of debating agents.
        n_rounds: Number of debate rounds.
        llm_client: LLMClient shared by all agents.
        aggregation: "majority_vote" (only option for now).
        seed: Random seed for inject_one_random.
    """

    _ROLE_NAMES = ["Analyst_A", "Analyst_B", "Analyst_C", "Analyst_D", "Analyst_E"]

    def __init__(
        self,
        n_agents: int,
        n_rounds: int,
        llm_client: LLMClient,
        aggregation: str = "majority_vote",
        seed: int = 42,
    ) -> None:
        self.n_agents   = n_agents
        self.n_rounds   = n_rounds
        self.aggregation = aggregation
        self._rng = random.Random(seed)

        self.agents = [
            DebateAgent(
                agent_id=self._ROLE_NAMES[i],
                role=self._ROLE_NAMES[i],
                llm_client=llm_client,
            )
            for i in range(n_agents)
        ]

    def run_debate(
        self,
        instruction: str,
        context: str,
        task_id: str,
        injected_context: Optional[str] = None,
        inject_mode: str = "inject_all",
        injected_agent_idx: Optional[int] = None,
    ) -> DebateResult:
        """
        Run a full debate and return the aggregated result.

        inject_mode:
            "inject_all"   – all agents receive injected_context
            "inject_first" – only agent at injected_agent_idx gets injected_context
            "clean"        – no injection (baseline)
        """
        if injected_context is None:
            injected_context = context

        actual_injected_idx: Optional[int] = None
        if inject_mode == "inject_all":
            actual_injected_idx = None
        elif inject_mode == "inject_first":
            actual_injected_idx = injected_agent_idx if injected_agent_idx is not None else 0
        elif inject_mode == "clean":
            actual_injected_idx = None
            injected_context = context
        else:
            raise ValueError(f"Unknown inject_mode: {inject_mode!r}")

        def _ctx(idx: int) -> str:
            if inject_mode in ("inject_all",):
                return injected_context
            if inject_mode == "clean":
                return context
            return injected_context if idx == actual_injected_idx else context

        round_history: List[DebateRound] = []
        total_cost = 0.0
        total_tokens = 0

        for round_num in range(1, self.n_rounds + 1):
            responses: List[AgentResponse] = []
            for idx, agent in enumerate(self.agents):
                resp = agent.generate_position(
                    instruction=instruction,
                    context=_ctx(idx),
                    debate_history=round_history,
                    round_num=round_num,
                )
                responses.append(resp)
                total_cost   += resp.completion.cost_usd
                total_tokens += resp.completion.input_tokens + resp.completion.output_tokens
            round_history.append(DebateRound(round_num=round_num, responses=responses))

        final_positions = [r.response_text for r in round_history[-1].responses]
        final_answer, consensus = _majority_vote(final_positions)

        return DebateResult(
            task_id=task_id,
            round_history=round_history,
            final_positions=final_positions,
            final_answer=final_answer,
            consensus_reached=consensus,
            injected_agent_idx=actual_injected_idx,
            total_cost_usd=total_cost,
            total_tokens=total_tokens,
        )

    def compute_conformity_pressure(
        self,
        debate_result: DebateResult,
        injected_agent_idx: int,
    ) -> float:
        """
        CPS = fraction of non-injected agents whose final position matches
        the injected agent's final discrete answer.
        """
        if not debate_result.final_positions:
            return 0.0
        injected_answer = _extract_discrete_answer(
            debate_result.final_positions[injected_agent_idx]
        )
        if injected_answer is None:
            return 0.0
        non_injected = [
            p for i, p in enumerate(debate_result.final_positions)
            if i != injected_agent_idx
        ]
        if not non_injected:
            return 0.0
        adopted = sum(1 for p in non_injected if _extract_discrete_answer(p) == injected_answer)
        return adopted / len(non_injected)
