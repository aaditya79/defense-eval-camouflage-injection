# Simple Paraphrasing Outperforms Production Safety Classifiers Against Domain-Camouflaged Injection: A Systematic Defense Evaluation

## Abstract

Prompt injection attacks embedded in retrieved documents pose a practical threat
to deployed language model agents. We evaluate six architectural defense
conditions across three models (Claude Haiku, Llama 3.1 8B, Gemini 2.0 Flash)
against domain-camouflaged injection payloads designed to mimic legitimate
professional documents. Baseline camouflage attack success rates (ASR) reach
13.3% on Haiku, 22.2% on both Llama and Gemini, with financial-domain tasks
showing the highest susceptibility at 33-40% across all models. Paraphrasing
reduces ASR to 4.4%, 11.1%, and 0.0% respectively -- outperforming Llama Guard
4 (12B), which achieves camouflage ASRs of 11.1%, 24.4%, and 20.0% while
incurring over 90% over-refusal rates on clean content for Haiku and Llama.
Two comparisons reach statistical significance at n=45: camouflage attacks are
significantly more effective than static payloads on Haiku (McNemar exact
p=0.031), and paraphrasing provides a significant reduction for Gemini
(Fisher exact p=0.001, Cohen's h=0.982). Remaining comparisons are suggestive
but underpowered at current sample sizes (minimum n=56-389 required for 80%
power). Defense effectiveness is model-dependent: spotlighting halves Haiku
camouflage ASR from 13.3% to 6.7% but provides zero reduction on Llama.
Gemini shows a static-camouflage inversion where static payloads (37.8% ASR)
outperform camouflage variants (22.2% ASR) across all three domains, reversing
the pattern seen on other models. Failure analysis across 141 defense breaches
finds task_alignment the dominant failure mode (64%), where agent outputs match
malicious goals because the injected conclusion resembles a legitimate
analytical result.
