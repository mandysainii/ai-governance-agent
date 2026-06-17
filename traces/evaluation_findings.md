# Agent Evaluation: Findings

**Meridian Governance Group: AI Policy Research Agent**
Source: `traces/evaluation_results.json`, `traces/judge_scores.csv` (7 runs total)

> **Data note: four-model claim not backed by data.** The notebook markdown
> (`03_agent_evaluation.ipynb`, cells 9/20/24) describes a **four-model**
> head-to-head including a self-hosted **Qwen 3.6**. The trace data contains
> only **three** models for the head-to-head query: Qwen never produced a
> result (the model was hosted locally but not reachable). Commentary below
> covers the three models that actually ran; the four-model framing should
> either be corrected or the Qwen run re-executed successfully.

---

## Head-to-head judge comparison

All three models answered the identical in-scope query:
*"What does the NIST AI RMF say about the 'Govern' function, and what are its core goals?"*

| Model | Overall | Acc / Rel / Comp / Clar | Cost | Latency | Iters | Tool calls | Quality-per-$ |
|---|---|---|---|---|---|---|---|
| **claude-sonnet-4-6** (primary) | **5.0** | 5 / 5 / 5 / 5 | $0.0876 | 35.3s | 5 | 4 | ~57 |
| **databricks-gpt-oss-120b** | **5.0** | 5 / 5 / 5 / 5 | $0.0040 | 11.4s | 5 | 4 | ~1,248 |
| **claude-haiku-4-5** (secondary) | **3.75** | 3 / 5 / 3 / 4 | $0.0120 | 8.5s | 3 | 2 | ~313 |
| ~~Qwen 3.6 (self-hosted)~~ |: |: |: |: |: |: | *did not run* |

### Commentary

On this question the two models that scored a perfect **5.0**: Sonnet and the
open-source **GPT-OSS-120B**: both ran the full agentic loop (**5 iterations,
4 tool calls**) and grounded their answers in retrieved evidence. The judge
praised both for capturing GOVERN's cross-cutting nature, its core goals, and
its six subcategories, with citations. Notably, GPT-OSS matched Sonnet's
quality at **~22× lower cost** ($0.004 vs $0.088) and **~3× lower latency**,
producing a clean cited table.

The quality gap was entirely with **Haiku (3.75)**, and the judge's rationale
pinpoints *why*: Haiku stopped early (**3 iterations, only 2 tool calls**) and
consequently **misrepresented the subcategory structure** (claimed two primary
categories instead of NIST's six), oversimplified GOVERN 2, and dropped nuances
like risk-management culture, risk tolerance, and TEVV policies. The gap was
driven by **accuracy (3)** and **completeness (3)**: not relevance (5) or
clarity (4). Haiku writes well and stays on-topic; it simply gathers less
evidence and so gets structural regulatory details wrong. This is the crux: the
cheaper model's terseness translates directly into factual gaps on the
regulatory specifics that matter for compliance work.

**On the open-source model:** GPT-OSS-120B *did* use the tools natively
(4 tool calls, same as Sonnet) rather than falling back to a single-shot
answer, and its citations/accuracy matched Sonnet's 5.0 here. That is a strong
data point for the build-vs-buy argument: but it is a single query, so treat
it as promising rather than conclusive.

---

## ROI calculation: which model should Meridian deploy?

ROI here is **judged quality per dollar** (judge overall quality ÷ measured
per-run cost from token usage at published pricing).

| Model | Quality | Cost | Quality-per-$ | vs primary |
|---|---|---|---|---|
| claude-sonnet-4-6 | 5.0 | $0.0876 | ~57 |: |
| claude-haiku-4-5 | 3.75 | $0.0120 | ~313 | **7.3× cheaper**, quality **−1.25** |
| databricks-gpt-oss-120b | 5.0 | $0.0040 | ~1,248 | ~22× cheaper, quality **±0** |

### Deployment recommendation

A **tiered approach**, not a single model.

- **Client-facing compliance deliverables → Claude Sonnet.** Where a wrong
  regulatory detail carries real liability, the ~7× cost premium over Haiku is
  justified: Haiku materially missed structural facts (the six GOVERN
  subcategories) on the head-to-head question, and that is exactly the failure
  mode you cannot ship to a client. Cost per query is still only ~$0.09.

- **Do not use Haiku for regulatory specifics.** Its better quality-per-dollar
  (313) is misleading: the dollars saved buy a measurable accuracy/completeness
  drop on the substance that matters. Reserve it (if at all) for low-stakes
  drafting or summarization with a human review step before delivery.

- **High-volume internal drafting / on-prem requirements → evaluate
  GPT-OSS-120B (build).** It matched Sonnet's quality at ~22× lower marginal
  cost and used tools natively. The build-vs-buy framing: self-hosting trades
  near-zero marginal cost for fixed infra + ops + data-governance burden, and
  keeps data on-prem. **Caveat: n = 1.** Before betting on it, run the full
  evaluation suite across all in-scope questions to confirm accuracy holds on
  harder regulatory comparisons, not just the GOVERN summary.

**Bottom line:** Claude Sonnet for client-facing compliance answers (accuracy
is paramount); open-source GPT-OSS as a strong build candidate for
high-volume / on-prem internal use *pending a broader eval*; Haiku not
recommended for substantive policy answers given the demonstrated accuracy gap.

---

## Full run inventory (all 7 traces)

| # | Model | Query (scope) | Overall | Iters | Tools | Cost | Latency |
|---|---|---|---|---|---|---|---|
| 0 | claude-sonnet-4-6 | NIST GOVERN (in-scope) | 5.0 | 5 | 4 | $0.0876 | 35.3s |
| 1 | claude-sonnet-4-6 | EU AI Act high-risk (in-scope) | 5.0 | 5 | 8 | $0.1358 | 50.4s |
| 2 | claude-sonnet-4-6 | NIST vs EU AI Act (in-scope) | 5.0 | 4 | 7 | $0.0846 | 42.7s |
| 3 | claude-sonnet-4-6 | Cookie recipe (out-of-scope) | 5.0 | 1 | 0 | $0.0076 | 5.6s |
| 4 | claude-sonnet-4-6 | Reverse linked list (out-of-scope) | 5.0 | 1 | 0 | $0.0089 | 7.1s |
| 5 | databricks-gpt-oss-120b | NIST GOVERN (head-to-head) | 5.0 | 5 | 4 | $0.0040 | 11.4s |
| 6 | claude-haiku-4-5 | NIST GOVERN (head-to-head) | 3.75 | 3 | 2 | $0.0120 | 8.5s |

Out-of-scope queries (#3, #4) scored 5.0: the agent correctly declined and
redirected rather than answering off-topic, which is the intended behavior.
