# Speaker Notes & Slide Data — Agent Architecture onwards

**Project:** Meridian Governance Group — AI Policy Research Agent
**Deck section:** Agent Architecture → end
**Data sources:** `traces/evaluation_results.json`, `traces/judge_scores.csv`, `traces/evaluation_findings.md`, `src/`, `03_agent_evaluation.ipynb`

> This is a superset — more detail than any one slide needs. Curate down. Each
> section gives (a) **Slide values/data** you can drop onto the slide and
> (b) **Speaker notes** for what to say out loud. Numbers marked *(prior run)*
> come from the existing 7-trace run (Sonnet on all in-scope/out-of-scope +
> a 3-model head-to-head). Items marked *(methodology / pending re-run)* are
> capabilities now in the code but not yet re-measured on the expanded set.

---

## Slide — Agent Architecture (overview)

**Slide values/data**
- Pattern: **ReAct** (Reason + Act) — the LLM plans, calls tools, observes results, then synthesizes a final grounded answer.
- One agent class, three interchangeable backends:
  - `anthropic` — Claude Sonnet 4.6 (primary), Claude Haiku 4.5 (secondary)
  - `databricks` — `databricks-gpt-oss-120b` (OpenAI-compatible wire protocol)
  - `openai_compat` — self-hosted Qwen 3.6 at `llm.londonary.com`
- Loop budget: **max 8 iterations**, `max_tokens=2048` per call.
- **Forced-synthesis fallback:** if the loop hits the iteration cap without a final answer, the agent makes one tool-free call that must answer from evidence already gathered (fixes "retrieved good chunks but ran out of tool budget").
- **RAG fallback:** if a backend can't do function-calling, the agent retrieves in code and does a single grounded completion.
- System prompt enforces: answer only AI-governance questions, ground every claim in tools, decline out-of-scope, cite documents by name.

**Speaker notes**
- The headline is *provider-agnostic*: the same agent logic runs on Claude, on a Databricks-hosted open model, and on a self-hosted model. That's deliberate — it lets us run a real build-vs-buy comparison on identical behavior, not on three different codebases.
- ReAct matters for compliance: the model doesn't answer from memory, it must go fetch the regulatory text first. The loop is reason → search → read → maybe search again → answer.
- Two safety nets worth calling out: the forced-synthesis turn (so we never return an empty answer just because the loop ran long) and the RAG single-shot fallback (so weaker local servers without tool-calling still produce grounded answers). Both exist because we saw real failure modes, not hypothetically.
- Tunable knobs all live in `src/config.py` so the notebook and modules never drift.

---

## Slide — Tools & Retrieval

**Slide values/data**
- Three tools exposed to the LLM:
  1. `search_policy_documents(query, top_k=4)` — semantic search across the whole corpus.
  2. `compare_frameworks(topic, doc_id_a, doc_id_b)` — same topic, two documents, side by side.
  3. `summarize_policy_topic(topic)` — deeper recall (top_k + 2) for one topic.
- Retrieval defaults: **top-K = 4** chunks; compare splits to ≥2 per document.
- Chunking: **1000 characters, 200-character overlap** (recursive character splitter).
- Two vector-store backends:
  - **Databricks Vector Search** (default in notebook 03) — index built with `databricks-bge-large-en`, queried via `query_text=` (no local embedder).
  - **Local numpy `SimpleVectorStore`** — fallback for local dev / EDA; embeddings via `all-MiniLM-L6-v2` (384-dim, CPU, no API cost).
- Every retrieval now also emits an **MLflow `RETRIEVER` span** carrying the retrieved chunks as Documents — this is what feeds the groundedness scorer.

**Speaker notes**
- Three tools, not one, because the questions aren't uniform: a single-doc lookup, a two-doc comparison, and a deep single-topic summary are genuinely different retrieval shapes. Giving the model the right tool reduces wasted iterations.
- The agent *chooses* which tool and how many times to call it — you'll see in the data that comparison questions naturally pulled 7–8 chunks across multiple calls, while a simple lookup used 4.
- We support two vector backends so the same agent runs inside Databricks (managed, BGE embeddings) or fully local (numpy + MiniLM). The notebook auto-falls-back if Vector Search is unreachable.
- The RETRIEVER span is the bridge to the eval story coming up: because retrieval is now traced with the actual chunks, MLflow's built-in groundedness judge can check the answer against exactly what was retrieved.

---

## Slide — Knowledge Base

**Slide values/data**
- Three authoritative, publicly available documents (no auth required):
  1. **NIST AI RMF 1.0** (NIST, 2023)
  2. **NIST AI 600-1 — Generative AI Profile** (NIST, 2024)
  3. **EU AI Act — Regulation 2024/1689** (EU, 2024)
- Pipeline (notebook 01): download PDFs → chunk → Delta table → Databricks Vector Search index (`ai_governance_endpoint` / `main.default.ai_governance_index`).

**Speaker notes**
- Two jurisdictions, two philosophies on purpose: NIST is a *voluntary US framework*, the EU AI Act is *binding law with fines*. The agent's value is helping an analyst hold both in view at once.
- All three are public domain / freely available, so there's no licensing or access constraint on the corpus — important for reproducibility and for clients.

---

## Slide — Evaluation Methodology (overview)

**Slide values/data**
- Framework: **MLflow GenAI Evaluation** (`mlflow.genai.evaluate`) with Arize-Phoenix/MLflow tracing.
- Scorers per query:
  - `policy_judge` — **custom LLM-as-judge**, 4 dimensions × 1–5: accuracy, relevance, completeness, clarity, plus an overall.
  - `RetrievalGroundedness` — **MLflow built-in**, is the answer backed by retrieved passages? (yes/no)
  - `RelevanceToQuery` — MLflow built-in.
  - `Safety` — MLflow built-in.
  - `cost_per_query` — custom, USD from token usage.
- Judge model: **Claude Sonnet** when an Anthropic key is present, else `databricks-gpt-oss-120b`.
- Out-of-scope handling baked into the rubric: a clean refusal scores **high** (declining is the correct behavior); attempting to answer off-topic is penalized.

**Speaker notes**
- We score on four dimensions rather than one number because the failure modes differ — a model can be perfectly *relevant* and *clear* while being *inaccurate*. You'll see exactly that with Haiku later.
- The judge is told the special case explicitly: for an out-of-scope question, the *right* answer is a polite refusal, and that should score 5. So our out-of-scope traces test the guardrail, not the knowledge.
- We deliberately moved groundedness onto MLflow's built-in scorer rather than maintaining our own — it reads the retriever span directly and gives us a standard, recognized metric.

---

## Slide — Expanded Test Queries

**Slide values/data**
- Query set grew from **5 → 24**:
  - In-scope: **3 → 18**
  - Out-of-scope: **2 → 6**
- In-scope coverage now spans:
  - All four NIST RMF functions — **Govern, Map, Measure, Manage**
  - Trustworthy-AI characteristics
  - NIST **GenAI Profile** (unique GenAI risks; suggested actions)
  - EU AI Act: **prohibited practices (Art. 5)**, **high-risk obligations**, **GPAI / systemic risk**, **transparency (chatbots/deepfakes)**, **penalties/fines**, **risk tiers**, **human oversight**
  - Cross-framework comparisons (risk approach, transparency, voluntary vs mandatory, GenAI Profile ↔ Govern)
- Out-of-scope now includes **adversarial near-misses**: CCPA / US state privacy law, medical advice, financial advice, plus the original cookie-recipe and coding prompts.
- Each in-scope query tags `expected_sources` (which document a good answer should cite).

**Speaker notes**
- Five queries was enough to demonstrate the pipeline; it is not enough to *trust* a comparison between models. Twenty-four gives breadth across every major section of all three documents.
- The most important additions are the **adversarial out-of-scope** ones. "What does the CCPA require?" *sounds* like a governance question — it's law, it's privacy — but it's not in our knowledge base. A good compliance tool must decline it rather than confidently improvise. That's a much harder guardrail test than a cookie recipe.
- The `expected_sources` tags let us later check retrieval is hitting the right framework, not just any framework.

---

## Slide — Rubric Thresholds (pass/fail)

**Slide values/data**
- **Pass threshold = 4.0 / 5** on the judge's overall score.
- **Groundedness threshold = 4.0 / 5** (custom path) / yes-rate (built-in path).
- Turns raw scores into reportable rates, e.g. *"Sonnet passed 80% of queries, Haiku passed 60%."*
- Helpers: `passed(overall)`, `pass_rate(overalls)` in `src/evaluation.py`.
- Notebook now runs the **full query set per model** and prints a per-model pass-rate + grounded-rate table.

**Speaker notes**
- A mean score of "4.3" doesn't tell a stakeholder whether the tool is safe to ship. A *pass rate* does: "this model gave a client-quality answer on 80% of queries" is a sentence an exec can act on.
- We set the bar at 4.0 because this is a compliance tool — "acceptable" (3) isn't good enough when a wrong regulatory detail carries liability. A passing answer has to be at least "good."
- Same idea applied to groundedness: a separate, explicit line for "is the answer actually backed by the source text."

---

## Slide — Groundedness Checks

**Slide values/data**
- Question answered: *does the answer rely on the retrieved EU AI Act / NIST passages, or on the model's general knowledge?*
- Implementation: MLflow built-in **`RetrievalGroundedness`** scorer reading the **RETRIEVER span** the toolbox emits (chunks as Documents: `page_content` + `metadata`).
- Out-of-scope refusals are **trivially grounded** (no factual claims to support).
- Catches two distinct failure modes:
  1. **Hallucination** — confident claims not in the retrieved text.
  2. **"Retrieved-but-ignored"** — right chunks fetched, answer didn't use them.
- Offline fallback retained in `src/evaluation.py` (`judge_groundedness` + `retrieved_context`) for runs without MLflow.

**Speaker notes**
- This is the single most important check for a *compliance* assistant. It's not enough to sound right — the answer has to be traceable to the actual regulation. Groundedness is what stops the model from answering the EU AI Act from memory.
- It also catches a subtler bug we care about: the agent can retrieve exactly the right passage and still fail to use it. A high accuracy score wouldn't necessarily catch that; a low groundedness score will.
- We chose MLflow's built-in over our own custom judge so the metric is standard and the retriever span is the single source of truth for "what the model actually saw."

---

## Slide — Head-to-Head Model Comparison

**Slide values/data** *(prior run — single query: NIST GOVERN function)*

| Model | Overall | Acc / Rel / Comp / Clar | Cost | Latency | Iters | Tool calls | Quality-per-$ |
|---|---|---|---|---|---|---|---|
| **Claude Sonnet 4.6** (primary) | **5.0** | 5 / 5 / 5 / 5 | $0.0876 | 35.3 s | 5 | 4 | ~57 |
| **Databricks GPT-OSS-120B** | **5.0** | 5 / 5 / 5 / 5 | $0.0040 | 11.4 s | 5 | 4 | ~1,248 |
| **Claude Haiku 4.5** (secondary) | **3.75** | 3 / 5 / 3 / 4 | $0.0120 | 8.5 s | 3 | 2 | ~313 |
| ~~Qwen 3.6 (self-hosted)~~ | — | — | — | — | — | — | *did not run* |

**Speaker notes**
- Two models tied at a perfect 5.0 — Sonnet and the open-source GPT-OSS-120B. Both ran the full agentic loop (5 iterations, 4 tool calls) and grounded their answers. The judge praised both for capturing GOVERN's cross-cutting nature, its core goals, and its six subcategories with citations.
- The striking number: **GPT-OSS matched Sonnet's quality at ~22× lower cost** ($0.004 vs $0.088) and ~3× lower latency. That's the headline of the build-vs-buy case.
- The entire quality gap is **Haiku at 3.75** — and *why* matters. Haiku stopped early (3 iterations, 2 tool calls) and therefore misrepresented the subcategory structure (said two primary categories instead of NIST's six), oversimplified, and dropped nuances. Its gap was **accuracy (3) and completeness (3)** — not relevance (5) or clarity (4).
- The lesson: Haiku writes well and stays on topic; it just gathers less evidence and gets the regulatory specifics wrong. For compliance, that's the exact failure you can't ship.
- **Honesty caveat:** this is a single query. Treat GPT-OSS's parity as *promising, not proven* (see Limitations).

---

## Slide — Out-of-Scope / Guardrail Results

**Slide values/data** *(prior run, Sonnet)*

| # | Query (scope) | Overall | Iters | Tools | Behavior |
|---|---|---|---|---|---|
| q4 | Cookie recipe (out-of-scope) | 5.0 | 1 | 0 | Declined + redirected |
| q5 | Reverse a linked list (out-of-scope) | 5.0 | 1 | 0 | Declined + redirected |

- Both out-of-scope queries scored **5.0**: agent correctly declined, did **0 tool calls**, and redirected to what it can help with.

**Speaker notes**
- The guardrail works as intended: zero retrieval, immediate polite refusal, redirect. No wasted tokens trying to answer off-topic.
- The expanded set adds the *hard* version of this test (CCPA, medical, financial) — those are the ones that will really prove the boundary, because they look more like real governance questions.

---

## Slide — ROI / Cost Analysis

**Slide values/data** *(prior run)*
- ROI defined as **judged quality ÷ measured per-run cost** (token usage at published pricing).

| Model | Quality | Cost | Quality-per-$ | vs primary |
|---|---|---|---|---|
| Claude Sonnet 4.6 | 5.0 | $0.0876 | ~57 | — |
| Claude Haiku 4.5 | 3.75 | $0.0120 | ~313 | **7.3× cheaper, quality −1.25** |
| Databricks GPT-OSS-120B | 5.0 | $0.0040 | ~1,248 | **~22× cheaper, quality ±0** |

- Pricing (USD / 1M tokens): Sonnet $3 in / $15 out; Haiku $1 in / $5 out; GPT-OSS ~$0.15 in / $0.60 out (estimated reference).
- Projected monthly cost @ 5,000 queries/month is computed in the notebook (cost-per-query × volume).
- Full in-scope query costs *(prior run, Sonnet):* GOVERN $0.0876 (4 tools), EU high-risk $0.1358 (8 tools), NIST-vs-EU $0.0846 (7 tools).

**Speaker notes**
- Quality-per-dollar alone is misleading. Haiku's 313 looks great until you remember the dollars saved bought a measurable accuracy drop on the substance that matters.
- The real ROI story is two-sided: GPT-OSS gives you Sonnet-level quality at a fraction of the marginal cost, *if* it holds up across more queries. Self-hosting trades near-zero marginal cost for fixed infra + ops + data-governance burden.
- Note comparison questions cost more — they trigger more tool calls (7–8 vs 4). Cost scales with how much evidence the question demands, which is the right behavior.

---

## Slide — Deployment Recommendation

**Slide values/data**
- **Tiered, not single-model.**
  - **Client-facing compliance deliverables → Claude Sonnet.** Accuracy is paramount; ~$0.09/query is cheap insurance against a wrong regulatory detail.
  - **Do not use Haiku for regulatory specifics.** Better quality-per-$ is misleading; it missed structural facts. Reserve for low-stakes drafting + human review.
  - **High-volume internal / on-prem → evaluate GPT-OSS-120B (build).** Matched Sonnet's quality at ~22× lower marginal cost, used tools natively. **Pending a broader eval.**

**Speaker notes**
- The recommendation isn't "pick the best model," it's "match the model to the stakes." High-liability, client-facing → Sonnet. High-volume, internal, on-prem-required → seriously evaluate self-hosting GPT-OSS.
- Be explicit that the GPT-OSS recommendation is conditional: run the full 24-query suite first. One perfect answer is encouraging, not a procurement decision.
- Bottom line for the slide: **Sonnet for client deliverables; GPT-OSS as a strong build candidate pending broader eval; Haiku not recommended for substantive policy answers.**

---

## Slide — Limitations & Caveats

**Slide values/data**
- **Single-query head-to-head (n=1).** GPT-OSS and Haiku were each judged on one question (NIST GOVERN). Expanded 24-query per-model run is built but not yet re-executed.
- **Qwen 3.6 never produced a result** in the prior run (hosted locally, not reachable) — the "four-model" framing in older notebook markdown is not backed by data; it was effectively three models.
- **Judge is itself an LLM.** Self-consistency / human spot-check recommended for high-stakes scoring.
- **Cost figures for OSS models are estimated reference prices**, not billed rates; self-hosted marginal cost is ~$0 but real cost is fixed GPU/infra.
- Retrieval quality bounded by chunking (1000/200) and top-K (4); not yet tuned per query type.

**Speaker notes**
- Lead with intellectual honesty — it's more persuasive than overclaiming. The pipeline and methodology are solid; the *comparative conclusions* rest on thin data until the expanded run completes.
- Specifically call out the Qwen gap and the n=1 issue so no one in the room over-reads the GPT-OSS result.
- The judge being an LLM is a known limitation of LLM-as-judge generally; we mitigate with a strict 4-dimension rubric and an independent groundedness check, but human review stays in the loop for client deliverables.

---

## Slide — What Changed This Iteration (eval hardening)

**Slide values/data**
- Added **groundedness checks** (MLflow `RetrievalGroundedness` + RETRIEVER spans).
- Added **rubric thresholds** (4.0 pass bar) → per-model pass-rate reporting.
- **Expanded queries 5 → 24** (in-scope 3→18, out-of-scope 2→6, incl. adversarial).
- Removed dead cells left from the MLflow migration; instrumented retrieval for tracing.

**Speaker notes**
- Frame this as moving from "demo that runs" to "evaluation you can trust." The three additions map exactly to three gaps: *how much do we test* (queries), *what's the bar* (thresholds), and *is it grounded* (groundedness).
- These are the pieces that turn a working agent into something you can defend in front of a compliance client.

---

## Slide — Next Steps

**Slide values/data**
- Re-run the full **24-query suite across all models** → populate real per-model pass rates & grounded rates.
- Fix / confirm the **Qwen 3.6** endpoint so the four-model comparison is real.
- Tune chunking / top-K per query type; add retrieval precision@k against `expected_sources`.
- Add human spot-check sample to validate the LLM judge.

**Speaker notes**
- The immediate next action is mechanical: the expanded suite and the new scorers are coded — running them turns every "methodology" claim on these slides into a measured number.
- After that, the higher-value work is retrieval tuning and a small human-validated sample to anchor the judge.
