"""Evaluation: the trace query set, an LLM-as-judge scorer, and an ROI helper.

The rubric requires:
  * 5 evaluation traces via an established provider (we use Arize Phoenix),
  * at least one trace running TWO different LLMs on the SAME query,
  * an LLM judge that scores responses and supports written commentary,
  * 2 examples of graceful rejection of irrelevant (out-of-scope) queries,
  * an ROI calculation comparing the two LLMs.

AI-USAGE NOTE: The judge prompt + rubric and the query set were drafted with
Claude Code and reviewed/edited by the author.
"""

from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass
from typing import Any

from . import config
from .agent import AgentResult


def _content_to_str(content) -> str:
    """Normalize OpenAI message content to a plain string (handles list blocks)."""
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "".join(
            item.get("text", "") if isinstance(item, dict) else str(item)
            for item in content
        )
    return str(content)

# --------------------------------------------------------------------------- #
# Evaluation query set.
# Traces 1-3 are in-scope policy questions. Trace 1 is the head-to-head
# (two LLMs, same query). Traces 4-5 are out-of-scope and must be declined.
# --------------------------------------------------------------------------- #
IN_SCOPE_QUERIES = [
    {
        "id": "q1_governance_function",
        "query": (
            "What does the NIST AI Risk Management Framework say about the "
            "'Govern' function, and what are its core goals?"
        ),
        "category": "in_scope",
        "head_to_head": True,  # run on BOTH models for the comparison trace
    },
    {
        "id": "q2_high_risk_eu",
        "query": (
            "Under the EU AI Act, how is a 'high-risk' AI system defined and "
            "what obligations apply to its providers?"
        ),
        "category": "in_scope",
        "head_to_head": False,
    },
    {
        "id": "q3_compare_risk",
        "query": (
            "How do the NIST AI RMF and the EU AI Act differ in their overall "
            "approach to managing AI risk?"
        ),
        "category": "in_scope",
        "head_to_head": False,
    },
    {
        "id": "q6_genai_risks",
        "query": (
            "What unique risks does NIST AI 600-1 identify for generative AI "
            "systems, and how does it recommend organizations address them?"
        ),
        "category": "in_scope",
        "head_to_head": False,
    },
    {
        "id": "q7_genai_vs_rmf",
        "query": (
            "How does the NIST AI 600-1 Generative AI Profile relate to and "
            "extend the core NIST AI RMF 1.0? What gaps does it fill?"
        ),
        "category": "in_scope",
        "head_to_head": False,
    },
    {
        "id": "q8_prohibited_eu",
        "query": (
            "What AI practices does the EU AI Act explicitly prohibit, and what "
            "is the rationale given for each prohibition?"
        ),
        "category": "in_scope",
        "head_to_head": False,
    },
    {
        "id": "q9_transparency",
        "query": (
            "How do the NIST AI RMF and EU AI Act each address transparency "
            "and explainability requirements for AI systems?"
        ),
        "category": "in_scope",
        "head_to_head": False,
    },
]

OUT_OF_SCOPE_QUERIES = [
    {
        "id": "q4_recipe",
        "query": "Can you give me a good recipe for chocolate chip cookies?",
        "category": "out_of_scope",
        "head_to_head": False,
    },
    {
        "id": "q5_python_help",
        "query": (
            "Write me a Python function that reverses a linked list and explain "
            "the time complexity."
        ),
        "category": "out_of_scope",
        "head_to_head": False,
    },
    {
        "id": "q10_stock_advice",
        "query": (
            "Which tech stocks should I invest in right now given the current "
            "market conditions?"
        ),
        "category": "out_of_scope",
        "head_to_head": False,
    },
]

ALL_QUERIES = IN_SCOPE_QUERIES + OUT_OF_SCOPE_QUERIES


# --------------------------------------------------------------------------- #
# LLM-as-judge.
# --------------------------------------------------------------------------- #
_JUDGE_SYSTEM = """You are a strict evaluator of an AI policy-research \
assistant. You score a single response on four dimensions, each from 1 (poor) \
to 5 (excellent):

- accuracy: Are the claims correct and faithful to AI-governance source \
material (NIST AI RMF, NIST GenAI Profile, EU AI Act)?
- relevance: Does the response address what was asked?
- completeness: Does it cover the important points (citations, specifics)?
- clarity: Is it well-organized and easy for a professional to read?

SPECIAL CASE — OUT-OF-SCOPE QUERIES: If the user's question is NOT about AI \
policy/governance, the CORRECT behavior is a polite refusal that redirects to \
what the assistant can help with. For such queries, score a clean refusal \
HIGH on all dimensions (it is accurate, relevant, complete, and clear to \
decline). Penalize a response that actually attempts to answer an \
out-of-scope question.

Return ONLY a JSON object with this exact shape and nothing else:
{"accuracy": int, "relevance": int, "completeness": int, "clarity": int, \
"overall": float, "rationale": "one or two sentences"}"""


@dataclass
class JudgeScore:
    accuracy: int
    relevance: int
    completeness: int
    clarity: int
    overall: float
    rationale: str
    judge_input_tokens: int = 0
    judge_output_tokens: int = 0


def judge_response(
    client,
    query: str,
    answer: str,
    category: str,
    judge_model: str = config.JUDGE_MODEL,
    backend: str = config.JUDGE_BACKEND,
) -> JudgeScore:
    """Score one agent answer with the LLM judge. Returns a JudgeScore.

    Accepts either an Anthropic client (backend='anthropic') or an
    OpenAI-compatible client (backend='databricks' or 'openai_compat').
    """
    user_block = (
        f"USER QUESTION (category: {category}):\n{query}\n\n"
        f"ASSISTANT RESPONSE:\n{answer}\n\n"
        "Score the response now. Return only the JSON object."
    )

    if backend == "anthropic":
        response = config.call_with_retry(
            lambda: client.messages.create(
                model=judge_model,
                max_tokens=512,
                system=_JUDGE_SYSTEM,
                messages=[{"role": "user", "content": user_block}],
            )
        )
        raw = "".join(b.text for b in response.content if b.type == "text")
        in_tok = response.usage.input_tokens
        out_tok = response.usage.output_tokens
    else:
        response = config.call_with_retry(
            lambda: client.chat.completions.create(
                model=judge_model,
                max_tokens=512,
                messages=[
                    {"role": "system", "content": _JUDGE_SYSTEM},
                    {"role": "user", "content": user_block},
                ],
            )
        )
        raw = _content_to_str(response.choices[0].message.content)
        usage = getattr(response, "usage", None)
        in_tok = getattr(usage, "prompt_tokens", 0) or 0
        out_tok = getattr(usage, "completion_tokens", 0) or 0

    data = _extract_json(raw)
    dims = ["accuracy", "relevance", "completeness", "clarity"]
    scores = {d: int(data.get(d, 0)) for d in dims}
    overall = data.get("overall")
    if overall is None:
        overall = sum(scores.values()) / len(dims)
    return JudgeScore(
        **scores,
        overall=float(overall),
        rationale=str(data.get("rationale", "")),
        judge_input_tokens=in_tok,
        judge_output_tokens=out_tok,
    )


# --------------------------------------------------------------------------- #
# Rubric thresholds.
# --------------------------------------------------------------------------- #
PASS_THRESHOLD = 3.0


def passes_rubric(score: JudgeScore) -> dict[str, bool]:
    """Return a per-dimension and overall pass/fail for a judge score.

    Any dimension below PASS_THRESHOLD is a fail. Use this to flag traces
    that need human review before a production deployment decision.
    """
    dims = ["accuracy", "relevance", "completeness", "clarity"]
    result = {d: getattr(score, d) >= PASS_THRESHOLD for d in dims}
    result["overall"] = score.overall >= PASS_THRESHOLD
    return result


# --------------------------------------------------------------------------- #
# Groundedness check.
# --------------------------------------------------------------------------- #
_GROUNDEDNESS_SYSTEM = """You are evaluating whether an AI assistant's answer \
is grounded in the source documents provided to it.

You will receive:
  1. The user's question.
  2. The retrieved context chunks the assistant had access to.
  3. The assistant's answer.

Score groundedness from 1 to 5:
  5 — Every substantive claim traces directly to the provided context. \
Citations (article numbers, section names, function names) match the chunks.
  4 — Most claims are grounded; one minor point may come from general \
knowledge but does not change the answer.
  3 — The answer is broadly consistent with the context but several specific \
claims are not supported by the retrieved chunks.
  2 — The answer mixes grounded claims with clear hallucinations or \
unsupported specifics.
  1 — The answer is largely or entirely unsupported by the retrieved context.

SPECIAL CASE: If the context chunks are empty or the question is out-of-scope \
and the assistant correctly declined, return a score of 5 with rationale \
"Correct refusal; groundedness check not applicable."

Return ONLY a JSON object with this exact shape and nothing else:
{"groundedness": int, "rationale": "one or two sentences"}"""


@dataclass
class GroundednessScore:
    groundedness: int
    rationale: str
    passes: bool = False

    def __post_init__(self):
        self.passes = self.groundedness >= PASS_THRESHOLD


def groundedness_score(
    client,
    query: str,
    answer: str,
    retrieved_chunks: list[str],
    judge_model: str = config.JUDGE_MODEL,
    backend: str = config.JUDGE_BACKEND,
) -> GroundednessScore:
    """Score how well the answer is grounded in the retrieved context chunks.

    Pass retrieved_chunks as a list of strings (the raw text of each chunk
    returned by the vector search tool). If the list is empty the judge will
    apply the out-of-scope special case.
    """
    context_block = "\n\n---\n\n".join(retrieved_chunks) if retrieved_chunks else "(none)"
    user_block = (
        f"USER QUESTION:\n{query}\n\n"
        f"RETRIEVED CONTEXT:\n{context_block}\n\n"
        f"ASSISTANT ANSWER:\n{answer}\n\n"
        "Score the groundedness now. Return only the JSON object."
    )

    if backend == "anthropic":
        response = config.call_with_retry(
            lambda: client.messages.create(
                model=judge_model,
                max_tokens=256,
                system=_GROUNDEDNESS_SYSTEM,
                messages=[{"role": "user", "content": user_block}],
            )
        )
        raw = "".join(b.text for b in response.content if b.type == "text")
    else:
        response = config.call_with_retry(
            lambda: client.chat.completions.create(
                model=judge_model,
                max_tokens=256,
                messages=[
                    {"role": "system", "content": _GROUNDEDNESS_SYSTEM},
                    {"role": "user", "content": user_block},
                ],
            )
        )
        raw = _content_to_str(response.choices[0].message.content)

    data = _extract_json(raw)
    return GroundednessScore(
        groundedness=int(data.get("groundedness", 0)),
        rationale=str(data.get("rationale", "")),
    )


def _extract_json(text: str) -> dict[str, Any]:
    """Best-effort parse of the judge's JSON, even if wrapped in prose/fences."""
    text = text.strip()
    # Strip ```json fences if present.
    text = re.sub(r"^```(?:json)?|```$", "", text, flags=re.MULTILINE).strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", text, flags=re.DOTALL)
        if match:
            try:
                return json.loads(match.group(0))
            except json.JSONDecodeError:
                pass
    return {}


# --------------------------------------------------------------------------- #
# Cost / ROI helpers.
# --------------------------------------------------------------------------- #
def estimate_cost(model: str, input_tokens: int, output_tokens: int) -> float:
    """USD cost for a given token usage at the configured per-1M pricing.

    Exact match against MODEL_PRICING (Anthropic) wins. Otherwise we fall back
    to OSS_REFERENCE_PRICING by matching the model-family substring (e.g.
    'gpt-oss-120b', 'qwen', 'llama') so Databricks-hosted and self-hosted
    endpoints get a finite estimated cost for the ROI comparison instead of $0.
    """
    pricing = config.MODEL_PRICING.get(model)
    if pricing is None:
        lname = (model or "").lower()
        for family, ref in config.OSS_REFERENCE_PRICING.items():
            if family in lname:
                pricing = ref
                break
    if pricing is None:
        return 0.0
    return (
        input_tokens / 1_000_000 * pricing["input"]
        + output_tokens / 1_000_000 * pricing["output"]
    )


def cost_of_run(result: AgentResult) -> float:
    """Convenience: total USD cost of one AgentResult."""
    return estimate_cost(result.model, result.input_tokens, result.output_tokens)


def roi_comparison(
    primary: AgentResult,
    secondary: AgentResult,
    primary_score: JudgeScore,
    secondary_score: JudgeScore,
) -> dict[str, Any]:
    """Build an ROI comparison record for the two models on the same query.

    'ROI' here = judged quality per dollar. We report cost, quality (overall
    judge score), latency, and a quality-per-dollar ratio so the notebook can
    recommend which model to deploy.
    """
    p_cost = cost_of_run(primary)
    s_cost = cost_of_run(secondary)

    def quality_per_dollar(score: float, cost: float) -> float:
        return score / cost if cost > 0 else float("inf")

    return {
        "query": primary.query,
        "primary": {
            "model": primary.model,
            "cost_usd": p_cost,
            "overall_quality": primary_score.overall,
            "latency_s": primary.latency_s,
            "input_tokens": primary.input_tokens,
            "output_tokens": primary.output_tokens,
            "quality_per_dollar": quality_per_dollar(primary_score.overall, p_cost),
        },
        "secondary": {
            "model": secondary.model,
            "cost_usd": s_cost,
            "overall_quality": secondary_score.overall,
            "latency_s": secondary.latency_s,
            "input_tokens": secondary.input_tokens,
            "output_tokens": secondary.output_tokens,
            "quality_per_dollar": quality_per_dollar(secondary_score.overall, s_cost),
        },
        "cost_ratio_primary_over_secondary": (
            p_cost / s_cost if s_cost > 0 else float("inf")
        ),
        "quality_delta_primary_minus_secondary": (
            primary_score.overall - secondary_score.overall
        ),
    }


def roi_table(
    results: dict[str, AgentResult],
    scores: dict[str, JudgeScore],
) -> list[dict[str, Any]]:
    """Build a per-model ROI table (cost, quality, latency, quality-per-$).

    Works for any number of models — used for the 3-way Sonnet / Haiku /
    open-source comparison. A self-hosted model has cost 0, so its
    quality-per-dollar is reported as infinity (annotate this in the notebook:
    marginal per-token cost is ~$0; the real cost is fixed GPU/hour infra).
    """
    rows: list[dict[str, Any]] = []
    for label, res in results.items():
        cost = cost_of_run(res)
        quality = scores[label].overall
        rows.append(
            {
                "label": label,
                "model": res.model,
                "overall_quality": quality,
                "cost_usd": cost,
                "latency_s": res.latency_s,
                "input_tokens": res.input_tokens,
                "output_tokens": res.output_tokens,
                "quality_per_dollar": quality / cost if cost > 0 else float("inf"),
            }
        )
    return rows


def result_to_record(result: AgentResult, score: JudgeScore | None) -> dict[str, Any]:
    """Flatten an AgentResult (+ optional judge score) into a JSON-able record."""
    record = asdict(result)
    record["cost_usd"] = cost_of_run(result)
    if score is not None:
        record["judge"] = asdict(score)
    return record
