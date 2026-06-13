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

import anthropic

from . import config
from .agent import AgentResult

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
    client: anthropic.Anthropic,
    query: str,
    answer: str,
    category: str,
    judge_model: str = config.JUDGE_MODEL,
) -> JudgeScore:
    """Score one agent answer with the LLM judge. Returns a JudgeScore."""
    user_block = (
        f"USER QUESTION (category: {category}):\n{query}\n\n"
        f"ASSISTANT RESPONSE:\n{answer}\n\n"
        "Score the response now. Return only the JSON object."
    )
    response = client.messages.create(
        model=judge_model,
        max_tokens=512,
        system=_JUDGE_SYSTEM,
        messages=[{"role": "user", "content": user_block}],
    )
    raw = "".join(b.text for b in response.content if b.type == "text")
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
        judge_input_tokens=response.usage.input_tokens,
        judge_output_tokens=response.usage.output_tokens,
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
    """USD cost for a given token usage at the configured per-1M pricing."""
    pricing = config.MODEL_PRICING.get(model)
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
