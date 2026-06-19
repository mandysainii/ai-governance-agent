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
#
# Each in-scope query carries an optional ``expected_sources`` hint: the
# short_name(s) of the document(s) a good answer should cite. It is used only
# for reporting/diagnosis (which framework the question targets) — the
# groundedness judge does not require it.
IN_SCOPE_QUERIES = [
    {
        "id": "q1_governance_function",
        "query": (
            "What does the NIST AI Risk Management Framework say about the "
            "'Govern' function, and what are its core goals?"
        ),
        "category": "in_scope",
        "head_to_head": True,  # run on BOTH models for the comparison trace
        "expected_sources": ["NIST AI RMF"],
    },
    {
        "id": "q2_high_risk_eu",
        "query": (
            "Under the EU AI Act, how is a 'high-risk' AI system defined and "
            "what obligations apply to its providers?"
        ),
        "category": "in_scope",
        "head_to_head": False,
        "expected_sources": ["EU AI Act"],
    },
    {
        "id": "q3_compare_risk",
        "query": (
            "How do the NIST AI RMF and the EU AI Act differ in their overall "
            "approach to managing AI risk?"
        ),
        "category": "in_scope",
        "head_to_head": False,
        "expected_sources": ["NIST AI RMF", "EU AI Act"],
    },
    # ---- Expanded in-scope set: the four NIST RMF functions, the GenAI
    # Profile, the major EU AI Act provisions, and cross-framework compares. ----
    {
        "id": "q6_map_function",
        "query": (
            "What is the purpose of the 'Map' function in the NIST AI RMF and "
            "what categories does it include?"
        ),
        "category": "in_scope",
        "head_to_head": False,
        "expected_sources": ["NIST AI RMF"],
    },
    {
        "id": "q7_measure_function",
        "query": (
            "What does the 'Measure' function of the NIST AI RMF call for when "
            "assessing AI risks?"
        ),
        "category": "in_scope",
        "head_to_head": False,
        "expected_sources": ["NIST AI RMF"],
    },
    {
        "id": "q8_manage_function",
        "query": (
            "How does the NIST AI RMF describe the 'Manage' function and the "
            "prioritization of risk responses?"
        ),
        "category": "in_scope",
        "head_to_head": False,
        "expected_sources": ["NIST AI RMF"],
    },
    {
        "id": "q9_trustworthy_characteristics",
        "query": (
            "What characteristics of trustworthy AI does the NIST AI RMF "
            "identify?"
        ),
        "category": "in_scope",
        "head_to_head": False,
        "expected_sources": ["NIST AI RMF"],
    },
    {
        "id": "q10_genai_unique_risks",
        "query": (
            "According to the NIST Generative AI Profile, what risks are unique "
            "to or amplified by generative AI?"
        ),
        "category": "in_scope",
        "head_to_head": False,
        "expected_sources": ["NIST GenAI Profile"],
    },
    {
        "id": "q11_genai_suggested_actions",
        "query": (
            "What kinds of suggested actions does the NIST Generative AI Profile "
            "recommend for managing generative AI risks?"
        ),
        "category": "in_scope",
        "head_to_head": False,
        "expected_sources": ["NIST GenAI Profile"],
    },
    {
        "id": "q12_prohibited_practices",
        "query": (
            "Which AI practices are prohibited under Article 5 of the EU AI Act?"
        ),
        "category": "in_scope",
        "head_to_head": False,
        "expected_sources": ["EU AI Act"],
    },
    {
        "id": "q13_gpai_obligations",
        "query": (
            "What obligations does the EU AI Act place on providers of "
            "general-purpose AI (GPAI) models, including those with systemic "
            "risk?"
        ),
        "category": "in_scope",
        "head_to_head": False,
        "expected_sources": ["EU AI Act"],
    },
    {
        "id": "q14_transparency_obligations",
        "query": (
            "What transparency obligations does the EU AI Act impose for "
            "chatbots, deepfakes, and AI-generated content?"
        ),
        "category": "in_scope",
        "head_to_head": False,
        "expected_sources": ["EU AI Act"],
    },
    {
        "id": "q15_penalties",
        "query": (
            "What administrative fines and penalties can be imposed for "
            "non-compliance with the EU AI Act?"
        ),
        "category": "in_scope",
        "head_to_head": False,
        "expected_sources": ["EU AI Act"],
    },
    {
        "id": "q16_risk_categories",
        "query": (
            "How does the EU AI Act's risk-based approach classify AI systems "
            "into different risk tiers?"
        ),
        "category": "in_scope",
        "head_to_head": False,
        "expected_sources": ["EU AI Act"],
    },
    {
        "id": "q17_compare_transparency",
        "query": (
            "How do the NIST AI RMF and the EU AI Act each address transparency "
            "and documentation of AI systems?"
        ),
        "category": "in_scope",
        "head_to_head": False,
        "expected_sources": ["NIST AI RMF", "EU AI Act"],
    },
    {
        "id": "q18_voluntary_vs_mandatory",
        "query": (
            "Is the NIST AI RMF mandatory, and how does its voluntary nature "
            "contrast with the legal force of the EU AI Act?"
        ),
        "category": "in_scope",
        "head_to_head": False,
        "expected_sources": ["NIST AI RMF", "EU AI Act"],
    },
    {
        "id": "q19_human_oversight",
        "query": (
            "What does the EU AI Act require regarding human oversight of "
            "high-risk AI systems?"
        ),
        "category": "in_scope",
        "head_to_head": False,
        "expected_sources": ["EU AI Act"],
    },
    {
        "id": "q20_genai_govern_overlap",
        "query": (
            "How does the NIST Generative AI Profile relate to the 'Govern' "
            "function of the core AI RMF?"
        ),
        "category": "in_scope",
        "head_to_head": False,
        "expected_sources": ["NIST GenAI Profile", "NIST AI RMF"],
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
    # ---- Expanded out-of-scope set. These probe the refusal boundary,
    # including adversarial cases that *sound* policy-adjacent but are not in
    # the knowledge base (US state law, medical/financial advice). ----
    {
        "id": "q21_medical_advice",
        "query": "I have a bad headache and a fever — what medication should I take?",
        "category": "out_of_scope",
        "head_to_head": False,
    },
    {
        "id": "q22_stock_advice",
        "query": "Should I buy NVIDIA stock right now? What's your price target?",
        "category": "out_of_scope",
        "head_to_head": False,
    },
    {
        "id": "q23_us_state_law",
        "query": (
            "What does the California Consumer Privacy Act require for selling "
            "personal data?"
        ),
        "category": "out_of_scope",
        "head_to_head": False,
    },
    {
        "id": "q24_general_chitchat",
        "query": "Write me a short motivational poem about Monday mornings.",
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
# Rubric thresholds — turn raw judge scores into pass/fail.
# A response "passes" if the judge's overall score (1-5) is at or above the
# threshold. This lets us report "Sonnet passed 80% of queries" instead of a
# bare list of numbers. 4.0/5 is the bar for a compliance tool: a passing
# answer is at least "good", not merely "acceptable".
# --------------------------------------------------------------------------- #
PASS_THRESHOLD = 4.0          # overall judge score (1-5) needed to pass
GROUNDEDNESS_THRESHOLD = 4.0  # groundedness score (1-5) needed to pass


def passed(overall: float, threshold: float = PASS_THRESHOLD) -> bool:
    """True if an overall judge score meets the pass/fail bar."""
    return float(overall) >= threshold


def pass_rate(overalls, threshold: float = PASS_THRESHOLD) -> float:
    """Fraction (0-1) of overall scores that pass. Empty input -> 0.0."""
    overalls = list(overalls)
    if not overalls:
        return 0.0
    return sum(1 for o in overalls if passed(o, threshold)) / len(overalls)


# --------------------------------------------------------------------------- #
# Groundedness — is the answer actually backed by the retrieved passages?
# This is the key check for a compliance tool: we must not let the model answer
# from general knowledge instead of the actual EU AI Act / NIST text. It also
# catches the "retrieved the right chunks but still failed to answer" failure,
# because an answer that ignores its evidence scores low.
# --------------------------------------------------------------------------- #
_GROUNDEDNESS_SYSTEM = """You are a strict groundedness checker for an AI \
policy-research assistant. You are given (a) the retrieved source passages the \
assistant had available and (b) the assistant's answer. Judge ONLY whether the \
factual claims in the answer are supported by the retrieved passages — NOT \
whether the answer is well-written or complete.

Score groundedness from 1 to 5:
- 5: every substantive claim is directly supported by the passages.
- 3: mostly supported, but one or more claims go beyond the passages.
- 1: the answer largely relies on outside/general knowledge or invents claims \
not present in the passages.

List any specific claims that are NOT supported by the passages.

SPECIAL CASE: If the answer is a polite refusal of an out-of-scope question \
(it makes no factual policy claims), it is trivially grounded — score 5 with an \
empty unsupported list.

Return ONLY a JSON object with this exact shape and nothing else:
{"groundedness": int, "unsupported_claims": ["..."], "rationale": "one or two \
sentences"}"""


@dataclass
class GroundednessScore:
    groundedness: int
    unsupported_claims: list
    rationale: str
    judge_input_tokens: int = 0
    judge_output_tokens: int = 0

    @property
    def grounded(self) -> bool:
        return self.groundedness >= GROUNDEDNESS_THRESHOLD


# Search tools whose outputs constitute the retrieved evidence.
_RETRIEVAL_TOOLS = {
    "search_policy_documents",
    "compare_frameworks",
    "summarize_policy_topic",
}


def retrieved_context(result: AgentResult) -> str:
    """Concatenate the passages the agent actually retrieved during a run.

    Pulls the output text of every retrieval tool call from the AgentResult.
    Returns "" if the agent answered without retrieving anything (which, for an
    in-scope question, is itself a groundedness red flag).
    """
    chunks = [
        str(tc.get("output", ""))
        for tc in result.tool_calls
        if tc.get("name") in _RETRIEVAL_TOOLS
    ]
    return "\n\n".join(c for c in chunks if c.strip())


def judge_groundedness(
    client,
    query: str,
    answer: str,
    context: str,
    category: str = "in_scope",
    judge_model: str = config.JUDGE_MODEL,
    backend: str = config.JUDGE_BACKEND,
) -> GroundednessScore:
    """Score whether ``answer`` is supported by the retrieved ``context``.

    Mirrors judge_response (same client/backend contract). For out-of-scope
    refusals the judge is told to return a trivially-grounded 5.
    """
    # No retrieval happened. For an in-scope question that means the answer is
    # ungrounded by construction; for an out-of-scope refusal it is fine.
    if not context.strip():
        if category == "out_of_scope":
            return GroundednessScore(5, [], "Out-of-scope refusal; no retrieval needed.")
        return GroundednessScore(
            1, ["entire answer"],
            "No passages were retrieved, so no claim can be grounded.",
        )

    user_block = (
        f"USER QUESTION (category: {category}):\n{query}\n\n"
        f"RETRIEVED PASSAGES:\n{context}\n\n"
        f"ASSISTANT ANSWER:\n{answer}\n\n"
        "Score groundedness now. Return only the JSON object."
    )

    if backend == "anthropic":
        response = config.call_with_retry(
            lambda: client.messages.create(
                model=judge_model,
                max_tokens=512,
                system=_GROUNDEDNESS_SYSTEM,
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
                    {"role": "system", "content": _GROUNDEDNESS_SYSTEM},
                    {"role": "user", "content": user_block},
                ],
            )
        )
        raw = _content_to_str(response.choices[0].message.content)
        usage = getattr(response, "usage", None)
        in_tok = getattr(usage, "prompt_tokens", 0) or 0
        out_tok = getattr(usage, "completion_tokens", 0) or 0

    data = _extract_json(raw)
    claims = data.get("unsupported_claims") or []
    if not isinstance(claims, list):
        claims = [str(claims)]
    return GroundednessScore(
        groundedness=int(data.get("groundedness", 0)),
        unsupported_claims=[str(c) for c in claims],
        rationale=str(data.get("rationale", "")),
        judge_input_tokens=in_tok,
        judge_output_tokens=out_tok,
    )


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


def result_to_record(
    result: AgentResult,
    score: JudgeScore | None,
    groundedness: GroundednessScore | None = None,
) -> dict[str, Any]:
    """Flatten an AgentResult (+ optional judge/groundedness scores) to JSON."""
    record = asdict(result)
    record["cost_usd"] = cost_of_run(result)
    if score is not None:
        record["judge"] = asdict(score)
        record["passed"] = passed(score.overall)
    if groundedness is not None:
        record["groundedness"] = asdict(groundedness)
        record["grounded"] = groundedness.grounded
    return record


# --------------------------------------------------------------------------- #
# Self-check (no network). Run: python -m src.evaluation
# --------------------------------------------------------------------------- #
if __name__ == "__main__":
    # Thresholds.
    assert passed(4.0) and passed(5.0) and not passed(3.9)
    assert pass_rate([5, 4, 3, 2]) == 0.5
    assert pass_rate([]) == 0.0

    # retrieved_context pulls only retrieval-tool outputs.
    r = AgentResult(query="q", model="m", answer="a", tool_calls=[
        {"name": "search_policy_documents", "output": "PASSAGE ONE"},
        {"name": "not_a_tool", "output": "ignore me"},
    ])
    ctx = retrieved_context(r)
    assert "PASSAGE ONE" in ctx and "ignore me" not in ctx

    # No-context groundedness: in-scope fails, out-of-scope refusal passes —
    # both without any LLM call.
    g_in = judge_groundedness(None, "q", "a", "", category="in_scope")
    g_out = judge_groundedness(None, "q", "I can't help with that.", "", category="out_of_scope")
    assert not g_in.grounded and g_out.grounded

    # Unique query IDs across the whole set.
    ids = [q["id"] for q in ALL_QUERIES]
    assert len(ids) == len(set(ids)), "duplicate query id"

    print(f"OK — {len(IN_SCOPE_QUERIES)} in-scope + {len(OUT_OF_SCOPE_QUERIES)} "
          f"out-of-scope queries; threshold/groundedness logic verified.")
