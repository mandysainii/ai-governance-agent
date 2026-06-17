"""PolicyResearchAgent — a ReAct agent supporting Anthropic and OpenAI-compatible backends.

Supports three backends via a single class:
  - "anthropic"    : Anthropic Messages API (Claude models)
  - "openai_compat": Any OpenAI-compatible endpoint (llm.londonary.com, llama.cpp, etc.)
  - "databricks"   : Databricks Model Serving (OpenAI-compatible wire protocol)

The ReAct loop (Reason + Act) runs natively via each provider's tool/function-
calling API. If function calling is unavailable (some local servers don't support
it), the openai_compat path falls back to retrieval-augmented single-shot.

AI-USAGE NOTE: drafted with Claude Code assistance; reviewed by the author.
"""

from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass, field
from typing import Any

from . import config
from .tools import TOOL_SCHEMAS, PolicyToolbox

# Qwen3 and other reasoning models may emit <think>...</think> blocks.
# Strip them from the final answer; the judge scores only the visible text.
_THINK_RE = re.compile(r"<think>.*?</think>", re.DOTALL)


def _content_to_str(content: Any) -> str:
    """Normalize an OpenAI message content to a plain string.

    Some endpoints return content as a list of typed blocks rather than a
    plain string. This handles both forms safely.
    """
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


@dataclass
class AgentResult:
    """Everything produced by one agent run — used for evaluation and tracing."""

    query: str
    model: str
    answer: str
    tool_calls: list[dict[str, Any]] = field(default_factory=list)
    iterations: int = 0
    stop_reason: str | None = None
    input_tokens: int = 0
    output_tokens: int = 0
    latency_s: float = 0.0
    transcript: list[dict[str, Any]] = field(default_factory=list)


def _to_openai_tools(schemas: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Translate Anthropic-style tool schemas to OpenAI 'function' tools."""
    return [
        {
            "type": "function",
            "function": {
                "name": s["name"],
                "description": s["description"],
                "parameters": s["input_schema"],
            },
        }
        for s in schemas
    ]


class PolicyResearchAgent:
    """ReAct agent that works across Anthropic, Databricks, and OpenAI-compatible backends."""

    def __init__(
        self,
        toolbox,
        client: Any | None = None,
        model: str = config.PRIMARY_MODEL,
        backend: str = "anthropic",
        system_prompt: str = config.SYSTEM_PROMPT,
        max_iterations: int = config.MAX_AGENT_ITERATIONS,
        max_tokens: int = config.MAX_TOKENS,
    ) -> None:
        self.toolbox = toolbox
        self.model = model
        self.backend = backend
        self.system_prompt = system_prompt
        self.max_iterations = max_iterations
        self.max_tokens = max_tokens

        if client is None:
            if backend == "anthropic":
                import anthropic
                client = anthropic.Anthropic(api_key=config.get_api_key())
            else:
                from openai import OpenAI
                client = OpenAI(
                    base_url=config.OPENSOURCE_BASE_URL,
                    api_key=config.get_opensource_api_key(),
                )
        self.client = client

        # Resolve the actual model ID for openai_compat (may auto-discover).
        if backend == "openai_compat" and not model:
            self.model = config.resolve_opensource_model(client)

        self._openai_tools = _to_openai_tools(TOOL_SCHEMAS)

    def run(self, query: str) -> AgentResult:
        result = AgentResult(query=query, model=self.model, answer="")
        start = time.perf_counter()
        if self.backend == "anthropic":
            self._run_anthropic(query, result)
        else:
            try:
                self._run_openai(query, result)
            except Exception as exc:
                result.transcript.append({
                    "role": "system",
                    "text": f"function calling unavailable ({exc}); using retrieval-augmented fallback",
                })
                self._run_rag_fallback(query, result)
        result.latency_s = time.perf_counter() - start
        return result

    # ------------------------------------------------------------------ #
    # Anthropic path
    # ------------------------------------------------------------------ #
    def _run_anthropic(self, query: str, result: AgentResult) -> None:
        messages: list[dict[str, Any]] = [{"role": "user", "content": query}]

        for _ in range(self.max_iterations):
            result.iterations += 1
            response = config.call_with_retry(
                lambda: self.client.messages.create(
                    model=self.model,
                    max_tokens=self.max_tokens,
                    system=self.system_prompt,
                    tools=TOOL_SCHEMAS,
                    messages=messages,
                )
            )
            result.input_tokens += response.usage.input_tokens
            result.output_tokens += response.usage.output_tokens
            result.stop_reason = response.stop_reason

            assistant_text = "".join(
                b.text for b in response.content if b.type == "text"
            )
            if assistant_text:
                result.transcript.append({"role": "assistant", "text": assistant_text})

            if response.stop_reason != "tool_use":
                result.answer = assistant_text
                return

            messages.append({"role": "assistant", "content": response.content})

            tool_results: list[dict[str, Any]] = []
            for block in response.content:
                if block.type != "tool_use":
                    continue
                output = self.toolbox.dispatch(block.name, block.input)
                result.tool_calls.append({"name": block.name, "input": block.input, "output": output})
                result.transcript.append({"role": "tool", "name": block.name, "input": block.input})
                tool_results.append({
                    "type": "tool_result",
                    "tool_use_id": block.id,
                    "content": output,
                })
            messages.append({"role": "user", "content": tool_results})

        # Iteration cap reached without a final answer. Rather than give up,
        # force a final synthesis turn WITHOUT tools so the model must write its
        # answer from the evidence already gathered (fixes low scores where the
        # right chunks were retrieved but the loop ran out of tool-call budget).
        if not result.answer:
            result.answer = self._force_final_anthropic(messages, result)

    _FINAL_SYNTHESIS_NUDGE = (
        "You have gathered sufficient evidence from the tools above. Do NOT "
        "call any more tools. Using only that retrieved evidence, write your "
        "complete final answer now, citing the source documents by name. If the "
        "evidence does not fully cover the question, answer with what it does "
        "support and say what is missing."
    )

    def _force_final_anthropic(self, messages: list[dict[str, Any]], result: AgentResult) -> str:
        """Make one tool-free call so the model commits to a final answer."""
        messages.append({"role": "user", "content": self._FINAL_SYNTHESIS_NUDGE})
        response = config.call_with_retry(
            lambda: self.client.messages.create(
                model=self.model,
                max_tokens=self.max_tokens,
                system=self.system_prompt,
                messages=messages,
            )
        )
        result.input_tokens += response.usage.input_tokens
        result.output_tokens += response.usage.output_tokens
        result.stop_reason = response.stop_reason
        answer = "".join(b.text for b in response.content if b.type == "text")
        if answer:
            result.transcript.append({"role": "assistant", "text": answer})
        return answer

    # ------------------------------------------------------------------ #
    # OpenAI-compatible path (Databricks + londonary + any llama.cpp server)
    # ------------------------------------------------------------------ #
    def _run_openai(self, query: str, result: AgentResult) -> None:
        messages: list[dict[str, Any]] = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": query},
        ]

        for _ in range(self.max_iterations):
            result.iterations += 1
            response = config.call_with_retry(
                lambda: self.client.chat.completions.create(
                    model=self.model,
                    max_tokens=self.max_tokens,
                    messages=messages,
                    tools=self._openai_tools,
                    tool_choice="auto",
                )
            )
            self._accumulate_openai_usage(result, response)
            choice = response.choices[0]
            result.stop_reason = choice.finish_reason
            msg = choice.message

            text = _THINK_RE.sub("", _content_to_str(msg.content)).strip()
            if text:
                result.transcript.append({"role": "assistant", "text": text})

            tool_calls = getattr(msg, "tool_calls", None)
            if not tool_calls:
                result.answer = text
                return

            messages.append({
                "role": "assistant",
                "content": _content_to_str(msg.content),
                "tool_calls": [
                    {
                        "id": tc.id,
                        "type": "function",
                        "function": {"name": tc.function.name, "arguments": tc.function.arguments},
                    }
                    for tc in tool_calls
                ],
            })

            for tc in tool_calls:
                try:
                    args = json.loads(tc.function.arguments or "{}")
                except json.JSONDecodeError:
                    args = {}
                output = self.toolbox.dispatch(tc.function.name, args)
                result.tool_calls.append({"name": tc.function.name, "input": args, "output": output})
                result.transcript.append({"role": "tool", "name": tc.function.name, "input": args})
                messages.append({"role": "tool", "tool_call_id": tc.id, "content": output})

        # Iteration cap reached — force a tool-free final synthesis turn rather
        # than giving up (see _run_anthropic for rationale).
        if not result.answer:
            result.answer = self._force_final_openai(messages, result)

    def _force_final_openai(self, messages: list[dict[str, Any]], result: AgentResult) -> str:
        """Make one tool-free call so the model commits to a final answer."""
        messages.append({"role": "user", "content": self._FINAL_SYNTHESIS_NUDGE})
        response = config.call_with_retry(
            lambda: self.client.chat.completions.create(
                model=self.model,
                max_tokens=self.max_tokens,
                messages=messages,
            )
        )
        self._accumulate_openai_usage(result, response)
        choice = response.choices[0]
        result.stop_reason = choice.finish_reason
        answer = _THINK_RE.sub("", _content_to_str(choice.message.content)).strip()
        if answer:
            result.transcript.append({"role": "assistant", "text": answer})
        return answer

    # ------------------------------------------------------------------ #
    # Fallback: retrieve in code, single grounded completion (no tool calls)
    # ------------------------------------------------------------------ #
    def _run_rag_fallback(self, query: str, result: AgentResult) -> None:
        result.tool_calls.clear()
        result.input_tokens = 0
        result.output_tokens = 0
        result.iterations = 1

        context = self.toolbox.search_policy_documents(query)
        result.tool_calls.append({
            "name": "search_policy_documents",
            "input": {"query": query},
            "output": context,
        })
        user = (
            "Use ONLY the following retrieved passages from the AI-policy "
            "knowledge base to answer. If the question is not about AI "
            "policy/governance, politely decline.\n\n"
            f"PASSAGES:\n{context}\n\nQUESTION: {query}"
        )
        response = config.call_with_retry(
            lambda: self.client.chat.completions.create(
                model=self.model,
                max_tokens=self.max_tokens,
                messages=[
                    {"role": "system", "content": self.system_prompt},
                    {"role": "user", "content": user},
                ],
            )
        )
        self._accumulate_openai_usage(result, response)
        choice = response.choices[0]
        result.stop_reason = choice.finish_reason
        result.answer = _THINK_RE.sub("", _content_to_str(choice.message.content)).strip()

    # ------------------------------------------------------------------ #
    @staticmethod
    def _accumulate_openai_usage(result: AgentResult, response: Any) -> None:
        usage = getattr(response, "usage", None)
        if usage is not None:
            result.input_tokens += getattr(usage, "prompt_tokens", 0) or 0
            result.output_tokens += getattr(usage, "completion_tokens", 0) or 0
