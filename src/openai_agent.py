"""OpenAICompatAgent — the same ReAct loop against an OpenAI-compatible server.

Target: a Qwen3 model self-hosted on a llama.cpp ``llama-server`` (e.g.
llm.londonary.com). llama.cpp's server speaks the OpenAI Chat Completions API,
so the official ``openai`` SDK pointed at the server's ``base_url`` is the
standard way to call it.

WHY THE OPENAI SDK IS HERE: this project's primary/secondary models are Claude
(see src/agent.py). This backend exists ONLY to benchmark an open-source model
for the assignment's ROI comparison. The OpenAI SDK is used purely as the wire
protocol for llama.cpp — not as a model provider.

Behavior: try native function-calling first; if the server doesn't support
tools, fall back to retrieval-augmented single-shot (we run the search in code
and hand the passages to the model). Either way we return the same AgentResult
the Anthropic agent produces, so the judge/ROI code is identical.

AI-USAGE NOTE: drafted with Claude Code, reviewed by the author.
"""

from __future__ import annotations

import json
import re
import time
from typing import Any

from . import config
from .agent import AgentResult
from .tools import TOOL_SCHEMAS, PolicyToolbox

# Qwen3 (and other reasoning models) may emit <think>...</think> in content.
# Strip it from the user-facing answer; the judge scores the final answer only.
_THINK_RE = re.compile(r"<think>.*?</think>", re.DOTALL)


def _to_openai_tools(schemas: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Translate our Anthropic-style tool schemas to OpenAI 'function' tools."""
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


class OpenAICompatAgent:
    """ReAct agent for an OpenAI-compatible (llama.cpp/Qwen3) endpoint."""

    def __init__(
        self,
        toolbox: PolicyToolbox,
        client: Any | None = None,
        model: str | None = None,
        system_prompt: str = config.SYSTEM_PROMPT,
        max_iterations: int = config.MAX_AGENT_ITERATIONS,
        max_tokens: int = config.MAX_TOKENS,
    ) -> None:
        if client is None:
            from openai import OpenAI  # imported lazily so Anthropic-only runs don't need it

            client = OpenAI(
                base_url=config.OPENSOURCE_BASE_URL,
                api_key=config.get_opensource_api_key(),
            )
        self.client = client
        self.toolbox = toolbox
        self.model = model or config.resolve_opensource_model(client)
        self.system_prompt = system_prompt
        self.max_iterations = max_iterations
        self.max_tokens = max_tokens
        self.openai_tools = _to_openai_tools(TOOL_SCHEMAS)

    def run(self, query: str) -> AgentResult:
        result = AgentResult(query=query, model=self.model, answer="")
        start = time.perf_counter()
        try:
            self._run_with_tools(query, result)
        except Exception as exc:
            # Server likely doesn't support function calling — degrade to RAG.
            result.transcript.append(
                {
                    "role": "system",
                    "text": f"function calling unavailable ({exc}); "
                    "using retrieval-augmented fallback",
                }
            )
            self._run_rag_fallback(query, result)
        result.latency_s = time.perf_counter() - start
        return result

    # --------------------------------------------------------------------- #
    # Primary path: native OpenAI-style function calling.
    # --------------------------------------------------------------------- #
    def _run_with_tools(self, query: str, result: AgentResult) -> None:
        messages: list[dict[str, Any]] = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": query},
        ]
        for _ in range(self.max_iterations):
            result.iterations += 1
            response = self.client.chat.completions.create(
                model=self.model,
                max_tokens=self.max_tokens,
                messages=messages,
                tools=self.openai_tools,
                tool_choice="auto",
            )
            self._add_usage(result, response)
            choice = response.choices[0]
            result.stop_reason = choice.finish_reason
            msg = choice.message

            text = _THINK_RE.sub("", msg.content or "").strip()
            if text:
                result.transcript.append({"role": "assistant", "text": text})

            tool_calls = getattr(msg, "tool_calls", None)
            if not tool_calls:
                result.answer = text
                return

            # Echo the assistant turn (must include the tool_calls).
            messages.append(
                {
                    "role": "assistant",
                    "content": msg.content or "",
                    "tool_calls": [
                        {
                            "id": tc.id,
                            "type": "function",
                            "function": {
                                "name": tc.function.name,
                                "arguments": tc.function.arguments,
                            },
                        }
                        for tc in tool_calls
                    ],
                }
            )

            for tc in tool_calls:
                try:
                    args = json.loads(tc.function.arguments or "{}")
                except json.JSONDecodeError:
                    args = {}
                output = self.toolbox.dispatch(tc.function.name, args)
                result.tool_calls.append(
                    {"name": tc.function.name, "input": args, "output": output}
                )
                result.transcript.append(
                    {"role": "tool", "name": tc.function.name, "input": args}
                )
                messages.append(
                    {"role": "tool", "tool_call_id": tc.id, "content": output}
                )

        if not result.answer:
            result.answer = (
                "I was unable to complete this request within the allotted "
                "reasoning steps. Please try rephrasing or narrowing the question."
            )

    # --------------------------------------------------------------------- #
    # Fallback path: retrieve in code, then a single grounded completion.
    # --------------------------------------------------------------------- #
    def _run_rag_fallback(self, query: str, result: AgentResult) -> None:
        # Reset any partial counters from the failed tool attempt.
        result.tool_calls.clear()
        result.input_tokens = 0
        result.output_tokens = 0
        result.iterations = 1

        context = self.toolbox.search_policy_documents(query)
        result.tool_calls.append(
            {
                "name": "search_policy_documents",
                "input": {"query": query},
                "output": context,
            }
        )
        user = (
            "Use ONLY the following retrieved passages from the AI-policy "
            "knowledge base to answer. If the question is not about AI "
            "policy/governance, politely decline as your instructions require "
            "(do not answer it).\n\n"
            f"PASSAGES:\n{context}\n\nQUESTION: {query}"
        )
        response = self.client.chat.completions.create(
            model=self.model,
            max_tokens=self.max_tokens,
            messages=[
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": user},
            ],
        )
        self._add_usage(result, response)
        choice = response.choices[0]
        result.stop_reason = choice.finish_reason
        result.answer = _THINK_RE.sub("", choice.message.content or "").strip()

    # --------------------------------------------------------------------- #
    @staticmethod
    def _add_usage(result: AgentResult, response: Any) -> None:
        usage = getattr(response, "usage", None)
        if usage is not None:
            result.input_tokens += getattr(usage, "prompt_tokens", 0) or 0
            result.output_tokens += getattr(usage, "completion_tokens", 0) or 0
