"""PolicyResearchAgent — a ReAct agent over the Anthropic tool_use API.

The ReAct pattern (Reason + Act) is implemented natively by Claude's tool_use
loop: Claude reasons about the question, calls one of the three policy tools,
reads the tool result, and repeats until it has enough evidence to give a final
answer. This module runs that loop manually so we can:
  * cap the number of iterations (out-of-scope / runaway protection),
  * record every step for inspection, and
  * let Phoenix/OpenTelemetry trace each Anthropic call.

AI-USAGE NOTE: The loop structure follows the Anthropic SDK's documented manual
agentic-loop pattern; it was drafted with Claude Code and reviewed by the author.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any

import anthropic

from . import config
from .tools import TOOL_SCHEMAS, PolicyToolbox


@dataclass
class AgentResult:
    """Everything produced by one agent run — handy for evaluation + traces."""

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


class PolicyResearchAgent:
    """Runs the ReAct tool loop for one model against the policy toolbox."""

    def __init__(
        self,
        toolbox: PolicyToolbox,
        client: anthropic.Anthropic | None = None,
        model: str = config.PRIMARY_MODEL,
        system_prompt: str = config.SYSTEM_PROMPT,
        max_iterations: int = config.MAX_AGENT_ITERATIONS,
        max_tokens: int = config.MAX_TOKENS,
    ) -> None:
        self.toolbox = toolbox
        self.client = client or anthropic.Anthropic(api_key=config.get_api_key())
        self.model = model
        self.system_prompt = system_prompt
        self.max_iterations = max_iterations
        self.max_tokens = max_tokens

    def run(self, query: str) -> AgentResult:
        """Execute the ReAct loop for a single user query and return a result."""
        messages: list[dict[str, Any]] = [{"role": "user", "content": query}]
        result = AgentResult(query=query, model=self.model, answer="")
        start = time.perf_counter()

        for _ in range(self.max_iterations):
            result.iterations += 1
            response = self.client.messages.create(
                model=self.model,
                max_tokens=self.max_tokens,
                system=self.system_prompt,
                tools=TOOL_SCHEMAS,
                messages=messages,
            )

            # Accumulate token usage across every turn of the loop.
            result.input_tokens += response.usage.input_tokens
            result.output_tokens += response.usage.output_tokens
            result.stop_reason = response.stop_reason

            # Record assistant text from this turn (reasoning / final answer).
            assistant_text = "".join(
                b.text for b in response.content if b.type == "text"
            )
            if assistant_text:
                result.transcript.append({"role": "assistant", "text": assistant_text})

            # Done — Claude produced a final answer with no further tool calls.
            if response.stop_reason != "tool_use":
                result.answer = assistant_text
                break

            # Append the assistant turn (must include the tool_use blocks).
            messages.append({"role": "assistant", "content": response.content})

            # Execute every tool the model requested this turn.
            tool_results: list[dict[str, Any]] = []
            for block in response.content:
                if block.type != "tool_use":
                    continue
                output = self.toolbox.dispatch(block.name, block.input)
                result.tool_calls.append(
                    {"name": block.name, "input": block.input, "output": output}
                )
                result.transcript.append(
                    {"role": "tool", "name": block.name, "input": block.input}
                )
                tool_results.append(
                    {
                        "type": "tool_result",
                        "tool_use_id": block.id,
                        "content": output,
                    }
                )

            messages.append({"role": "user", "content": tool_results})
        else:
            # Loop exhausted without a final answer — fail gracefully.
            if not result.answer:
                result.answer = (
                    "I was unable to complete this request within the allotted "
                    "reasoning steps. Please try rephrasing or narrowing the "
                    "question."
                )

        result.latency_s = time.perf_counter() - start
        return result
