"""Meridian Governance Group — AI Policy Research Agent.

Source package for the USD final project. Modules:
    config        — document URLs, model IDs, chunking + tool parameters
    vector_store  — numpy cosine-similarity vector store (save/load)
    tools         — the agent's three callable tools (search/compare/summarize)
    agent         — PolicyResearchAgent ReAct loop over the Anthropic tool_use API
    evaluation    — evaluation queries, LLM-judge scoring, trace runner
"""
