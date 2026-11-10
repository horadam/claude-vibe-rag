"""The one place that knows how to call each chat provider.

Generation and the eval judge describe what they want (a system prompt, a
user message, optionally a tool the model must call) and this module turns it
into an Anthropic or OpenAI request. Everything above it is provider-neutral.

Keys: with no api_key argument the client uses the key from .env and is
cached. A key passed in (typed into the UI) gets a client for that call only,
so user keys are never kept in module state.
"""

import json
import logging
from dataclasses import dataclass
from functools import cache

import anthropic
import openai

from src.config import API_KEY_ENV, PROVIDERS, Provider, require_env

logger = logging.getLogger(__name__)

# Newer OpenAI models reason before answering, and those hidden tokens count
# against max_completion_tokens. Without headroom a short answer can be cut
# off to nothing, so the visible-output budget gets this added on top.
OPENAI_REASONING_HEADROOM: int = 4096


class ProviderOutputError(Exception):
    """The provider answered, but not with usable output (empty, cut off, refused)."""


@dataclass
class Completion:
    """A text answer plus what it cost."""

    text: str
    model: str
    input_tokens: int
    output_tokens: int


@dataclass
class Tool:
    """A tool the model must call, in a provider-neutral shape."""

    name: str
    description: str
    schema: dict


@dataclass
class ToolCall:
    """The arguments of a forced tool call. error is set when there are none to use."""

    arguments: object
    model: str
    error: str | None = None


def check_provider(provider: str) -> Provider:
    """Return provider unchanged if it's supported, else raise ValueError."""
    if provider not in PROVIDERS:
        raise ValueError(f"unknown provider {provider!r}; use one of {', '.join(PROVIDERS)}")
    return provider  # type: ignore[return-value]


@cache
def _env_openai() -> openai.OpenAI:
    """OpenAI client using the .env key, created on first use."""
    return openai.OpenAI(api_key=require_env(API_KEY_ENV["openai"]))


@cache
def _env_anthropic() -> anthropic.Anthropic:
    """Anthropic client using the .env key, created on first use."""
    return anthropic.Anthropic(api_key=require_env(API_KEY_ENV["anthropic"]))


def openai_client(api_key: str | None = None) -> openai.OpenAI:
    """OpenAI client for this key, or the shared .env client when key is None."""
    return openai.OpenAI(api_key=api_key) if api_key else _env_openai()


def anthropic_client(api_key: str | None = None) -> anthropic.Anthropic:
    """Anthropic client for this key, or the shared .env client when key is None."""
    return anthropic.Anthropic(api_key=api_key) if api_key else _env_anthropic()


def complete(
    provider: str,
    model: str,
    system: str,
    user: str,
    max_tokens: int,
    api_key: str | None = None,
) -> Completion:
    """Send one system + user message and return the text answer."""
    if check_provider(provider) == "anthropic":
        message = anthropic_client(api_key).messages.create(
            model=model,
            max_tokens=max_tokens,
            system=system,
            messages=[{"role": "user", "content": user}],
        )
        text = "".join(block.text for block in message.content if block.type == "text")
        if not text:
            raise ProviderOutputError(f"{message.model} returned no text (stop_reason={message.stop_reason})")
        return Completion(text, message.model, message.usage.input_tokens, message.usage.output_tokens)

    response = openai_client(api_key).chat.completions.create(
        model=model,
        max_completion_tokens=max_tokens + OPENAI_REASONING_HEADROOM,
        messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
    )
    choice = response.choices[0]
    if not choice.message.content:
        reason = choice.message.refusal or f"finish_reason={choice.finish_reason}"
        raise ProviderOutputError(f"{response.model} returned no text ({reason})")
    usage = response.usage
    return Completion(
        choice.message.content,
        response.model,
        usage.prompt_tokens if usage else 0,
        usage.completion_tokens if usage else 0,
    )


def complete_with_tool(
    provider: str,
    model: str,
    system: str,
    user: str,
    tool: Tool,
    max_tokens: int,
    api_key: str | None = None,
) -> ToolCall:
    """Force the model to call tool and return its arguments.

    Missing or undecodable arguments come back as ToolCall.error rather than an
    exception, so the caller can record them as a failed grade.
    """
    if check_provider(provider) == "anthropic":
        message = anthropic_client(api_key).messages.create(
            model=model,
            max_tokens=max_tokens,
            system=system,
            tools=[{"name": tool.name, "description": tool.description, "input_schema": tool.schema}],
            tool_choice={"type": "tool", "name": tool.name},
            messages=[{"role": "user", "content": user}],
        )
        uses = [block for block in message.content if block.type == "tool_use"]
        if not uses:
            return ToolCall(None, message.model, f"model did not call {tool.name}")
        return ToolCall(uses[0].input, message.model)

    response = openai_client(api_key).chat.completions.create(
        model=model,
        max_completion_tokens=max_tokens + OPENAI_REASONING_HEADROOM,
        tools=[{
            "type": "function",
            "function": {"name": tool.name, "description": tool.description, "parameters": tool.schema},
        }],
        tool_choice={"type": "function", "function": {"name": tool.name}},
        messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
    )
    calls = response.choices[0].message.tool_calls or []
    if not calls:
        return ToolCall(None, response.model, f"model did not call {tool.name}")
    raw = calls[0].function.arguments
    # OpenAI sends arguments as a JSON string; Anthropic sends an object.
    try:
        return ToolCall(json.loads(raw), response.model)
    except json.JSONDecodeError as exc:
        return ToolCall(None, response.model, f"{tool.name} arguments are not valid JSON: {exc}")
