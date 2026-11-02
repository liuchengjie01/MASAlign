"""Model client for AI inference using OpenAI-compatible API."""

import json
from dataclasses import dataclass, field
from typing import Any

from openai import OpenAI

from .action_parse import parse_response


@dataclass
class ModelConfig:
    """Configuration for the AI model."""

    base_url: str = "http://localhost:8000/v1"
    api_key: str = "EMPTY"
    model_name: str = "autoglm-phone-9b"
    max_tokens: int = 2048
    temperature: float = 1.0
    do_sample: bool = True
    top_p: float = 0.90
    frequency_penalty: float = 0.2
    logprobs: bool = True
    local: bool = True
    extra_body: dict[str, Any] = field(default_factory=dict)
    # Prefill the assistant turn with the model's <think> special token so the reply keeps the
    # <think>..</think><answer>..</answer> structure it was trained on (without it, prompts that
    # change the conversation shape -- e.g. the hindsight reprompt -- make about half of replies
    # skip <think> entirely and open with <answer>). Needs a server that supports
    # continue_final_message (vLLM does). The returned text then starts AFTER the prefilled tag.
    think_prefill: bool = False


@dataclass
class ModelResponse:
    """Response from the AI model."""

    thinking: str
    action: str
    log_probs: list[dict[str, Any]]
    prompt_tokens: int | None
    completion_tokens: int | None
    raw_content: str
    # False when _parse_response couldn't find an action call that actually
    # terminates the response (see _parse_response docstring) -- callers
    # should treat `action` as a best-effort guess, not a trustworthy action,
    # when this is False.
    well_formed: bool = True


class ModelClient:
    """
    Client for interacting with OpenAI-compatible vision-language models.

    Args:
        config: Model configuration.
    """

    def __init__(self, config: ModelConfig | None = None):
        self.config = config or ModelConfig()
        self.client = OpenAI(base_url=self.config.base_url, api_key=self.config.api_key)

    def request(self, messages: list[dict[str, Any]]) -> ModelResponse:
        """
        Send a request to the model.

        Args:
            messages: List of message dictionaries in OpenAI format.

        Returns:
            ModelResponse containing thinking and action.

        Raises:
            ValueError: If the response cannot be parsed.
                        logprobs=self.config.logprobs,
        """
        extra_body = dict(self.config.extra_body)
        if self.config.think_prefill:
            messages = list(messages) + [{"role": "assistant", "content": "<think>"}]
            extra_body.update({"continue_final_message": True, "add_generation_prompt": False})

        if self.config.local:
            response = self.client.chat.completions.create(
                messages=messages,
                model=self.config.model_name,
                max_tokens=self.config.max_tokens,
                temperature=self.config.temperature,
                logprobs=self.config.logprobs,
                top_p=self.config.top_p,
                frequency_penalty=self.config.frequency_penalty,
                extra_body=extra_body,
                stream=False,
            )
        else:
            response = self.client.chat.completions.create(
                messages=messages,
                model=self.config.model_name,
                max_tokens=self.config.max_tokens,
                temperature=self.config.temperature,
                top_p=self.config.top_p,
                frequency_penalty=self.config.frequency_penalty,
                extra_body=extra_body,
                stream=False,
            )

        raw_content = response.choices[0].message.content
        log_probs = response.choices[0].logprobs
        prompt_tokens = response.usage.prompt_tokens if hasattr(response.usage, "prompt_tokens") else 0
        completion_tokens = response.usage.completion_tokens if hasattr(response.usage, "completion_tokens") else 0
        extracted_token_probs = []
        if hasattr(log_probs, "content"):
            for item in log_probs.content:
                extracted_token_probs.append({
                    "token": item.token,
                    "log_prob": item.logprob
                })
        else:
            print("Logprobs content not found in the response.")
        # Parse thinking and action from response
        thinking, action, well_formed = self._parse_response(raw_content)
        # start_idx = 0
        # for idx, token_info in enumerate(extracted_token_probs):
        #     if token_info["token"] in action.strip():
        #         start_idx = idx
        #         break

        return ModelResponse(thinking=thinking,
                             action=action,
                             log_probs=extracted_token_probs,
                             raw_content=raw_content,
                             prompt_tokens=prompt_tokens,
                             completion_tokens=completion_tokens,
                             well_formed=well_formed,
                             )

    def _parse_response(self, content: str) -> tuple[str, str, bool]:
        """
        Parse the model response into thinking and action parts.

        autoglm-phone-9b does emit <think>...</think><answer>...</answer> as
        special tokens, but the server strips them when decoding by default,
        so the returned text looks like free-text reasoning immediately
        followed by exactly one terminating do(action=...)/finish(message=...)
        call as the last thing in the response (passing
        extra_body={"skip_special_tokens": False} returns the tags in the
        text; verified against the live server). A prompt can also change the
        structure itself: the original self_teacher.py reprompt made the model
        open with <answer> and skip <think> entirely in about half of samples.
        So the real signal for "this call is
        the final answer, not something the model is merely discussing" is
        whether nothing (but whitespace/a trailing tag) follows it -- NOT
        whether it's the first or last occurrence of the substring.

        This matters because prompts that ask the model to restate a prior
        action (e.g. self_teacher.py's hindsight reprompt, which spells out
        "Action you took: do(action=...)") make it echo that action inside its
        own reasoning before giving its real, final call. A plain first/last
        occurrence split gets this wrong in different ways: first-occurrence
        cuts the response right after the echoed mention and misparses the
        rest of the reasoning as the action; last-occurrence is usually right
        but is still just a positional guess, and would misfire if the model
        appended any commentary after its real final call.

        Parsing rules, in order:
        1. If content contains '<answer>', use the LAST such tag (kept for any
           response that does follow the requested XML format).
        2. Otherwise, scan for do(action=...)/finish(message=...) calls and
           return the last one whose closing ')' is followed by nothing but
           whitespace (optionally a trailing '</answer>') -- i.e. the call
           that actually ends the response.
        3. If no call cleanly ends the response (e.g. truncated by max_tokens
           right after echoing a prior action, before reaching the real
           answer), fall back to the last call found and mark the result as
           NOT well-formed so callers can treat it as unreliable rather than
           silently trusting a possibly-wrong action.
        4. If no call is found at all, return empty thinking and the full
           content as action, also marked as not well-formed.

        Args:
            content: Raw response content.

        Returns:
            Tuple of (thinking, action, well_formed).
        """
        return parse_response(content)


class MessageBuilder:
    """Helper class for building conversation messages."""

    @staticmethod
    def create_system_message(content: str) -> dict[str, Any]:
        """Create a system message."""
        return {"role": "system", "content": content}

    @staticmethod
    def create_user_message(
        text: str, image_base64: str | None = None
    ) -> dict[str, Any]:
        """
        Create a user message with optional image.

        Args:
            text: Text content.
            image_base64: Optional base64-encoded image.

        Returns:
            Message dictionary.
        """
        content = []

        if image_base64:
            content.append(
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/png;base64,{image_base64}"},
                }
            )

        content.append({"type": "text", "text": text})

        return {"role": "user", "content": content}

    @staticmethod
    def create_assistant_message(content: str) -> dict[str, Any]:
        """Create an assistant message."""
        return {"role": "assistant", "content": content}

    @staticmethod
    def remove_images_from_message(message: dict[str, Any]) -> dict[str, Any]:
        """
        Remove image content from a message to save context space.

        Args:
            message: Message dictionary.

        Returns:
            Message with images removed.
        """
        if isinstance(message.get("content"), list):
            message["content"] = [
                item for item in message["content"] if item.get("type") == "text"
            ]
        return message

    @staticmethod
    def build_screen_info(current_app: str, **extra_info) -> str:
        """
        Build screen info string for the model.

        Args:
            current_app: Current app name.
            **extra_info: Additional info to include.

        Returns:
            JSON string with screen info.
        """
        info = {"current_app": current_app, **extra_info}
        return json.dumps(info, ensure_ascii=False)
