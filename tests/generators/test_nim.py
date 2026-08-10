# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import os
import pytest
import httpx
import openai
from unittest.mock import MagicMock, patch

import garak.exception
from garak.attempt import Message, Turn, Conversation
from garak.exception import GarakException
import garak.cli
from garak.generators.nim import NVOpenAIChat
from garak.generators.openai import OpenAICompatible


def _make_api_status_error(
    status_code: int, url: str = "http://localhost/v1/chat/completions"
) -> openai.APIStatusError:
    """Build an openai.APIStatusError with a real httpx.Response for a given HTTP status."""
    request = httpx.Request("POST", url)
    response = httpx.Response(status_code, request=request)
    return openai.APIStatusError(f"HTTP {status_code}", response=response, body=None)


def _make_prompt() -> Conversation:
    return Conversation([Turn(role="user", content=Message("test prompt"))])


@pytest.fixture
def nim_generator(monkeypatch):
    """NVOpenAIChat with mocked client; no real API key required."""
    monkeypatch.setenv(NVOpenAIChat.ENV_VAR, "test-fake-key-for-unit-tests")
    mock_client = MagicMock()
    mock_client.chat.completions = MagicMock()
    with patch("openai.OpenAI", return_value=mock_client):
        g = NVOpenAIChat(name="org/test-model")
    return g


def test_transient_retry_codes_in_default_params():
    """transient_retry_codes should be present in DEFAULT_PARAMS and contain expected codes."""
    codes = OpenAICompatible.DEFAULT_PARAMS["transient_retry_codes"]
    for code in (408, 429, 502, 503, 504):
        assert code in codes, f"Expected {code} in transient_retry_codes"
    for code in (400, 401, 403, 404):
        assert code not in codes, f"Expected {code} NOT in transient_retry_codes"


@pytest.mark.parametrize("code", [408, 502, 503, 504])
def test_transient_http_error_raises_backoff_trigger(nim_generator, code):
    """A transient status code should cause _call_model to raise GeneratorBackoffTrigger
    so that the backoff decorator can schedule a retry."""
    prompt = _make_prompt()
    nim_generator.generator = MagicMock()
    nim_generator.generator.create.side_effect = _make_api_status_error(code)

    # Call the underlying function without the backoff decorator so the test does not
    # need to wait for retry delays or exhaust the fibonacci sequence.
    unwrapped = OpenAICompatible._call_model.__wrapped__
    with pytest.raises(garak.exception.GeneratorBackoffTrigger):
        unwrapped(nim_generator, prompt)


@pytest.mark.parametrize("code", [400, 403, 404, 422])
def test_terminal_http_error_returns_none(nim_generator, code):
    """A terminal (non-transient) status code should cause _call_model to return [None]
    so that the current attempt is skipped and the probe run continues."""
    prompt = _make_prompt()
    nim_generator.generator = MagicMock()
    nim_generator.generator.create.side_effect = _make_api_status_error(code)

    unwrapped = OpenAICompatible._call_model.__wrapped__
    result = unwrapped(nim_generator, prompt)
    assert result == [None], f"Expected [None] for HTTP {code}, got {result!r}"


def test_nim_generic_exception_message_improved(nim_generator):
    """Non-APIStatusError exceptions should report the exception type, not the old
    generic 'Is the model name spelled correctly?' message."""
    prompt = _make_prompt()

    with patch(
        "garak.generators.openai.OpenAICompatible._call_model",
        side_effect=RuntimeError("connection reset by peer"),
    ):
        with pytest.raises(GarakException) as exc_info:
            nim_generator._call_model(prompt)

    error_text = str(exc_info.value)
    assert "Is the model name spelled correctly?" not in error_text


@pytest.mark.skipif(
    os.getenv(NVOpenAIChat.ENV_VAR, None) is None,
    reason=f"NIM API key is not set in {NVOpenAIChat.ENV_VAR}",
)
def test_nim_instantiate():
    g = NVOpenAIChat(name="google/gemma-2b")


@pytest.mark.skipif(
    os.getenv(NVOpenAIChat.ENV_VAR, None) is None,
    reason=f"NIM API key is not set in {NVOpenAIChat.ENV_VAR}",
)
def test_nim_generate_1():
    g = NVOpenAIChat(name="google/gemma-2b")
    result = g._call_model(
        Conversation([Turn(role="user", content=Message("this is a test"))])
    )
    assert isinstance(result, list), "NIM _call_model should return a list"
    assert len(result) == 1, "NIM _call_model result list should have one item"
    assert isinstance(result[0], Message), "NIM _call_model should return a list"
    result = g.generate(
        Conversation([Turn(role="user", content=Message("this is a test"))])
    )
    assert isinstance(result, list), "NIM generate() should return a list"
    assert (
        len(result) == 1
    ), "NIM generate() result list should have one item using default generations_this_call"
    assert isinstance(
        result[0], Message
    ), "NIM generate() should return a list of Turns"


@pytest.mark.skipif(
    os.getenv(NVOpenAIChat.ENV_VAR, None) is None,
    reason=f"NIM API key is not set in {NVOpenAIChat.ENV_VAR}",
)
def test_nim_parallel_attempts():
    garak.cli.main(
        "-m nim -p lmrc.Anthropomorphisation -g 1 -n google/gemma-2b --parallel_attempts 10".split()
    )
    assert True


@pytest.mark.skipif(
    os.getenv(NVOpenAIChat.ENV_VAR, None) is None,
    reason=f"NIM API key is not set in {NVOpenAIChat.ENV_VAR}",
)
def test_nim_hf_detector():
    garak.cli.main("-m nim -p lmrc.Bullying -g 1 -n google/gemma-2b".split())
    assert True


@pytest.mark.skipif(
    os.getenv(NVOpenAIChat.ENV_VAR, None) is None,
    reason=f"NIM API key is not set in {NVOpenAIChat.ENV_VAR}",
)
def test_nim_conservative_api():  # extraneous params can throw 422
    g = NVOpenAIChat(name="nvidia/nemotron-4-340b-instruct")
    result = g._call_model(
        Conversation([Turn(role="user", content=Message("this is a test"))])
    )
    assert isinstance(result, list), "NIM _call_model should return a list"
    assert len(result) == 1, "NIM _call_model result list should have one item"
    assert isinstance(
        result[0], Message
    ), "NIM _call_model should return a list of Messages"
    result = g.generate(
        Conversation([Turn(role="user", content=Message("this is a test"))])
    )
    assert isinstance(result, list), "NIM generate() should return a list"
    assert (
        len(result) == 1
    ), "NIM generate() result list should have one item using default generations_this_call"
    assert isinstance(
        result[0], Message
    ), "NIM generate() should return a list of Turns"


def test_nim_vision_prep():
    test_prompt = "test vision prompt"
    t = Conversation(
        [
            Turn(
                "user",
                Message(text=test_prompt, data_path="tests/_assets/tinytrans.gif"),
            )
        ]
    )
    from garak.generators.nim import Vision

    v = Vision  # skip instantiation, not req'd
    setattr(v, "max_input_len", 100_000)
    setattr(v, "embed_data", True)
    vision_conv = Vision._prepare_prompt(v, t)
    assert (
        vision_conv.last_message().text
        == test_prompt
        + ' <img src="data:image/gif;base64,R0lGODlhAQABAIABAP///wAAACH5BAEKAAEALAAAAAABAAEAAAICTAEAOw==" />'
    )
    delattr(v, "max_input_len")  # remove to avoid follow on test impacts
    delattr(v, "embed_data")
