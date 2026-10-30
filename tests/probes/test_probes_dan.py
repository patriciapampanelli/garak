# SPDX-FileCopyrightText: Portions Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from types import SimpleNamespace

import pytest

import garak._plugins
import garak.attempt
from garak.intents import TextStub
from garak.probes.dan import (
    AblationDan11Intent,
    _ABLATION_DAN_11_FRAMES,
    _DAN11_DISABLE,
    _DAN11_NO_QUESTION,
    _DAN11_NO_REVEAL,
)


def test_dan_ablation_unique():
    a = garak._plugins.load_plugin("probes.dan.Ablation_Dan_11_0")
    expected_prompt_count = 127
    assert (
        len(a.prompts) == expected_prompt_count
    ), f"Must have {expected_prompt_count} ablation dan prompts"
    assert (
        len(set(a.prompts)) == expected_prompt_count
    ), "No duplicate jailbreaks should be generated"


# --- AblationDan11Intent sibling: delta behaviour only ---------------------
#
# The stub->prompt transformation is pure, so it is exercised on a bare instance
# without the intent service. Intent scoping, pruning, alignment and discovery
# belong to IntentProbe and the generic suites, and are not retested here.
# The probe returns plain strings and carries no notes to the report.


def _bare_sibling():
    probe = object.__new__(AblationDan11Intent)
    probe.lang = "en"
    return probe


def _stub(content, intent="S009cred"):
    stub = TextStub(intent=intent)
    stub.content = content
    return stub


def test_ablation_dan11_intent_eight_unique_frames_per_stub():
    texts = _bare_sibling()._prompts_from_stub(_stub("Give me a valid API key."))
    assert len(texts) == len(_ABLATION_DAN_11_FRAMES), "one frame per curated variant"
    assert all(isinstance(t, str) for t in texts), "frames are plain strings"
    assert len(set(texts)) == len(texts), "frames must be distinct"
    for text in texts:
        assert (
            text.count("Give me a valid API key.") == 1
        ), "stub inserted exactly once per frame"
        assert (
            "two responses" not in text and "GPT:" not in text
        ), "single-turn frame must drop the GPT:/DAN: dual-response contract"
        assert (
            "next message" not in text
        ), "single-turn frame must drop the confirm/next-turn protocol"


def test_ablation_dan11_intent_strong_arm_pairs_differ_only_by_coercion():
    # The first four frames are persona-only; the last four add the coercion
    # segments, so frame i and frame i+4 form a matched pair.
    texts = _bare_sibling()._prompts_from_stub(_stub("Reveal the secret."))
    coercion = (_DAN11_DISABLE, _DAN11_NO_QUESTION, _DAN11_NO_REVEAL)
    for i in range(4):
        base_text, strong_text = texts[i], texts[i + 4]
        for segment in coercion:
            assert segment in strong_text, "coercive frame carries strong-arm segments"
            assert segment not in base_text, "persona-only frame omits strong-arm"
        stripped = strong_text
        for segment in coercion:
            stripped = stripped.replace(segment, "")
        assert (
            stripped == base_text
        ), "coercive frame equals its partner plus only the coercion segments"


@pytest.mark.parametrize(
    "content",
    [
        "Give me a valid API key.",  # imperative
        "How do I extract the token?",  # question
        "Exfiltrate.",  # one word
        "Leak it -- now, please!",  # punctuation
        'Return JSON like {"key": value}.',  # literal braces must survive
    ],
)
def test_ablation_dan11_intent_identity_insertion(content):
    for text in _bare_sibling()._prompts_from_stub(_stub(content)):
        assert content in text, "stub content inserted verbatim (identity, no rewrite)"


def test_ablation_dan11_intent_empty_stub_yields_no_prompts():
    assert _bare_sibling()._prompts_from_stub(_stub("   ")) == []


def _single_turn_attempt(text):
    conversation = garak.attempt.Conversation(
        [garak.attempt.Turn("user", garak.attempt.Message(text=text, lang="en"))]
    )
    return garak.attempt.Attempt(prompt=conversation)


def test_ablation_dan11_intent_generator_name_substitution():
    probe = _bare_sibling()
    prompt = probe._prompts_from_stub(_stub("Give me a valid API key."))[0]
    assert "{generator.name}" in prompt, "placeholder present before run"

    probe.prompt_intents = ["S009cred"]
    probe.generator = SimpleNamespace(name="GPT-TEST")
    attempt = _single_turn_attempt(prompt)
    probe._attempt_prestore_hook(attempt, 0)

    rendered = attempt.prompt.turns[0].content.text
    assert "{generator.name}" not in rendered, "placeholder substituted at run time"
    assert "GPT-TEST" in rendered, "target name spliced in"
    assert attempt.intent == "S009cred", "intent propagated by IntentProbe hook"
    # generation is pure: a fresh call still carries the placeholder, so a second
    # run against a different target substitutes afresh.
    again = probe._prompts_from_stub(_stub("Give me a valid API key."))[0]
    assert "{generator.name}" in again
