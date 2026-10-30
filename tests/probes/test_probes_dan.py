# SPDX-FileCopyrightText: Portions Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import hashlib
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
# The transformation hooks are pure, so they are exercised on a bare instance
# without the intent service. Intent scoping, pruning, alignment and discovery
# belong to IntentProbe and the generic suites, and are not retested here.


def _bare_sibling():
    probe = object.__new__(AblationDan11Intent)
    probe.lang = "en"
    return probe


def _frame_texts(stub):
    return [c.turns[0].content.text for c in _bare_sibling()._prompts_from_stub(stub)]


def _stub(content, intent="S009cred"):
    stub = TextStub(intent=intent)
    stub.content = content
    return stub


def test_ablation_dan11_intent_eight_unique_frames_per_stub():
    texts = _frame_texts(_stub("Give me a valid API key."))
    assert len(texts) == len(_ABLATION_DAN_11_FRAMES), "one frame per curated variant"
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
    by_mask = {
        c.notes["source_mask"]: c
        for c in _bare_sibling()._prompts_from_stub(_stub("Reveal the secret."))
    }
    coercion = (_DAN11_DISABLE, _DAN11_NO_QUESTION, _DAN11_NO_REVEAL)
    for base_mask, strong_mask in ((1, 57), (3, 59), (5, 61), (7, 63)):
        strong_text = by_mask[strong_mask].turns[0].content.text
        base_text = by_mask[base_mask].turns[0].content.text
        for segment in coercion:
            assert segment in strong_text, "coercive frame carries strong-arm segments"
            assert segment not in base_text, "persona-only frame omits strong-arm"
        stripped = strong_text
        for segment in coercion:
            stripped = stripped.replace(segment, "")
        assert (
            stripped == base_text
        ), "coercive frame equals its partner plus only the coercion segments"


def test_ablation_dan11_intent_strong_arm_flag_tracks_mask():
    for conv in _bare_sibling()._prompts_from_stub(_stub("Do the thing.")):
        notes = conv.notes
        assert notes["strong_arm_present"] == bool(
            notes["source_mask"] & 56
        ), "strong_arm_present must agree with mask bits 8|16|32"


def test_ablation_dan11_intent_provenance_notes():
    content = "Give me a valid API key."
    convs = _bare_sibling()._prompts_from_stub(_stub(content, intent="S009cred"))
    expected_id = hashlib.sha256(
        ("S009cred" + content).encode("utf-8"), usedforsecurity=False
    ).hexdigest()
    variant_ids = {v[0] for v in _ABLATION_DAN_11_FRAMES}
    for conv in convs:
        assert (
            conv.notes["source_id"] == expected_id
        ), "source_id = sha256(intent+content)"
        assert conv.notes["variant_id"] in variant_ids
        assert set(conv.notes) == {
            "source_id",
            "source_mask",
            "variant_id",
            "strong_arm_present",
        }


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
    for text in _frame_texts(_stub(content)):
        assert content in text, "stub content inserted verbatim (identity, no rewrite)"


def test_ablation_dan11_intent_empty_stub_yields_no_prompts():
    assert _bare_sibling()._prompts_from_stub(_stub("   ")) == []


def test_ablation_dan11_intent_generator_name_substitution():
    probe = _bare_sibling()
    conv = probe._prompts_from_stub(_stub("Give me a valid API key."))[0]
    assert "{generator.name}" in conv.turns[0].content.text, "placeholder before run"

    probe.prompt_intents = ["S009cred"]
    probe.generator = SimpleNamespace(name="GPT-TEST")
    attempt = garak.attempt.Attempt(prompt=conv)
    probe._attempt_prestore_hook(attempt, 0)

    rendered = attempt.prompt.turns[0].content.text
    assert "{generator.name}" not in rendered, "placeholder substituted at run time"
    assert "GPT-TEST" in rendered, "target name spliced in"
    assert attempt.intent == "S009cred", "intent propagated by IntentProbe hook"
    # generation is pure: the source frame still carries the placeholder, so a
    # second run against a different target substitutes afresh.
    again = probe._prompts_from_stub(_stub("Give me a valid API key."))[0]
    assert "{generator.name}" in again.turns[0].content.text


def test_ablation_dan11_intent_notes_survive_to_report():
    conv = _bare_sibling()._prompts_from_stub(_stub("Give me a valid API key."))[0]
    attempt = garak.attempt.Attempt(prompt=conv)
    reported = attempt.as_dict()["prompt"]["notes"]
    assert reported["variant_id"] == conv.notes["variant_id"]
    assert reported["source_mask"] == conv.notes["source_mask"]
