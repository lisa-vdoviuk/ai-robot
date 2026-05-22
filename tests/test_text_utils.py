from voicepi.text_utils import (
    clean_for_tts,
    extract_tag,
    extract_tag_progress,
    fallback_visible_rationale,
    is_rejected_transcript,
    normalize_transcript,
    strip_llm_markup,
)


def test_extract_tag():
    raw = "<rationale>short reason</rationale><answer>Hello there.</answer>"
    assert extract_tag(raw, "rationale") == "short reason"
    assert extract_tag(raw, "answer") == "Hello there."


def test_extract_tag_progress():
    raw = "abc <answer>Hello"
    assert extract_tag_progress(raw, "answer") == "Hello"


def test_clean_for_tts():
    assert clean_for_tts("<answer>Answer: **Hello**</answer>") == "Hello"


def test_strip_llm_markup():
    raw = "<rationale>debug only</rationale><answer>Hi.</answer>"
    assert strip_llm_markup(raw) == "Hi."


def test_reject_noise():
    assert normalize_transcript(" Um! ") == "um"
    assert is_rejected_transcript("um", 2, 1, ["um"])
    assert not is_rejected_transcript("hello there", 2, 1, ["um"])


def test_fallback_visible_rationale():
    assert "transcript" in fallback_visible_rationale("hello", "Hi").lower()


def test_streaming_answer_extractor_does_not_emit_closing_tag():
    from voicepi.text_utils import StreamingAnswerExtractor

    parser = StreamingAnswerExtractor("answer")
    chunks = ["<rationale>x</rationale><answer>Run this script.", "</ans", "wer>"]
    out = "".join(parser.feed(c) for c in chunks)
    assert out == "Run this script."
    assert "answer" not in out
    assert "<" not in out and ">" not in out


def test_clean_for_tts_removes_partial_tags():
    from voicepi.text_utils import clean_for_tts

    assert clean_for_tts("Run it.</answer>") == "Run it."
    assert clean_for_tts("Run it.</ans") == "Run it."
