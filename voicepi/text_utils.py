from __future__ import annotations

import re
import unicodedata

_SENTENCE_END_RE = re.compile(r"(.+?[.!?…]+)(?:\s+|$)", re.DOTALL)
_TAG_RE = re.compile(r"<[^>]+>")
_SPACE_RE = re.compile(r"\s+")
_URL_RE = re.compile(r"https?://\S+|www\.\S+", re.IGNORECASE)
_EMOJI_RE = re.compile(r"[\U00010000-\U0010ffff]", flags=re.UNICODE)

_TTS_WORD_REPLACEMENTS: tuple[tuple[re.Pattern[str], str], ...] = (
    (re.compile(r"\bLLM\b"), "L L M"),
    (re.compile(r"\bAI\b"), "A I"),
    (re.compile(r"\bTTS\b"), "T T S"),
    (re.compile(r"\bSTT\b"), "S T T"),
    (re.compile(r"\bVAD\b"), "V A D"),
    (re.compile(r"\bJSON\b"), "J S O N"),
    (re.compile(r"\bESP32\b", re.IGNORECASE), "E S P thirty two"),
    (re.compile(r"\bGPIO\b"), "G P I O"),
    (re.compile(r"\bFPS\b"), "frames per second"),
    (re.compile(r"\bms\b"), "milliseconds"),
)


def _strip_control_markup(text: str) -> str:
    text = text or ""
    text = re.sub(r"<rationale\s*>.*?</rationale\s*>", "", text, flags=re.DOTALL | re.IGNORECASE)
    text = re.sub(r"<tool_call\s*>.*?</tool_call\s*>", "", text, flags=re.DOTALL | re.IGNORECASE)
    text = re.sub(r"</?(?:answer|rationale|tool_call|tool|json)[^>]*>", "", text, flags=re.IGNORECASE)
    text = _TAG_RE.sub("", text)
    # Remove a dangling partial tag left by a streaming chunk, e.g. "hello </ans".
    text = re.sub(r"<[^\s<>]*$", "", text)
    return text


def clean_for_tts(text: str) -> str:
    """Return text safe and pleasant to send to Piper.

    This is intentionally more aggressive than normal UI cleanup. It removes XML-ish
    LLM tags, markdown bullets/code markers, emojis, URLs, and isolated punctuation
    that Piper can sometimes pronounce literally, for example "exclamation mark".
    """
    text = unicodedata.normalize("NFKC", text or "")
    text = _strip_control_markup(text)
    text = _URL_RE.sub(" a link ", text)
    text = _EMOJI_RE.sub("", text)

    # Markdown / code / list noise.
    text = re.sub(r"```.*?```", " ", text, flags=re.DOTALL)
    text = re.sub(r"`([^`]*)`", r"\1", text)
    text = re.sub(r"^\s*[-*+]\s+", "", text, flags=re.MULTILINE)
    text = re.sub(r"[*_#>\[\]{}]", "", text)
    text = re.sub(r"\b(?:Answer|Response)\s*:\s*", "", text, flags=re.IGNORECASE)

    # Spoken-friendly symbols. Keep words rather than letting Piper guess symbols.
    text = text.replace("&", " and ")
    text = text.replace("@", " at ")
    text = text.replace("%", " percent ")
    text = text.replace("°", " degrees ")
    text = text.replace("+", " plus ")
    text = text.replace("=", " equals ")
    text = re.sub(r"(?<=\d)\s*/\s*(?=\d)", " over ", text)

    # Piper can read a lonely ! as a literal punctuation name. Use periods for speech.
    text = re.sub(r"!+", ".", text)
    text = re.sub(r"[;:]+", ",", text)
    text = re.sub(r"[\u2018\u2019]", "'", text)
    text = re.sub(r"[\u201c\u201d]", '"', text)
    text = re.sub(r"[\u2013\u2014]", ", ", text)
    text = text.replace("…", ".")

    # Remove punctuation-only fragments such as " . " or " ? " from streaming chunks.
    text = re.sub(r"(?:^|\s)[.!?,]+(?=\s|$)", " ", text)
    text = re.sub(r"[<>]", "", text)

    for pattern, replacement in _TTS_WORD_REPLACEMENTS:
        text = pattern.sub(replacement, text)

    # Avoid long awkward silences from repeated punctuation.
    text = re.sub(r"\.{2,}", ".", text)
    text = re.sub(r",{2,}", ",", text)
    text = _SPACE_RE.sub(" ", text)
    return text.strip(" ,")


def normalize_transcript(text: str) -> str:
    text = (text or "").strip().lower()
    text = re.sub(r"[^a-z0-9'\-\s]", " ", text)
    text = _SPACE_RE.sub(" ", text)
    return text.strip()


def is_rejected_transcript(text: str, min_chars: int, min_words: int, reject_phrases: list[str]) -> bool:
    norm = normalize_transcript(text)
    if len(norm) < min_chars:
        return True
    words = [w for w in norm.split() if w]
    if len(words) < min_words:
        return True
    return norm in {normalize_transcript(p) for p in reject_phrases}


def extract_tag(raw: str, tag: str) -> str:
    m = re.search(rf"<{tag}\s*>(.*?)</{tag}\s*>", raw or "", flags=re.DOTALL | re.IGNORECASE)
    return (m.group(1).strip() if m else "")


def extract_tag_progress(raw: str, tag: str) -> str:
    start = re.search(rf"<{tag}\s*>", raw or "", flags=re.IGNORECASE)
    if not start:
        return ""
    content = raw[start.end():]
    end = re.search(rf"</{tag}\s*>", content, flags=re.IGNORECASE)
    if end:
        content = content[: end.start()]
    # Never expose or speak a partial closing tag while streaming.
    content = re.sub(r"</?[^>]*$", "", content)
    return content


def strip_llm_markup(raw: str) -> str:
    text = extract_tag(raw, "answer") or (raw or "")
    text = re.sub(r"<rationale\s*>.*?</rationale\s*>", "", text, flags=re.DOTALL | re.IGNORECASE)
    text = re.sub(r"<tool_call\s*>.*?</tool_call\s*>", "", text, flags=re.DOTALL | re.IGNORECASE)
    text = _TAG_RE.sub("", text)
    text = re.sub(r"^\s*(?:Answer|Response)\s*:\s*", "", text, flags=re.IGNORECASE)
    text = _SPACE_RE.sub(" ", text)
    return text.strip()


def fallback_visible_rationale(user_text: str, answer: str) -> str:
    cleaned_user = normalize_transcript(user_text)
    if not cleaned_user:
        return "No reliable user transcript was available, so I could not infer intent confidently."
    if len(cleaned_user.split()) <= 2:
        return "The transcript was very short, so I treated it as potentially ambiguous and kept the response cautious."
    return "I interpreted the speech transcript as a direct conversational request and answered concisely without external sources."


class StreamingAnswerExtractor:
    """Streaming parser for the <answer> section.

    It emits only user-speakable answer text. Unlike regex progress extraction,
    it does not leak the closing tag while the model is still streaming tokens.
    """

    def __init__(self, answer_tag: str = "answer") -> None:
        self.answer_tag = answer_tag.lower()
        self.state = "before"  # before | in_answer | maybe_tag | done
        self.tag_buf = ""
        self.open_re = re.compile(rf"<\s*{re.escape(self.answer_tag)}\s*>", re.IGNORECASE)
        self.close_re = re.compile(rf"<\s*/\s*{re.escape(self.answer_tag)}\s*>", re.IGNORECASE)

    def feed(self, token: str) -> str:
        out: list[str] = []
        for ch in token or "":
            if self.state == "done":
                continue

            if self.state == "before":
                self.tag_buf += ch
                m = self.open_re.search(self.tag_buf)
                if m:
                    self.state = "in_answer"
                    remainder = self.tag_buf[m.end():]
                    self.tag_buf = ""
                    if remainder:
                        out.append(self.feed(remainder))
                elif len(self.tag_buf) > 512:
                    # The model missed <answer>. Fall back to speaking plain text after a small buffer.
                    out.append(clean_for_tts(self.tag_buf))
                    self.tag_buf = ""
                    self.state = "in_answer"

            elif self.state == "in_answer":
                if ch == "<":
                    self.tag_buf = ch
                    self.state = "maybe_tag"
                else:
                    out.append(ch)

            elif self.state == "maybe_tag":
                self.tag_buf += ch
                if self.close_re.fullmatch(self.tag_buf):
                    self.tag_buf = ""
                    self.state = "done"
                    continue
                # Still could be a closing tag, wait for more characters.
                if self.tag_buf.lower().startswith("</answer") and ">" not in self.tag_buf:
                    continue
                if self.tag_buf.endswith(">"):
                    # Some other tag inside answer. Drop it from speech.
                    self.tag_buf = ""
                    self.state = "in_answer"
                    continue
                if len(self.tag_buf) > 32:
                    # Probably a literal '<' expression, not a tag.
                    out.append(self.tag_buf)
                    self.tag_buf = ""
                    self.state = "in_answer"

        return "".join(out)


class SentenceBuffer:
    def __init__(self, min_chars: int = 22, max_chars: int = 220) -> None:
        self.buf = ""
        self.min_chars = min_chars
        self.max_chars = max_chars

    def feed(self, text: str) -> list[str]:
        self.buf += text
        out: list[str] = []
        while True:
            m = _SENTENCE_END_RE.match(self.buf)
            if not m:
                if len(self.buf) >= self.max_chars:
                    cut = self.buf.rfind(" ", 0, self.max_chars)
                    if cut < self.min_chars:
                        cut = self.max_chars
                    sent = self.buf[:cut].strip()
                    self.buf = self.buf[cut:].lstrip()
                    if sent:
                        out.append(sent)
                    continue
                break
            sent = m.group(1).strip()
            rest = self.buf[m.end():]
            if len(sent) < self.min_chars and len(rest) < self.max_chars:
                break
            self.buf = rest
            if sent:
                out.append(sent)
        return out

    def flush(self) -> str:
        text = self.buf.strip()
        self.buf = ""
        return text
