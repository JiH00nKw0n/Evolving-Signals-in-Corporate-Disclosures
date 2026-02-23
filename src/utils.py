import re
from enum import Enum
from typing import List, Dict, Tuple

import numpy as np
import pandas as pd
from bs4 import BeautifulSoup
from nltk import sent_tokenize


class SpeakerType(Enum):
    """Conference call speaker types"""
    EXECUTIVE = "executive"
    ANALYST = "analyst"
    OPERATOR = "operator"
    UNKNOWN = "unknown"


class ConferenceSection(Enum):
    """Conference call section classification"""
    PRESENTATION = "presentation"
    QA = "qa"
    UNKNOWN = "unknown"


def listdict_to_indexed_string_nltk_2(blocks: List[Dict[str, str]]) -> tuple[str, dict]:
    """
    [{'speaker': str, 'content': str}, ...] ->
    <speaker>{speaker}/<speaker>\n**index 0**: "..."
    """

    def split_sentences(text: str) -> List[str]:
        t = re.sub(r"\s+", " ", (text or "")).strip()
        if not t:
            return []
        sents = sent_tokenize(t)
        return [s.strip() for s in sents if s.strip()]

    lines: List[str] = []
    index_dict = dict()
    i = 0
    for item in blocks:
        speaker = "<speaker>" + (item.get("speaker") or "Unknown").strip() + "/<speaker>"
        content = item.get("content") or ""
        sents = split_sentences(content)

        lines.append(speaker)
        for s in sents:
            lines.append(f'**index {i}**: "{s}"')
            index_dict[i] = s
            i += 1

    return "\n".join(lines), index_dict


def chunk_earnings_html(text: str, include_metadata: bool = False) -> List[Dict[str, str]]:
    """
    Parse an earnings-call HTML transcript into a list of dicts.

    Args:
        text: HTML content of earnings call
        include_metadata: If True, includes speaker_type and section metadata

    Returns:
        List of dicts with keys: speaker, content, and optionally speaker_type, section
    """
    soup = BeautifulSoup(text, "html.parser")
    elements = soup.find_all(["strong", "p"])

    chunks: List[Dict[str, str]] = []
    current_speaker: str | None = None

    for el in elements:
        if el.name == "strong":
            # Get the text (merging nested spans like <span>-</span>)
            speaker_text = el.get_text(" ", strip=True)

            # Normalize separators " - " or " – "
            parts = [p.strip() for p in re.split(r"\s*[-–]\s*", speaker_text, maxsplit=1)]
            if len(parts) == 2 and parts[0] and parts[1]:
                current_speaker = f"{parts[0]} - {parts[1]}"
            else:
                current_speaker = speaker_text or "(Unknown)"

        elif el.name == "p":
            content = el.get_text("\n", strip=True)
            if not content:
                continue
            speaker = current_speaker or "(Unknown)"
            chunks.append({"speaker": speaker, "content": content})

    # Add metadata if requested
    if include_metadata:
        for i, chunk in enumerate(chunks):
            chunk['speaker_type'] = classify_speaker_type(chunk['speaker']).value
            chunk['section'] = identify_conference_section(chunks, i).value

    return chunks


def classify_speaker_type(speaker: str) -> SpeakerType:
    """Classify speaker type"""
    speaker_lower = speaker.lower()

    if "operator" in speaker_lower:
        return SpeakerType.OPERATOR
    elif any(keyword in speaker_lower for keyword in ["executive", "ceo", "cfo", "president", "chairman"]):
        return SpeakerType.EXECUTIVE
    elif any(keyword in speaker_lower for keyword in ["analyst", "analysts"]):
        return SpeakerType.ANALYST
    else:
        return SpeakerType.UNKNOWN


def identify_conference_section(chunks: List[Dict[str, str]], current_index: int) -> ConferenceSection:
    """Identify conference call section"""

    # Method 1: Check if ANALYST speech has actually appeared (strongest signal)
    analyst_appeared = False
    for i in range(0, current_index + 1):
        if i < len(chunks):
            speaker_type = classify_speaker_type(chunks[i]['speaker'])
            if speaker_type == SpeakerType.ANALYST:
                analyst_appeared = True
                break

    # If ANALYST has already appeared, it's Q&A section
    if analyst_appeared:
        return ConferenceSection.QA

    # Method 2: Check for Q&A start keywords around current chunk
    # (but ignore keywords that appear too early)
    if current_index > len(chunks) * 0.3:  # Check keywords only after 30% of total
        context_start = max(0, current_index - 5)
        for i in range(context_start, current_index + 1):
            if i < len(chunks):
                content = chunks[i]['content'].lower()
                # More specific Q&A start keywords
                if any(
                        phrase in content for phrase in [
                            "take our first question", "take questions",
                            "open for questions", "q&a session",
                            "questions from the audience",
                            "turn to questions", "begin the q&a"
                        ]
                ):
                    return ConferenceSection.QA

    # Method 3: Pattern-based detection - Check ANALYST -> EXECUTIVE pattern
    context_start = max(0, current_index - 2)
    context_end = min(len(chunks), current_index + 2)

    for i in range(context_start, context_end - 1):
        if i < len(chunks) and i + 1 < len(chunks):
            current_speaker = classify_speaker_type(chunks[i]['speaker'])
            next_speaker = classify_speaker_type(chunks[i + 1]['speaker'])

            # ANALYST question -> EXECUTIVE answer pattern
            if current_speaker == SpeakerType.ANALYST and next_speaker == SpeakerType.EXECUTIVE:
                return ConferenceSection.QA

    # Default: PRESENTATION
    return ConferenceSection.PRESENTATION


def get_logprobs_thresholds(csv_path: str, min_threshold: float, max_threshold: float) -> Tuple[float, float]:
    """
    Calculate logprobs thresholds based on percentiles.

    Args:
        csv_path: Path to CSV file containing logprobs column
        min_threshold: Upper percentile threshold (e.g., 0.2 for top 20%)
        max_threshold: Lower percentile threshold (e.g., 0.8 for top 80%)

    Returns:
        Tuple of (min_threshold_score, max_threshold_score)
    """
    df = pd.read_csv(csv_path)
    logprobs = df['logprobs'].dropna().values

    min_percentile = (1 - min_threshold) * 100
    max_percentile = (1 - max_threshold) * 100

    min_score = np.percentile(logprobs, min_percentile)
    max_score = np.percentile(logprobs, max_percentile)

    return min_score, max_score


def listdict_to_indexed_string_nltk(blocks: list[dict]) -> str:
    """
    [{'speaker': str, 'content': str}, ...] ->
    speaker\n**index 0**: "..."
    (NLTK sent_tokenize based)
    """
    import nltk
    try:
        from nltk.tokenize import sent_tokenize
    except LookupError:
        nltk.download('punkt', quiet=True)
        from nltk.tokenize import sent_tokenize
    except Exception:
        nltk.download('punkt', quiet=True)
        from nltk.tokenize import sent_tokenize

    def split_sentences(text: str) -> list[str]:
        t = re.sub(r'\s+', ' ', (text or '')).strip()
        if not t:
            return []
        sents = sent_tokenize(t)
        return [s.strip() for s in sents if s.strip()]

    lines = []
    i = 0
    for item in blocks:
        speaker = "<speaker>" + (item.get('speaker') or 'Unknown').strip() + "/<speaker>"
        content = item.get('content') or ''
        sents = split_sentences(content)

        lines.append(speaker)
        for s in sents:
            lines.append(f'**index {i}**: "{s}"')
            i += 1

    return "\n".join(lines)


if __name__ == "__main__":
    print(
        listdict_to_indexed_string_nltk(
            chunk_earnings_html(
                open("data/earnings/US67066G1040=Q4_2025_NVIDIA_Corporation=2025-02-26.html", "r").read()
            )
        )
    )
