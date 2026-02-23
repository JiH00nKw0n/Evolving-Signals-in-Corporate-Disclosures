import asyncio
import json
from typing import List, Dict, Any

from pydantic import BaseModel

from src.client import client, get_google_client
from src.utils import chunk_earnings_html

SYSTEM_PROMPT = """
<task>
From the provided inputs, extract all performance metric and target labels
and classify them into two separate lists according to the section
in which they were mentioned:

- presentation: items mentioned in the prepared presentation or opening remarks
- analyst_qa: items mentioned during analyst question and answer sessions

The term "performance metric and target labels" includes every explicitly stated
financial result, non-financial result, and key business indicator discussed or mentioned. Both GAAP and non-GAAP labels if explicitly distinguished.

## Section classification
- presentation: items mentioned in prepared remarks/opening statements before Q&A begins.
  Typical speakers: executives with "- Executives" in the name, before the first analyst question.
- analyst_qa: items mentioned during the Q&A (analyst questions and executive answers after Q&A starts).
  Q&A typically starts after an Operator prompt like "[Operator Instructions]" or when an analyst first speaks
  (speaker label contains "- Analysts") and continues until the call ends.
- Operator-only lines do not themselves contain items, but they can mark Q&A start.
- Include only labels explicitly stated in content
- Deduplicate near-duplicates to one normalized target per section
</task>


<inputs>earnings-call transcript as indexed JSON dialog</inputs>

<output>
- target:
  - Short noun phrase only
  - **Quarter results:** prefix with "Quarterly" and do NOT include any years or quarter numbers
  - **Fiscal-year results:** prefix with "Yearly" and do NOT include any years
  - **Guidance/targets/other indicators:** no period prefixes unless inherently part of the label; avoid dates, years, and quarter numbers
  - No numbers, units, currency symbols, or percent signs anywhere
  - Prefer consistent terminology across products/segments
- index: integer index of the utterance containing the quote
</output>

<format>
{
    "presentation": List[Dict],
    "analyst_qa": List[Dict]
}
The dictionary must follow the following structure:
{
    "target": str # Normalized short label for the metric/target according to the rules
    "index": int # Zero-based index of the utterance in the transcript where the quote appears
}
</format>
"""

USER_PROMPT = """
<transcript>
{transcript}
</transcript>
"""

class TargetItem(BaseModel):
    target: str  # Short normalized metric/target name according to rules
    index: int  # Index where the utterance is located in the transcript


class TargetOutput(BaseModel):
    presentation: List[TargetItem]
    analyst_qa: List[TargetItem]


async def get_targets(
        transcript: str,
        print_usage: bool = False,
        model_name: str = "gpt-5-2025-08-07",
        reasoning_effort: str = "high",
        use_google = True,
) -> TargetOutput:
    if use_google:
        google_client = get_google_client()
        if google_client is None:
            print("Google client not configured, falling back to OpenAI")
            use_google = False

    if use_google:
        from google.genai import types
        reasoning_response = await google_client.aio.models.generate_content(
            model="gemini-2.5-pro",
            config=types.GenerateContentConfig(
                thinking_config=types.ThinkingConfig(thinking_budget=32768),
                system_instruction=SYSTEM_PROMPT,
                temperature=0.0,
                response_mime_type="application/json",
                response_schema=TargetOutput,
            ),
            contents=USER_PROMPT.format(transcript=transcript),
        )
        return reasoning_response.parsed
    else:
        response = await client.beta.chat.completions.parse(
            model=model_name,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": USER_PROMPT.format(transcript=transcript),
                },
            ],
            response_format=TargetOutput,
            reasoning_effort=reasoning_effort,
            seed=2025,
        )
        if print_usage:
            print(response.usage.model_dump())
            print(response.choices[0].message.parsed.model_dump())
        return response.choices[0].message.parsed


def map_llm_targets_to_chunks(llm_result: TargetOutput, chunks: List[Dict[str, str]]) -> Dict[str, Any]:
    """
    Map LLM results by chunks and convert to the same format as base_target
    """
    targets_by_section = {
        "presentation": set(),
        "qa": set(),
        "all": set()
    }

    chunk_entities = []

    # Initialize chunk entities structure
    for i, chunk in enumerate(chunks):
        chunk_data = {
            "index": i,
            "speaker": chunk['speaker'],
            "speaker_type": chunk.get('speaker_type', 'unknown'),
            "section": chunk.get('section', 'unknown'),
            "targets": [],
            "content_preview": chunk['content'][:100] + "..." if len(chunk['content']) > 100 else chunk['content']
        }
        chunk_entities.append(chunk_data)

    # Map presentation targets
    for target_item in llm_result.presentation:
        target_name = target_item.target.lower()
        targets_by_section["presentation"].add(target_name)
        targets_by_section["all"].add(target_name)

        # Add to corresponding chunk
        if target_item.index < len(chunk_entities):
            chunk_entities[target_item.index]["targets"].append(target_name)

    # Map analyst_qa targets (mapped to "qa" section for consistency)
    for target_item in llm_result.analyst_qa:
        target_name = target_item.target.lower()
        targets_by_section["qa"].add(target_name)
        targets_by_section["all"].add(target_name)

        # Add to corresponding chunk
        if target_item.index < len(chunk_entities):
            chunk_entities[target_item.index]["targets"].append(target_name)

    return {
        "targets_by_section": targets_by_section,
        "chunk_entities": chunk_entities
    }


def calculate_llm_statistics(chunks: List[Dict[str, str]], targets_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Calculate statistics from LLM extraction results
    """
    targets_by_section = targets_data["targets_by_section"]

    # Statistics by speaker type
    speaker_stats = {}
    for chunk in chunks:
        speaker_type = chunk.get('speaker_type', 'unknown')
        if speaker_type not in speaker_stats:
            speaker_stats[speaker_type] = 0
        speaker_stats[speaker_type] += 1

    # Statistics by section
    section_stats = {}
    for chunk in chunks:
        section = chunk.get('section', 'unknown')
        if section not in section_stats:
            section_stats[section] = 0
        section_stats[section] += 1

    return {
        "total_targets": len(targets_by_section["all"]),
        "presentation_targets": len(targets_by_section["presentation"]),
        "qa_targets": len(targets_by_section["qa"]),
        "speaker_stats": speaker_stats,
        "section_stats": section_stats,
        "total_chunks": len(chunks)
    }


async def analyze_earnings_call(html_content: str) -> Dict[str, Any]:
    """
    Analyze earnings call HTML with LLM to extract targets and return analysis results
    Same output schema as analyze_earnings_call() from base_target
    """
    # Parse HTML and separate chunks (including metadata)
    chunks = chunk_earnings_html(html_content, include_metadata=True)

    # Convert chunks to transcript format
    transcript = "\n".join([f"**Index {i}**: {chunk}" for i, chunk in enumerate(chunks)])

    # Extract targets using LLM
    llm_result = await get_targets(transcript)

    # Map LLM results to base_target format
    targets_data = map_llm_targets_to_chunks(llm_result, chunks)

    # Calculate statistics
    stats = calculate_llm_statistics(chunks, targets_data)

    # Same output format as base_target
    return {
        "targets_by_section": {k: list(v) for k, v in targets_data["targets_by_section"].items()},
        "total_targets": stats["total_targets"],
        "presentation_targets": stats["presentation_targets"],
        "qa_targets": stats["qa_targets"],
        "speaker_stats": stats["speaker_stats"],
        "section_stats": stats["section_stats"],
        "total_chunks": stats["total_chunks"]
    }


async def analyze_earnings_call_detailed(html_content: str) -> Dict[str, Any]:
    """
    Detailed analysis of earnings call HTML with LLM to extract targets by chunks and overall statistics
    Same output schema as analyze_earnings_call_detailed() from base_target
    """
    # Parse HTML and separate chunks (including metadata)
    chunks = chunk_earnings_html(html_content, include_metadata=True)

    # Convert chunks to transcript format
    transcript = "\n".join([f"**Index {i}**: {chunk}" for i, chunk in enumerate(chunks)])

    # Extract targets using LLM
    llm_result = await get_targets(transcript)

    # Map LLM results to base_target format
    targets_data = map_llm_targets_to_chunks(llm_result, chunks)

    # Calculate statistics
    stats = calculate_llm_statistics(chunks, targets_data)

    # Complete targets list (deduplicated)
    all_targets = list(targets_data["targets_by_section"]["all"])

    # Same output format as base_target
    return {
        "chunk_entities": targets_data["chunk_entities"],
        "total_entities": {
            "targets": all_targets  # LLM extracts only targets
        },
        "entity_counts": {
            "targets": len(all_targets)
        },
        "targets_by_section": {k: list(v) for k, v in targets_data["targets_by_section"].items()},
        "total_targets": stats["total_targets"],
        "presentation_targets": stats["presentation_targets"],
        "qa_targets": stats["qa_targets"],
        "speaker_stats": stats["speaker_stats"],
        "section_stats": stats["section_stats"],
        "total_chunks": stats["total_chunks"]
    }


if __name__ == "__main__":
    # Test execution
    try:
        with open("data/earnings/US67066G1040=Q2_2025_NVIDIA_Corporation=2024-08-28.html", "r") as f:
            html_content = f.read()

        print("=== Testing LLM-based Earnings Call Analysis ===")
        results = asyncio.run(analyze_earnings_call(html_content))

        print(f"Total chunks: {results['total_chunks']}")
        print(f"Total unique targets: {results['total_targets']}")
        print(f"Presentation targets: {results['presentation_targets']}")
        print(f"Q&A targets: {results['qa_targets']}")

        print("\n=== Speaker Statistics ===")
        for speaker_type, count in results['speaker_stats'].items():
            print(f"{speaker_type}: {count} utterances")

        print("\n=== Section Statistics ===")
        for section, count in results['section_stats'].items():
            print(f"{section}: {count} chunks")

        print("\n=== Sample Targets (first 20) ===")
        for i, target in enumerate(results['targets_by_section']['all'][:20]):
            print(f"{i + 1:2d}. {target}")

    except FileNotFoundError:
        print("Test file not found. Please check the path.")
    except Exception as e:
        print(f"Error during analysis: {e}")
