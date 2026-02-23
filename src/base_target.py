from typing import List, Dict, Set

import spacy

from src.utils import SpeakerType


def load_spacy_model():
    """Load spaCy model"""
    try:
        return spacy.load("en_core_web_sm")
    except OSError:
        print("spaCy English model not found. Please install it with:")
        print("python -m spacy download en_core_web_sm")
        raise


def chunk_earnings_html(text: str) -> List[Dict[str, str]]:
    """
    Parse earnings call HTML and separate into speaker-specific chunks
    Use utils.py function with metadata included
    """
    from src.utils import chunk_earnings_html as utils_chunk_earnings_html
    return utils_chunk_earnings_html(text, include_metadata=True)


def extract_targets_from_text(text: str, nlp) -> Set[str]:
    """
    Extract targets from text using spaCy
    Extract based on Products, Money, Percent entities according to paper methodology
    """
    doc = nlp(text)
    targets = set()

    # 1. Extract Products entities directly as targets
    for ent in doc.ents:
        if ent.label_ == "PRODUCT":
            target = ent.text.strip().lower()
            if target and len(target) > 1:
                targets.add(target)

    # 2. Extract related noun phrases from Money and Percent entities
    for ent in doc.ents:
        if ent.label_ in ["MONEY", "PERCENT"]:
            # Find tokens of this entity
            ent_tokens = [token for token in doc if ent.start <= token.i < ent.end]

            for token in ent_tokens:
                # Find noun phrases grammatically connected to Money/Percent entity
                target_phrases = find_related_noun_phrases(token, doc)
                for phrase in target_phrases:
                    clean_phrase = phrase.strip().lower()
                    if clean_phrase and len(clean_phrase) > 2:
                        targets.add(clean_phrase)

    return targets


def find_related_noun_phrases(token, doc) -> List[str]:
    """
    Function to find noun phrases grammatically related to given token (Money/Percent)
    Implementation of Part-of-Speech analysis methodology from the paper
    """
    related_phrases = []

    # Track grammatical relationships through token's head
    head = token.head

    # 1. When Money entity is the object or subject of a verb
    if head.pos_ in ["VERB", "AUX"]:
        # Find subject
        for child in head.children:
            if child.dep_ == "nsubj" and child.pos_ in ["NOUN", "PROPN"]:
                noun_phrase = extract_noun_phrase(child)
                if noun_phrase:
                    related_phrases.append(noun_phrase)

        # Find object
        for child in head.children:
            if child.dep_ == "dobj" and child.pos_ in ["NOUN", "PROPN"]:
                noun_phrase = extract_noun_phrase(child)
                if noun_phrase:
                    related_phrases.append(noun_phrase)

    # 2. When Percent entity modifies a noun
    elif head.pos_ in ["NOUN", "PROPN"]:
        noun_phrase = extract_noun_phrase(head)
        if noun_phrase:
            related_phrases.append(noun_phrase)

    # 3. Find related nouns in prepositional phrases
    for ancestor in token.ancestors:
        if ancestor.dep_ == "prep":
            for child in ancestor.children:
                if child.dep_ == "pobj" and child.pos_ in ["NOUN", "PROPN"]:
                    noun_phrase = extract_noun_phrase(child)
                    if noun_phrase:
                        related_phrases.append(noun_phrase)

    return related_phrases


def extract_noun_phrase(token) -> str:
    """Extract noun phrase centered on token"""
    # Construct noun phrase from token's subtree
    phrase_tokens = []

    # Collect left modifiers of the token
    left_modifiers = [child for child in token.children
                      if child.i < token.i and child.dep_ in ["amod", "compound", "det"]]
    left_modifiers.sort(key=lambda x: x.i)
    phrase_tokens.extend(left_modifiers)

    # Add central token
    phrase_tokens.append(token)

    # Collect right modifiers of the token
    right_modifiers = [child for child in token.children
                       if child.i > token.i and child.dep_ in ["amod", "compound"]]
    right_modifiers.sort(key=lambda x: x.i)
    phrase_tokens.extend(right_modifiers)

    if phrase_tokens:
        return " ".join([t.text for t in phrase_tokens])
    return ""


def extract_targets_by_section(chunks: List[Dict[str, str]], nlp) -> Dict[str, Set[str]]:
    """
    Extract targets by conference call section
    """
    targets_by_section = {
        "presentation": set(),
        "qa": set(),
        "all": set()
    }

    for chunk in chunks:
        # Extract targets only from Executive statements (reflecting paper methodology)
        if chunk['speaker_type'] == SpeakerType.EXECUTIVE.value:
            targets = extract_targets_from_text(chunk['content'], nlp)

            section = chunk['section']
            if section in targets_by_section:
                targets_by_section[section].update(targets)

            targets_by_section["all"].update(targets)

    return targets_by_section


def extract_entities_by_type_from_text(text: str, nlp) -> Dict[str, List[str]]:
    """
    Extract entities by type from text with detailed categorization
    """
    doc = nlp(text)
    entities = {
        "product": [],
        "money": [],
        "percent": []
    }

    for ent in doc.ents:
        if ent.label_ == "PRODUCT":
            entities["product"].append(ent.text.strip())
        elif ent.label_ == "MONEY":
            entities["money"].append(ent.text.strip())
        elif ent.label_ == "PERCENT":
            entities["percent"].append(ent.text.strip())

    return entities


def extract_entities_from_chunks(chunks: List[Dict[str, str]], nlp) -> List[Dict[str, any]]:
    """
    Extract product, money, percent entities for each chunk
    """
    chunk_entities = []

    for i, chunk in enumerate(chunks):
        entities = extract_entities_by_type_from_text(chunk['content'], nlp)

        chunk_data = {
            "index": i,
            "speaker": chunk['speaker'],
            "speaker_type": chunk['speaker_type'],
            "section": chunk['section'],
            "entities": entities,
            "content_preview": chunk['content'][:100] + "..." if len(chunk['content']) > 100 else chunk['content']
        }

        chunk_entities.append(chunk_data)

    return chunk_entities


def extract_related_phrases_by_entity_type(text: str, nlp) -> Dict[str, List[str]]:
    """
    Extract related noun phrases of Money and Percent entities by type
    """
    doc = nlp(text)
    related_phrases = {
        "money_related": [],
        "percent_related": []
    }

    for ent in doc.ents:
        if ent.label_ == "MONEY":
            # Find related noun phrases for Money entity
            ent_tokens = [token for token in doc if ent.start <= token.i < ent.end]
            for token in ent_tokens:
                phrases = find_related_noun_phrases(token, doc)
                for phrase in phrases:
                    clean_phrase = phrase.strip()
                    if clean_phrase and len(clean_phrase) > 2:
                        related_phrases["money_related"].append(clean_phrase)

        elif ent.label_ == "PERCENT":
            # Find related noun phrases for Percent entity
            ent_tokens = [token for token in doc if ent.start <= token.i < ent.end]
            for token in ent_tokens:
                phrases = find_related_noun_phrases(token, doc)
                for phrase in phrases:
                    clean_phrase = phrase.strip()
                    if clean_phrase and len(clean_phrase) > 2:
                        related_phrases["percent_related"].append(clean_phrase)

    return related_phrases


def analyze_earnings_call_detailed(html_content: str) -> Dict[str, any]:
    """
    Detailed analysis of earnings call HTML with chunk-by-chunk entity extraction and targets extraction
    """
    # Load spaCy model
    nlp = load_spacy_model()

    # Parse HTML and separate into chunks
    chunks = chunk_earnings_html(html_content)

    # Extract entities by chunk
    chunk_entities = extract_entities_from_chunks(chunks, nlp)

    # Extract targets by section (existing method)
    targets_by_section = extract_targets_by_section(chunks, nlp)

    # Overall statistics
    total_entities = {
        "product": [],
        "money": [],
        "percent": []
    }

    for chunk_data in chunk_entities:
        for entity_type, entities in chunk_data["entities"].items():
            total_entities[entity_type].extend(entities)

    # Statistics by speaker type
    speaker_stats = {}
    for chunk in chunks:
        speaker_type = chunk['speaker_type']
        if speaker_type not in speaker_stats:
            speaker_stats[speaker_type] = 0
        speaker_stats[speaker_type] += 1

    # Statistics by section
    section_stats = {}
    for chunk in chunks:
        section = chunk['section']
        if section not in section_stats:
            section_stats[section] = 0
        section_stats[section] += 1

    return {
        "chunk_entities": chunk_entities,
        "total_entities": {
            "product": list(set(total_entities["product"])),  # Remove duplicates
            "money": list(set(total_entities["money"])),
            "percent": list(set(total_entities["percent"]))
        },
        "entity_counts": {
            "product": len(set(total_entities["product"])),
            "money": len(set(total_entities["money"])),
            "percent": len(set(total_entities["percent"]))
        },
        "targets_by_section": {k: list(v) for k, v in targets_by_section.items()},
        "total_targets": len(targets_by_section["all"]),
        "presentation_targets": len(targets_by_section["presentation"]),
        "qa_targets": len(targets_by_section["qa"]),
        "speaker_stats": speaker_stats,
        "section_stats": section_stats,
        "total_chunks": len(chunks)
    }


def analyze_earnings_call(html_content: str) -> Dict[str, any]:
    """
    Analyze earnings call HTML to extract targets and return analysis results
    (Maintained for backward compatibility)
    """
    # Load spaCy model
    nlp = load_spacy_model()

    # Parse HTML and separate into chunks
    chunks = chunk_earnings_html(html_content)

    # Extract targets by section
    targets_by_section = extract_targets_by_section(chunks, nlp)

    # Statistics by speaker type
    speaker_stats = {}
    for chunk in chunks:
        speaker_type = chunk['speaker_type']
        if speaker_type not in speaker_stats:
            speaker_stats[speaker_type] = 0
        speaker_stats[speaker_type] += 1

    # Statistics by section
    section_stats = {}
    for chunk in chunks:
        section = chunk['section']
        if section not in section_stats:
            section_stats[section] = 0
        section_stats[section] += 1

    return {
        "targets_by_section": {k: list(v) for k, v in targets_by_section.items()},
        "total_targets": len(targets_by_section["all"]),
        "presentation_targets": len(targets_by_section["presentation"]),
        "qa_targets": len(targets_by_section["qa"]),
        "speaker_stats": speaker_stats,
        "section_stats": section_stats,
        "total_chunks": len(chunks)
    }


if __name__ == "__main__":
    # Run test
    try:
        with open("data/earnings/AN8068571086=Q1_2008_Schlumberger_Limited=2008-04-18.html", "rb") as f:
            html_content = f.read()

        results = analyze_earnings_call(html_content.decode('utf-8'))

        print("=== Earnings Call Target Analysis Results ===")
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
