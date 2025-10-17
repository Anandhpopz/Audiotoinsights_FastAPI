from typing import Dict, Any, List, Tuple
import re
import math
from collections import Counter

# Lightweight stopword list (small, extend as needed)
STOPWORDS = {
    'the', 'and', 'is', 'in', 'it', 'of', 'to', 'a', 'an', 'for', 'on', 'that',
    'this', 'with', 'as', 'are', 'was', 'were', 'be', 'by', 'at', 'from', 'or',
    'we', 'you', 'i', 'they', 'he', 'she', 'have', 'has', 'had'
}

# Small sentiment lexicon (extend for better accuracy)
POSITIVE = {'good', 'great', 'excellent', 'happy', 'satisfied', 'awesome', 'love', 'pleasant', 'fantastic'}
NEGATIVE = {'bad', 'terrible', 'poor', 'unhappy', 'hate', 'awful', 'complaint', 'complain', 'dissatisfied', 'problem'}

# Intent keywords mapping
INTENT_KEYWORDS = {
    'booking_inquiry': {'book', 'booking', 'reserve', 'reservation', 'availability'},
    'complaint': {'complain', 'complaint', 'not working', 'bad', 'issue', 'problem', 'angry'},
    'feedback': {'feedback', 'suggestion', 'recommend', 'review'},
    'price_check': {'price', 'cost', 'charge', 'rate', 'fee', 'how much', 'pricey'},
    'general_info': {'info', 'information', 'details', 'tell me', 'what is'}
}

# Topic keyword sets for a simple topic model
TOPIC_KEYWORDS = {
    'amenities': {'wifi', 'breakfast', 'pool', 'parking', 'ac', 'air conditioning', 'bathroom', 'laundry'},
    'pricing': {'price', 'cost', 'cheap', 'expensive', 'rate', 'discount'},
    'availability': {'available', 'availability', 'vacancy', 'booked', 'full'},
    'booking': {'book', 'reserve', 'reservation', 'check-in', 'checkin', 'cancellation'},
    'complaint': {'complaint', 'problem', 'issue', 'delay', 'broken', 'dirty'}
}


def _tokenize(text: str) -> List[str]:
    text = text.lower()
    # replace punctuation with spaces
    text = re.sub(r"[^a-z0-9₹$%\. ]", ' ', text)
    tokens = [t for t in text.split() if t and t not in STOPWORDS]
    return tokens


def _sentiment_score(tokens: List[str]) -> float:
    pos = sum(1 for t in tokens if t in POSITIVE)
    neg = sum(1 for t in tokens if t in NEGATIVE)
    if pos == 0 and neg == 0:
        return 0.0
    return (pos - neg) / max(1, pos + neg)


def _classify_intent(text: str, tokens: List[str]) -> Tuple[str, float]:
    text_l = text.lower()
    scores = {}
    for intent, kws in INTENT_KEYWORDS.items():
        # count keyword matches (allow multi-word phrases by checking text)
        score = 0
        for kw in kws:
            if ' ' in kw:
                if kw in text_l:
                    score += 2
            else:
                score += tokens.count(kw)
        scores[intent] = score
    # pick best
    best_intent = max(scores, key=lambda k: scores[k])
    total = sum(scores.values())
    confidence = (scores[best_intent] / total) if total > 0 else 0.0
    if total == 0:
        return ('unknown', 0.0)
    return (best_intent, round(confidence, 2))


def _extract_named_entities(text: str) -> List[Dict[str, str]]:
    entities = []
    # price patterns: $123, ₹123, 123 USD, 123 INR
    for m in re.finditer(r"(₹|\$)\s?\d+[\d,]*(?:\.\d+)?", text):
        entities.append({'type': 'PRICE', 'text': m.group(0)})
    for m in re.finditer(r"\b\d+[\d,]*\s?(inr|usd|eur)\b", text, flags=re.IGNORECASE):
        entities.append({'type': 'PRICE', 'text': m.group(0)})

    # simple NER: capitalized words sequences as potential proper nouns (Hostel names, Cities)
    for m in re.finditer(r"\b([A-Z][a-z0-9]+(?:\s+[A-Z][a-z0-9]+)*)\b", text):
        token = m.group(1)
        # filter out sentences starting words like 'I' or single common words
        if len(token) > 1 and not token.isupper():
            # heuristically categorize
            ent_type = 'ENTITY'
            if 'hostel' in token.lower() or 'inn' in token.lower() or 'hotel' in token.lower():
                ent_type = 'HOSTEL'
            entities.append({'type': ent_type, 'text': token})

    # amenities: look for common amenities words
    for amenity in TOPIC_KEYWORDS['amenities']:
        if re.search(rf"\b{re.escape(amenity)}\b", text, flags=re.IGNORECASE):
            entities.append({'type': 'AMENITY', 'text': amenity})

    # deduplicate while preserving order
    seen = set()
    uniq = []
    for e in entities:
        key = (e['type'], e['text'].lower())
        if key not in seen:
            seen.add(key)
            uniq.append(e)
    return uniq


def _extract_keywords(tokens: List[str], top_n: int = 8) -> List[Tuple[str, int]]:
    counts = Counter(tokens)
    return counts.most_common(top_n)


def _topic_model(tokens: List[str]) -> List[Tuple[str, float]]:
    scores = {}
    token_set = set(tokens)
    for topic, kws in TOPIC_KEYWORDS.items():
        matches = sum(1 for kw in kws if kw in token_set)
        scores[topic] = matches
    # normalize to 0..1
    max_score = max(scores.values()) if scores else 0
    result = []
    for t, s in scores.items():
        score = (s / max_score) if max_score > 0 else 0.0
        result.append((t, round(score, 2)))
    # return topics sorted by score desc
    result.sort(key=lambda x: x[1], reverse=True)
    return result


def nlp_pipeline(text: str) -> Dict[str, Any]:
    """Expanded light-weight NLP pipeline.

    Returns a dict with:
      - word_count, sentence_count, sample
      - sentiment: {label, score}
      - intent: {label, confidence}
      - named_entities: list of {type, text}
      - keywords: list of (keyword, count)
      - topics: list of (topic, score)

    This implementation uses simple heuristics and is intended for local testing
    and prototyping. Replace components with stronger NLP libraries for
    production accuracy (e.g., spaCy, transformers, or cloud NLP APIs).
    """
    if not text or not text.strip():
        return {}

    words = text.split()
    sentences = [s for s in re.split(r'[\.!?]+', text) if s.strip()]
    tokens = _tokenize(text)

    # Sentiment (score in -1..1)
    s_score = _sentiment_score(tokens)
    if s_score > 0.2:
        s_label = 'positive'
    elif s_score < -0.2:
        s_label = 'negative'
    else:
        s_label = 'neutral'

    # Intent
    intent_label, intent_conf = _classify_intent(text, tokens)

    # NER
    entities = _extract_named_entities(text)

    # Keywords
    keywords = _extract_keywords(tokens)

    # Topic modeling (very lightweight)
    topics = _topic_model(tokens)

    return {
        'word_count': len(words),
        'sentence_count': len(sentences),
        'sample': ' '.join(words[:20]),
        'sentiment': {'label': s_label, 'score': round(s_score, 2)},
        'intent': {'label': intent_label, 'confidence': intent_conf},
        'named_entities': entities,
        'keywords': [{'word': k, 'count': c} for k, c in keywords],
        'topics': [{'topic': t, 'score': s} for t, s in topics]
    }

