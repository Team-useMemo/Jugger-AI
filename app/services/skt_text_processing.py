import re
import numpy as np
from typing import List, Dict
from sklearn.metrics.pairwise import cosine_similarity
from fastapi import HTTPException

from sentence_transformers import SentenceTransformer
from keybert import KeyBERT
from kiwipiepy import Kiwi

from app.services.mongo_service import get_user_categories,user_exists
from app.utils.schedule_utils import extract_schedules
from app.utils.url_utils import validate_urls


embedding_model = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")
keyword_model = KeyBERT(model=embedding_model)

kiwi = Kiwi()

url_pattern = re.compile(r'https?://[a-zA-Z0-9./?=&_%:-]+')

async def clean_text_and_extract_urls(sentence: str):
    urls = url_pattern.findall(sentence)
    if urls:
        valid, invalid = await validate_urls(urls)
    else:
        valid, invalid = [], []

    clean_text = url_pattern.sub('', sentence).strip()
    return clean_text, valid, invalid

def get_sentence_embedding(text: str):
    return embedding_model.encode(text)

def extract_keywords(text: str, top_k: int = 3) -> List[str]:
    tokens = kiwi.tokenize(text)

    nouns = [t.form for t in tokens if t.tag.startswith('NN')]

    if not nouns:
        return ["기타"]

    joined_text = " ".join(nouns)

    keywords = keyword_model.extract_keywords(
        joined_text,
        top_n=top_k,
        keyphrase_ngram_range=(1, 2),
        stop_words=None
    )

    return [kw[0] for kw in keywords] if keywords else ["기타"]


async def classify_paragraph_with_user(
    user_uuid: str,
    paragraph: str,
    threshold: float = 0.5
) -> Dict:

    if not await user_exists(user_uuid):
        raise HTTPException(
            status_code=404,
            detail=f"User with uuid '{user_uuid}' not found."
        )

    sentences = paragraph.split("\n")

    para_emb = get_sentence_embedding(paragraph)

    user_categories = await get_user_categories(user_uuid)
    if not user_categories:
        keywords = extract_keywords(paragraph, top_k=3)
        return {
            "category": "no",
            "recommend_category": keywords,
            "sentences": [
                {
                    "text": s.strip() or "빈 문장",
                    "urls": url_pattern.findall(s) or None
                } for s in sentences
            ]
        }

    category_vectors = {
        cat["name"]: get_sentence_embedding(cat["name"])
        for cat in user_categories if "name" in cat
    }

    best_category, best_similarity = None, 0.0
    for name, vec in category_vectors.items():
        sim = cosine_similarity([para_emb], [vec])[0][0]
        if sim > best_similarity:
            best_similarity = sim
            best_category = name

    if best_similarity >= threshold:
        return_category = best_category
        recommended = [best_category]
    else:
        return_category = "no"
        recommended = extract_keywords(paragraph, top_k=3)

    processed_sentences = []
    for s in sentences:
        text, urls, invalid_urls = await clean_text_and_extract_urls(s)
        schedules = extract_schedules(s)
        processed_sentences.append({
            "text": text or "URL 포함 문장",
            "urls": urls or None,
            "invalid_urls": invalid_urls or None,
            "schedules": schedules or None,
        })

    return {
        "category": return_category,
        "recommend_category": recommended,
        "sentences": processed_sentences,
    }