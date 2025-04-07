import re
import numpy as np
from typing import List, Dict
from sklearn.metrics.pairwise import cosine_similarity
from fastapi import HTTPException

from sentence_transformers import SentenceTransformer
from keybert import KeyBERT
from kiwipiepy import Kiwi

from app.services.mongo_service import get_user_categories,user_exists


embedding_model = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")
keyword_model = KeyBERT(model=embedding_model)

kiwi = Kiwi()

url_pattern = re.compile(r'https?://[a-zA-Z0-9./?=&_%:-]+')

def clean_text_and_extract_urls(sentence: str):
    urls = url_pattern.findall(sentence)
    clean_text = url_pattern.sub('', sentence).strip()
    return clean_text, urls

def get_sentence_embedding(text: str):
    return embedding_model.encode(text)

def extract_keywords(text: str, top_k: int = 3) -> List[str]:
    # 1) 먼저 Kiwi로 명사 추출
    tokens = kiwi.tokenize(text)

    nouns = [t.form for t in tokens if t.tag.startswith('NN')]

    if not nouns:
        return ["기타"]

    # 2) 추출된 명사들을 띄어쓰기로 합치기 → KeyBERT 입력
    joined_text = " ".join(nouns)

    # 3) KeyBERT 호출: ngram 범위 (1,2), 불용어 'korean' 적용
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
    threshold: float = 0.6
) -> Dict:

    # ✅ 사용자 존재 여부 먼저 확인
    if not await user_exists(user_uuid):
        raise HTTPException(
            status_code=404,
            detail=f"User with uuid '{user_uuid}' not found."
        )

    sentences = paragraph.split("\n")
    para_emb = get_sentence_embedding(paragraph)

    # 1. 사용자 카테고리 불러오기
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

    # 2. 카테고리 임베딩
    category_vectors = {
        cat["name"]: get_sentence_embedding(cat["name"])
        for cat in user_categories if "name" in cat
    }

    # 3. 문단 임베딩 vs 카테고리 벡터
    best_category, best_similarity = None, 0.0
    for name, vec in category_vectors.items():
        sim = cosine_similarity([para_emb], [vec])[0][0]
        if sim > best_similarity:
            best_similarity = sim
            best_category = name

    # 4. 임계값 이상이면 해당 카테고리, 아니면 새 키워드
    if best_similarity >= threshold:
        return_category = best_category
        recommended = [best_category]
    else:
        return_category = "no"
        recommended = extract_keywords(paragraph, top_k=3)

    # 5. 문장별 처리
    processed_sentences = []
    for s in sentences:
        text, urls = clean_text_and_extract_urls(s)
        sub_cat = "관련 링크" if urls else (recommended[0] if recommended else "기타")
        processed_sentences.append({
            "text": text or "URL 포함 문장",
            "urls": urls or None
        })

    return {
        "category": return_category,
        "recommend_category": recommended,
        "sentences": processed_sentences
    }