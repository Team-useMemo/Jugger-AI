from sklearn.metrics.pairwise import cosine_similarity
from typing import List, Dict
import numpy as np
from models.kobert_classifier import get_sentence_embedding

# 기존 카테고리 저장소 (초기 없음)
category_embeddings: Dict[str, List[np.ndarray]] = {}

# 문장 분류 함수 (유사 카테고리 확인 및 생성)
def classify_or_create_category(sentences: List[str], threshold: float = 0.7):
    global category_embeddings
    categorized_sentences = {}

    for sentence in sentences:
        sentence_embedding = get_sentence_embedding(sentence)

        best_category = None
        best_similarity = 0

        # 기존 카테고리와 유사도 비교
        for category, embeddings in category_embeddings.items():
            similarities = cosine_similarity([sentence_embedding], embeddings).mean()
            if similarities > best_similarity:
                best_similarity = similarities
                best_category = category

        # 기존 카테고리 존재 시 추가, 없으면 새 카테고리 생성
        if best_similarity >= threshold and best_category:
            categorized_sentences.setdefault(best_category, []).append(sentence)
            category_embeddings[best_category].append(sentence_embedding)
        else:
            new_category = f"Category_{len(category_embeddings) + 1}"
            category_embeddings[new_category] = [sentence_embedding]
            categorized_sentences.setdefault(new_category, []).append(sentence)

    return categorized_sentences
