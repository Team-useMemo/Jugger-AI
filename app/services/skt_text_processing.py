import re
from sklearn.metrics.pairwise import cosine_similarity
from typing import List, Dict
import numpy as np
from models.skt_kobert import get_sentence_embedding

# ✅ 기존 카테고리 리스트
CATEGORY_LABELS = {
    "여행": ["도쿄 여행", "유럽 배낭 여행", "국내 캠핑", "비행기 예약", "숙소 추천"],
    "교통": ["지하철 환승", "고속열차 이용", "비행기 탑승", "대중교통 이용", "렌터카 예약"],
    "쇼핑": ["면세점 할인", "백화점 세일", "전자제품 구매", "패션 브랜드 쇼핑", "기념품 구매"],
    "음식": ["일본 라멘 맛집", "프랑스 빵 추천", "한식당 방문", "카페 탐방", "길거리 음식"],
    "기타": ["문화 체험", "미술관 방문", "박물관 견학", "자연 탐방", "테마파크 방문"]
}

# ✅ URL 정규 표현식
url_pattern = re.compile(r'https?://[a-zA-Z0-9./?=&_%:-]+')

# ✅ 문장에서 URL을 추출하고 문장 내에서 분리하는 함수
def extract_urls_from_sentences(sentence: str):
    urls = url_pattern.findall(sentence)  # 정확한 URL만 추출
    text_without_urls = url_pattern.sub(' ', sentence).strip()  # URL을 공백으로 대체
    text_without_urls = re.sub(r'\s+', ' ', text_without_urls)  # 연속된 공백 제거
    return text_without_urls, urls

# ✅ 카테고리 벡터 저장 (초기 설정)
category_embeddings = {
    category: np.mean([get_sentence_embedding(example) for example in examples], axis=0)
    for category, examples in CATEGORY_LABELS.items()
}

# ✅ 새로운 카테고리 추천 함수
def generate_new_category(text: str):
    words = text.split()
    for word in words:
        if len(word) > 1:  # 짧은 단어 제외
            return word  # 첫 번째 의미 있는 단어를 카테고리로 사용
    return "기타"

# ✅ 문단 단위 카테고리 설정 및 문장별 세부 분류 (기존 카테고리 검증 추가)
def classify_paragraph(paragraph: str, threshold: float = 0.7):
    global category_embeddings
    sentences = paragraph.split("\n")  # 문장 단위로 분리
    processed_sentences = []
    paragraph_embedding = get_sentence_embedding(paragraph)

    # ✅ 문단 전체 카테고리 설정
    best_category = None
    best_similarity = 0

    for category, category_vector in category_embeddings.items():
        similarity = cosine_similarity([paragraph_embedding], [category_vector])[0][0]
        if similarity > best_similarity:
            best_similarity = similarity
            best_category = category

    # ✅ 기존 카테고리에 없으면 "category": "no", 추천 카테고리 제공
    if best_similarity < threshold:
        recommend_category = generate_new_category(paragraph)
        if recommend_category not in category_embeddings:
            category_embeddings[recommend_category] = paragraph_embedding
        return_category = "no"
    else:
        recommend_category = best_category
        return_category = best_category

    # ✅ 문장별 하위 분류
    for sentence in sentences:
        text_part, urls = extract_urls_from_sentences(sentence)
        sentence_embedding = get_sentence_embedding(text_part) if text_part else None

        # URL 포함 여부 확인
        if urls:
            sub_category = "관련 링크"
        else:
            sub_category = recommend_category  # 추천된 카테고리 사용

        processed_sentences.append({
            "text": text_part if text_part else "URL 포함 문장",
            "sub_category": sub_category,
            "urls": urls if urls else None
        })

    return {
        "category": return_category,  # 기존 카테고리에 있으면 해당 카테고리, 없으면 "no"
        "recommend_category": recommend_category,  # 기존 카테고리가 없을 때 추천 카테고리 제공
        "sentences": processed_sentences
    }
