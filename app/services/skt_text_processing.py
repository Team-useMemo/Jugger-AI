# -*- coding: utf-8 -*-
import re
import asyncio
from functools import lru_cache
from typing import List, Optional, Dict, Tuple

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

from sentence_transformers import SentenceTransformer
from keybert import KeyBERT
from kiwipiepy import Kiwi

from app.utils.schedule_utils import extract_schedules
from app.utils.url_utils import validate_urls

URL_PATTERN = re.compile(r'https?://[a-zA-Z0-9./?=&_%:-]+')

embedding_model = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")
keyword_model = KeyBERT(model=embedding_model)
kiwi = Kiwi()

# 순수 시점/시간 토큰만 남김 (동사류 제거)
TIME_TRIGGERS = [
    "오늘","내일","모레","이번주","다음주","이번 달","다음 달",
    "오전","오후","아침","점심","저녁","밤","새벽",
    "시","분","요일","주말","평일"
]

# ----------------- 유틸 -----------------
def _normalize(s: str) -> str:
    """공백 정규화 + None 방지"""
    return " ".join((s or "").split())

def _token_nouns(text: str) -> List[str]:
    return [t.form for t in kiwi.tokenize(text) if t.tag.startswith('NN')]

def _extract_time_tokens(text: str) -> List[str]:
    seen, out = set(), []
    for t in TIME_TRIGGERS:
        if t in text and t not in seen:
            seen.add(t)
            out.append(t)
    return out

def _dedup_keep_order(items: List[str]) -> List[str]:
    seen, out = set(), []
    for it in items:
        s = _normalize(it)
        low = s.lower()
        if not s or low in seen:
            continue
        seen.add(low)
        out.append(s)
    return out

# ✅ 품사 필터: 동사/형용사/어미/파생접사 포함 라벨 제거
_PRED_PREFIX = ("VV", "VA", "VX", "VCP", "VCN")   # 동사/형용사/보조/계사
_ENDING_PREFIX = ("EP", "EF", "EC", "ETN", "ETM") # 어미/전성어미
_DERIV_PREFIX  = ("XSV", "XSA")                   # 파생 접사(용언화)

def _contains_predicate_or_ending(label: str) -> bool:
    for tok in kiwi.tokenize(label):
        tag = tok.tag
        if tag.startswith(_PRED_PREFIX) or tag.startswith(_ENDING_PREFIX) or tag.startswith(_DERIV_PREFIX):
            return True
    return False

def _sanitize_labels(
    labels: List[str],
    user_categories: Optional[List[str]],
    k: int = 5,
    extras: Optional[List[str]] = None,
) -> List[str]:
    """
    - 대소문자/공백 무시 중복 제거
    - 동사/형용사/어미 포함 라벨 제거 (예: '가야해', '해야 해')
    - 동일 단어 반복 제거 ('내일 내일', '학교 학교')
    - '시간단어 + 다른단어' 복합 라벨은 user_categories에 '정확히 그 문구'가 있을 때만 허용
    - '기타'는 최대 1회만 허용
    - 부족하면 extras로 보충(동일 규칙 적용), 그래도 부족하면 '기타' 1회 추가
    """
    uc_set = { _normalize(s) for s in (user_categories or []) if isinstance(s, str) and s.strip() }
    out: List[str] = []
    seen: set = set()
    has_etc = False

    def try_add(raw: str) -> bool:
        nonlocal has_etc
        s = _normalize(raw)
        if not s:
            return False

        low = s.lower()
        if low in seen:
            return False

        if s == "기타":
            if has_etc:
                return False
            has_etc = True

        # 품사 필터
        if _contains_predicate_or_ending(s):
            return False

        parts = s.split()
        # 동일 단어 반복
        if len(parts) >= 2 and len(set(parts)) == 1:
            return False

        # 시간단어 포함 복합 라벨은 화이트리스트에 없으면 제거
        if len(parts) >= 2 and any(p in TIME_TRIGGERS for p in parts):
            if s not in uc_set:
                return False

        out.append(s)
        seen.add(low)
        return True

    # 1차: 원본 후보
    for lab in labels:
        if len(out) >= k:
            break
        try_add(lab)

    # 2차: 보충 후보(extras)
    if len(out) < k and extras:
        for e in extras:
            if len(out) >= k:
                break
            try_add(e)

    # 3차: '기타' 1회 보충
    if len(out) < k and not has_etc:
        try_add("기타")

    return out[:k]

async def batch_validate_urls(sentences: List[str]) -> Tuple[List[Optional[List[str]]], List[Optional[List[str]]]]:
    per_sentence_urls, all_urls = [], []
    for s in sentences:
        urls = URL_PATTERN.findall(s)
        per_sentence_urls.append(urls)
        all_urls.extend(urls)
    uniq = sorted(set(all_urls))
    valid_map, invalid_map = {}, {}
    if uniq:
        valid, invalid = await validate_urls(uniq)
        vs, iset = set(valid or []), set(invalid or [])
        for u in uniq:
            valid_map[u] = u in vs
            invalid_map[u] = u in iset
    per_sentence_valid, per_sentence_invalid = [], []
    for urls in per_sentence_urls:
        v  = [u for u in urls if valid_map.get(u, False)]
        iv = [u for u in urls if invalid_map.get(u, False)]
        per_sentence_valid.append(v or None)
        per_sentence_invalid.append(iv or None)
    return per_sentence_valid, per_sentence_invalid

def _build_keywords_local(text: str, top_k: int = 12) -> List[str]:
    nouns = _token_nouns(text)
    if not nouns:
        return ["기타"]
    raw = keyword_model.extract_keywords(
        " ".join(nouns),
        top_n=top_k,
        keyphrase_ngram_range=(1, 2)
    )
    return _dedup_keep_order([kw for kw, _ in raw])

@lru_cache(maxsize=256)
def _encode_categories_cached(cat_tuple: tuple) -> np.ndarray:
    return embedding_model.encode(list(cat_tuple))

# --------------- 메인(로컬) ---------------
async def classify_paragraph(
    paragraph: str,
    user_categories: Optional[List[str]] = None,
    threshold: float = 0.5
) -> Dict:

    sentences = paragraph.split("\n")
    clean_texts = [URL_PATTERN.sub('', s).strip() or "URL 포함 문장" for s in sentences]
    urls_valid, urls_invalid = await batch_validate_urls(sentences)
    schedules = [extract_schedules(s) or None for s in sentences]

    # 문단 임베딩
    para_emb = await asyncio.to_thread(embedding_model.encode, paragraph)

    # category 결정(로컬 매칭)
    category, best_cat = "no", None
    if user_categories:
        cats = [c.strip() for c in user_categories if isinstance(c, str) and c.strip()]
        if cats:
            cat_tuple = tuple(cats)
            try:
                cat_embs = _encode_categories_cached(cat_tuple)
            except Exception:
                cat_embs = await asyncio.to_thread(embedding_model.encode, cats)
            sims = cosine_similarity([para_emb], np.asarray(cat_embs))[0]
            idx = int(np.argmax(sims))
            best_sim = float(sims[idx])
            bc = cats[idx]
            if best_sim >= threshold:
                category, best_cat = bc, bc

    # 추천(로컬만; 중복/불필요 제거 + 보충)
    base_keywords = _build_keywords_local(paragraph, top_k=12)   # 명사 기반 후보 넉넉히
    time_tokens  = _extract_time_tokens(paragraph)               # 단독 시간어만
    prelim = ([best_cat] if best_cat else []) + base_keywords + time_tokens

    # 보충 후보: user_categories + 시간어 + 주요 명사
    noun_extras = _token_nouns(paragraph)[:8]
    extras = (user_categories or []) + time_tokens + noun_extras

    recs = _sanitize_labels(prelim, user_categories, k=5, extras=extras)

    processed = [{
        "text": clean_texts[i],
        "urls": urls_valid[i],
        "invalid_urls": urls_invalid[i],
        "schedules": schedules[i],
    } for i in range(len(sentences))]

    return {
        "category": category,
        "recommend_category": recs,  # 중복 없음, '기타' 최대 1회
        "sentences": processed,
    }
