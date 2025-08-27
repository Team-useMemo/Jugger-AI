# -*- coding: utf-8 -*-
import os, json, time, asyncio, re
from typing import List, Optional, Dict

# dotenv 지연 로드
try:
    from dotenv import load_dotenv, find_dotenv
    p = find_dotenv(".env", usecwd=True)
    if p: load_dotenv(p, override=False)
except Exception:
    pass

import google.generativeai as genai
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

from app.utils.schedule_utils import extract_schedules
from app.utils.url_utils import validate_urls

DEFAULT_MODEL_NAME = "gemini-1.5-flash"
URL_PATTERN = re.compile(r'https?://[a-zA-Z0-9./?=&_%:-]+')
TIME_WORDS = {
    "오늘","내일","모레","이번주","다음주","이번","다음",
    "오전","오후","아침","점심","저녁","밤","새벽",
    "시","분","요일","주말","평일"
}

# 로컬에서 category 계산용 (동일 로직 재사용)
_embedding = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")

def _ensure_gemini() -> str:
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise RuntimeError("GEMINI_API_KEY 가 비어있습니다. --env-file .env 또는 환경변수 설정 필요")
    genai.configure(api_key=api_key)
    return os.getenv("GEMINI_MODEL", DEFAULT_MODEL_NAME)

def _normalize(s: str) -> str:
    return " ".join((s or "").split())

def _sanitize_labels(
    labels: List[str],
    user_categories: Optional[List[str]],
    k: int = 5,
    extras: Optional[List[str]] = None,
) -> List[str]:
    uc_set = { _normalize(s) for s in (user_categories or []) if isinstance(s, str) and s.strip() }
    out, seen = [], set()
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

        parts = s.split()
        # 동일 단어 반복 ("내일 내일")
        if len(parts) >= 2 and len(set(parts)) == 1:
            return False
        # 시간 단어 포함 복합 조합은 UC에 정확히 있을 때만 허용
        if len(parts) >= 2 and any(p in TIME_WORDS for p in parts):
            if s not in uc_set:
                return False

        out.append(s)
        seen.add(low)
        return True

    # 1차: 모델이 준 라벨
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

    # 3차: '기타' 1회만 보충
    if len(out) < k and not has_etc:
        try_add("기타")

    return out[:k]

def _dedup_top_k(items: List[str], k: int = 5) -> List[str]:
    # (기본 dedup) — sanitize 전에 1차 정리
    seen, out = set(), []
    for it in items:
        s = _normalize(it)
        if not s:
            continue
        low = s.lower()
        if low in seen:
            continue
        seen.add(low); out.append(s)
        if len(out) >= k:
            break
    return out[:k]

async def recommend_categories_with_gemini(
    paragraph: str,
    user_categories: Optional[List[str]],
    k: int = 5,
    timeout_sec: int = 12,
    max_retries: int = 2,
) -> List[str]:

    model_name = _ensure_gemini()
    uc = user_categories or []

    system_instruction = (
        "너는 한국어 문단을 1~3단어의 간결한 '카테고리 라벨'로 분류하는 시스템이다. "
        "출력은 JSON 한 줄만 반환한다."
    )

    prompt = f"""
아래 문단에 대한 추천 카테고리를 최대 {k}개 반환하세요.

[목표]
- 1~3단어의 간결한 '카테고리 라벨'을 중요도 순으로 추천합니다.
- 가능한 경우에는 제공된 user_categories에서 우선 선택합니다.
- 그러나 user_categories에 적절한 라벨이 전혀 없거나 의미적으로 거리가 멀다면,
  목록 밖의 새로운 일반 라벨을 만들어도 됩니다.

[하드 금지 규칙]
- 시간 단어(예: 오늘/내일/오전/오후/시/분 등)와 다른 단어의 단순 결합
  (예: "내일 학교")은 user_categories에 그 **정확한 문구**가 있을 때만 허용합니다.
- 동일 단어 반복(예: "내일 내일")은 금지합니다.

[우선 규칙]
1) user_categories에 문단과 가까운 라벨이 있으면 그 라벨을 우선 사용(화이트리스트 우선).
2) 문장에 날짜/시간/의무 표현이 있으면 '약속' 또는 '일정' 계열 라벨을 반드시 고려합니다.
   - user_categories 안에 동등 라벨이 있으면 그걸 사용,
   - 없으면 목록 밖 라벨로 새로 제안할 수 있습니다.
3) 문장에 정확히 등장하는 단어가 user_categories에 있으면 그 라벨을 우선 포함합니다.
4) 동의어/유사어는 user_categories의 정확한 표기로 매핑합니다.

[출력 형식]
오직 JSON 한 줄만:
{{"recommend_category": ["라벨1","라벨2", ...]}}

[user_categories]
{uc}

[문단]
\"\"\"{paragraph}\"\"\"
"""

    model = genai.GenerativeModel(model_name=model_name, system_instruction=system_instruction)

    def _call():
        return model.generate_content(
            [prompt],
            generation_config=genai.GenerationConfig(
                temperature=0.1,
                max_output_tokens=256,
                response_mime_type="application/json",
            ),
            safety_settings=None,
        )

    start = time.time()
    for attempt in range(max_retries + 1):
        try:
            resp = await asyncio.wait_for(asyncio.to_thread(_call), timeout=timeout_sec)
            text = (resp.text or "").strip()
            try:
                data = json.loads(text)
                cats = data.get("recommend_category", [])
            except Exception:
                lb, rb = text.find("["), text.rfind("]")
                cats = json.loads(text[lb:rb+1]) if (lb!=-1 and rb!=-1 and rb>lb) else []
            # 1차 dedup + 2차 sanitize
            time_extras = [w for w in TIME_WORDS if w in paragraph]
            extras = (user_categories or []) + time_extras
            return _sanitize_labels(
                labels=_dedup_top_k([str(c) for c in cats], k=max(k, 5)),
                user_categories=user_categories,
                k=k,
                extras=extras
            )

        except Exception:
            if attempt >= max_retries or (time.time()-start) > (timeout_sec*(max_retries+1)):
                raise
            await asyncio.sleep(0.4*(attempt+1))

async def classify_paragraph_gemini(
    paragraph: str,
    user_categories: Optional[List[str]] = None,
    threshold: float = 0.5,
    k: int = 5,
) -> Dict:
    # 1) 문장/텍스트 정리
    sentences = paragraph.split("\n")
    clean_texts = [URL_PATTERN.sub("", s).strip() or "URL 포함 문장" for s in sentences]

    # 2) URL 배치 검증
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

    urls_valid, urls_invalid = [], []
    for urls in per_sentence_urls:
        urls_valid.append([u for u in urls if valid_map.get(u, False)] or None)
        urls_invalid.append([u for u in urls if invalid_map.get(u, False)] or None)

    # 3) 스케줄 추출
    schedules = [extract_schedules(s) or None for s in sentences]

    # 4) category 계산(로컬 벡터 매칭)
    category = "no"
    best_cat = None
    para_emb = await asyncio.to_thread(_embedding.encode, paragraph)

    if user_categories:
        cats = [c.strip() for c in user_categories if isinstance(c, str) and c.strip()]
        if cats:
            cat_embs = await asyncio.to_thread(_embedding.encode, cats)
            sims = cosine_similarity([para_emb], np.asarray(cat_embs))[0]
            idx = int(np.argmax(sims))
            best_sim = float(sims[idx])
            bc = cats[idx]
            if best_sim >= threshold:
                category = bc
                best_cat = bc

    # 5) 추천: Gemini → seed에 best_cat 추가 → 최종 정제(중복 제거, '기타' 최대 1회)
    gemini_labels = await recommend_categories_with_gemini(paragraph, user_categories, k=k)
    seed = ([best_cat] if best_cat else []) + gemini_labels
    recs = _sanitize_labels(
        labels=seed,
        user_categories=user_categories,
        k=k,
        extras=(user_categories or []) + [w for w in TIME_WORDS if w in paragraph],
    )

    # 6) 문장별 결과
    processed = [
        {
            "text": clean_texts[i],
            "urls": urls_valid[i],
            "invalid_urls": urls_invalid[i],
            "schedules": schedules[i],
        }
        for i in range(len(sentences))
    ]

    return {"category": category, "recommend_category": recs, "sentences": processed}
