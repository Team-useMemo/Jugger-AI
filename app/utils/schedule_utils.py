# -*- coding: utf-8 -*-
import re, pytz
from datetime import datetime, timedelta
from dateparser.search import search_dates

KST = pytz.timezone("Asia/Seoul")

# ① 한국어 시간대 → AM/PM 매핑
_KR_MERIDIEM = {
    "오전": "AM", "아침": "AM", "새벽": "AM",
    "오후": "PM", "점심": "PM", "낮": "PM", "저녁": "PM", "밤": "PM",
}

# ② (시간대) 10시[ 30분][에]  /  10:30 패턴
_TIME_PAT = re.compile(
    r'(?P<mer>(오전|오후|아침|점심|저녁|밤|새벽))?\s*'
    r'(?P<hour>\d{1,2})\s*시(?:\s*(?P<minute>\d{1,2})\s*분?)?\s*에?'
    r'|(?P<hour2>\d{1,2})\s*:\s*(?P<minute2>\d{2})'
)

# ③ 오늘/내일/모레/글피 매핑
_REL_DAY = {
    "오늘": 0, "내일": 1, "모레": 2, "글피": 3,
}

def _kr_time_to_en(match: re.Match) -> str:
    """한국어 시간 표현 → dateparser가 이해할 영어(10:30 AM)로 변환"""
    mer = match.group('mer')
    hour = match.group('hour') or match.group('hour2')
    minute = match.group('minute') or match.group('minute2') or '00'
    meridiem = _KR_MERIDIEM.get(mer, '').strip()  # mer 없으면 ''
    return f'{hour}:{minute} {meridiem}'.strip()

def _preprocess(text: str) -> str:
    """시간·날짜 한국어 표현을 영어식으로 치환"""
    return _TIME_PAT.sub(_kr_time_to_en, text)

def _resolve_relative_date(raw: str, base: datetime) -> datetime:
    """
    '오늘/내일…' 과 같이 dateparser가 시간만 돌려준 경우
    직접 날짜를 보정해 줌.
    """
    for k, offset in _REL_DAY.items():
        if k in raw:
            return base + timedelta(days=offset)
    return base

def extract_schedules(text: str):
    """
    한국어 문장에서 날짜·시간을 ISO‑8601(+09:00)로 추출.
    '오늘 아침 10시' → 2025‑04‑10T10:00:00+09:00
    """
    now = datetime.now(KST)
    preprocessed = _preprocess(text)

    results = search_dates(
        preprocessed,
        languages=["ko", "en"],
        settings={
            "PREFER_DATES_FROM": "future",
            "TIMEZONE": "Asia/Seoul",
            "RETURN_AS_TIMEZONE_AWARE": True,
        },
    )

    schedules = []
    if results:
        for raw, dt in results:
            # dateparser가 '10:00 AM' → 오늘 날짜 없이 시간만 줄 수 있음
            if dt.date() == datetime.utcnow().date() and raw.strip().startswith(tuple(_KR_MERIDIEM.keys())):
                dt = _resolve_relative_date(raw, now).replace(
                    hour=dt.hour, minute=dt.minute, second=0, microsecond=0
                )
                dt = KST.localize(dt) if dt.tzinfo is None else dt
            elif dt.tzinfo is None:
                dt = KST.localize(dt)
            schedules.append({"raw": raw.strip(), "datetime": dt.isoformat()})
    return schedules
