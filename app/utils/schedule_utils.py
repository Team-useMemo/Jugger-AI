import re
from datetime import datetime, timedelta
from typing import List, Dict, Optional

import pytz
from dateparser import parse as dp_parse
from dateparser.search import search_dates


KST = pytz.timezone("Asia/Seoul")
_DEFAULT_DURATION = timedelta(0)

_KR_MERIDIEM = {
    "오전": "AM", "아침": "AM", "새벽": "AM",
    "오후": "PM", "점심": "PM", "낮": "PM", "저녁": "PM", "밤": "PM",
}

_TIME_PAT = re.compile(
    r'(?P<mer>(오전|오후|아침|점심|저녁|밤|새벽))?\s*'
    r'(?P<hour>\d{1,2})\s*시(?:\s*(?P<minute>\d{1,2})\s*분?)?\s*에?'
    r'|(?P<hour2>\d{1,2})\s*:\s*(?P<minute2>\d{2})'
)

_REL_DAY = {"오늘": 0, "내일": 1, "모레": 2, "글피": 3}

_RANGE_PAT = re.compile(
    r'(?P<start>(?:오전|오후|아침|점심|저녁|밤|새벽)?\s*\d{1,2}(?::?\d{0,2})?\s*시?(?:\s*\d{1,2}\s*분?)?)\s*'
    r'(부터|~|-|–|—)\s*'
    r'(?P<end>(?:오전|오후|아침|점심|저녁|밤|새벽)?\s*\d{1,2}(?::?\d{0,2})?\s*시?(?:\s*\d{1,2}\s*분?)?)'
    r'(까지)?'
)

_CLEAN_PAT = re.compile(
    r'(나는|제가|우리는|저는)?\s*'
    r'(오늘|내일|모레|글피)?\s*'
    r'(오전|오후|아침|점심|저녁|밤|새벽)?\s*'
)

def _kr_time_to_en(match: re.Match) -> str:
    mer = match.group('mer')
    hour = match.group('hour') or match.group('hour2')
    minute = match.group('minute') or match.group('minute2') or '00'
    meridiem = _KR_MERIDIEM.get(mer, '').strip()
    return f'{hour}:{minute} {meridiem}'.strip()


def _preprocess(text: str) -> str:
    return _TIME_PAT.sub(_kr_time_to_en, text)


def _resolve_relative_date(raw: str, base: datetime) -> datetime:
    for k, offset in _REL_DAY.items():
        if k in raw:
            return base + timedelta(days=offset)
    return base


def _parse_time(text: str, base: datetime) -> Optional[datetime]:
    pre = _preprocess(text)
    dt = dp_parse(
        pre,
        languages=["ko", "en"],
        settings={
            "PREFER_DATES_FROM": "future",
            "TIMEZONE": "Asia/Seoul",
            "RETURN_AS_TIMEZONE_AWARE": True,
        },
    )
    if dt is None:
        return None
    if dt.tzinfo is None:
        dt = KST.localize(dt)

    if dt.date() == datetime.utcnow().date():
        dt = _resolve_relative_date(text, base).replace(
            hour=dt.hour, minute=dt.minute, second=0, microsecond=0
        )
        if dt.tzinfo is None:
            dt = KST.localize(dt)
    return dt


def _extract_task(sentence: str, time_span: str) -> str:
    residual = sentence.replace(time_span, '', 1)
    residual = _CLEAN_PAT.sub('', residual)
    residual = residual.lstrip(' ,.-').strip()
    return residual or "작업 내용 미확정"


def extract_schedules(text: str) -> List[Dict]:

    now = datetime.now(KST)
    schedules: List[Dict] = []

    sentences = re.split(r'[.\n!?]', text)
    sentences = [s.strip() for s in sentences if s.strip()]

    for sent in sentences:
        for m in _RANGE_PAT.finditer(sent):
            raw_span = m.group(0)
            start_dt = _parse_time(m.group('start'), now)
            end_dt = _parse_time(m.group('end'), start_dt or now)

            if start_dt and end_dt and end_dt < start_dt:
                end_dt += timedelta(hours=12)

            schedules.append({
                "raw": sent,
                "startDate": start_dt.isoformat() if start_dt else None,
                "endDate": end_dt.isoformat() if end_dt else None,
                "task": _extract_task(sent, raw_span),
            })

        remaining = _RANGE_PAT.sub('', sent)
        singles = search_dates(
            _preprocess(remaining),
            languages=["ko", "en"],
            settings={
                "PREFER_DATES_FROM": "future",
                "TIMEZONE": "Asia/Seoul",
                "RETURN_AS_TIMEZONE_AWARE": True,
            },
        )
        if singles:
            for raw_span, dt in singles:
                if dt.tzinfo is None:
                    dt = KST.localize(dt)
                dt = _resolve_relative_date(raw_span, now).replace(
                    hour=dt.hour, minute=dt.minute, second=0, microsecond=0
                )
                if dt.tzinfo is None:
                    dt = KST.localize(dt)

                schedules.append({
                    "raw": sent,
                    "startDate": dt.isoformat(),
                    "endDate": (dt + _DEFAULT_DURATION).isoformat(),
                    "task": _extract_task(sent, raw_span),
                })

    return schedules
