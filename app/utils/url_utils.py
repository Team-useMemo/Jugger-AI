import validators
import httpx
from typing import List, Tuple

async def validate_urls(urls: List[str], timeout: float = 5.0) -> Tuple[List[str], List[str]]:
    """
    주어진 URL 리스트를 형식(정규식) + 실제 응답 두 단계로 검증.
    반환: (valid_urls, invalid_urls)
    """
    syntactically_valid = [u for u in urls if validators.url(u)]

    if not syntactically_valid:
        return [], urls  # 전부 형식 오류

    valid, invalid = [], []
    async with httpx.AsyncClient(timeout=timeout, follow_redirects=True) as client:
        tasks = [client.head(u) for u in syntactically_valid]  # HEAD 요청으로 가볍게
        responses = await httpx.AsyncClient.gather(*tasks, return_exceptions=True)

        for u, resp in zip(syntactically_valid, responses):
            if isinstance(resp, Exception):
                invalid.append(u)
            elif 200 <= resp.status_code < 400:
                valid.append(u)
            else:
                invalid.append(u)
    return valid, invalid
