import asyncio
from typing import List, Tuple

import httpx
import validators


async def validate_urls(
    urls: List[str],
    timeout: float = 5.0,
    max_concurrency: int = 20,
    ok_status: range = range(200, 400),
) -> Tuple[List[str], List[str]]:

    syntactically_valid = [u for u in urls if validators.url(u)]
    syntactically_invalid = [u for u in urls if u not in syntactically_valid]

    if not syntactically_valid:
        return [], syntactically_invalid

    valid, invalid = [], syntactically_invalid

    async with httpx.AsyncClient(
        timeout=timeout,
        follow_redirects=True,
        headers={"User-Agent": "url-validator/1.0"},
    ) as client:

        sem = asyncio.Semaphore(max_concurrency)

        async def check(u: str):
            async with sem:
                try:
                    r = await client.head(u)
                    if r.status_code == 405:
                        r = await client.get(u, headers={"Range": "bytes=0-0"})
                    if r.status_code in ok_status:
                        valid.append(u)
                    else:
                        invalid.append(u)
                except Exception:
                    invalid.append(u)

        await asyncio.gather(*(check(u) for u in syntactically_valid))

    return valid, invalid
