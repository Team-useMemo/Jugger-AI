import re

def separate_text_and_urls(text):
    # URL 정규 표현식 패턴 (도메인과 경로 부분만 추출)
    url_pattern = re.compile(r'https?://[a-zA-Z0-9.-]+\.[a-zA-Z]{2,6}(/\S*)?')

    # URL 추출 (finditer 사용)
    urls = [match.group() for match in url_pattern.finditer(text)]

    # URL을 제외한 일반 텍스트 추출
    text_without_urls = url_pattern.sub('', text).strip()

    return text_without_urls, urls

# 테스트
sample_text = "이 영화는 너무 재미있어요! 예매는http://ticket.com 에서 할 수 있어요."
text, urls = separate_text_and_urls(sample_text)

print("📌 일반 텍스트:", text)
print("🔗 추출된 URL:", urls)
