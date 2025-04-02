import torch
from kobert_tokenizer import KoBERTTokenizer
from transformers import BertModel

# SKT KoBERT 모델 및 토크나이저 로드
tokenizer = KoBERTTokenizer.from_pretrained('skt/kobert-base-v1')
model = BertModel.from_pretrained('skt/kobert-base-v1')

# 문장 임베딩 함수
def get_sentence_embedding(text):
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True)
    with torch.no_grad():
        outputs = model(**inputs)
    return outputs.last_hidden_state[:, 0, :].squeeze().numpy()  # 첫 번째 CLS 토큰 벡터 반환
