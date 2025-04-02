import torch
# from kobert_tokenizer import KoBERTTokenizer
# from transformers import BertModel
#
# # SKT KoBERT 모델 및 토크나이저 로드
# tokenizer = KoBERTTokenizer.from_pretrained('skt/kobert-base-v1')
# model = BertModel.from_pretrained('skt/kobert-base-v1')
# Load model directly
from transformers import AutoTokenizer, AutoModel

tokenizer = AutoTokenizer.from_pretrained("skt/kobert-base-v1")
model = AutoModel.from_pretrained("skt/kobert-base-v1")
# from transformers import AutoTokenizer, AutoModel

# tokenizer = AutoTokenizer.from_pretrained("skt/kobert-base-v1", trust_remote_code=True,use_fast=False)
# model = AutoModel.from_pretrained("skt/kobert-base-v1", trust_remote_code=True)
# 문장 임베딩 함수
# def get_sentence_embedding(text):
#     inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True)
#     with torch.no_grad():
#         outputs = model(**inputs)
#     return outputs.last_hidden_state[:, 0, :].squeeze().numpy()  # 첫 번째 CLS 토큰 벡터 반환
def get_sentence_embedding(text):
    inputs = tokenizer(
        text,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=512  # KoBERT의 최대 입력 길이
    )
    inputs.pop("token_type_ids", None)  # KoBERT는 token_type_ids 필요 없음

    with torch.no_grad():
        outputs = model(**inputs)
    return outputs.last_hidden_state[:, 0, :].squeeze().numpy()  # CLS 토큰의 임베딩 반환
