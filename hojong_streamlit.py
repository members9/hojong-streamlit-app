import streamlit as st
import faiss
import pickle
import numpy as np
import random
import itertools
from collections import deque
from openai import OpenAI

# 기본 설정
SIMILARITY_THRESHOLD = 0.30
MAX_HISTORY_LEN = 10

# OpenAI Client (환경변수 또는 .streamlit/secrets.toml에 설정)
client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])

# FAISS 인덱스 및 메타데이터 로드
index = faiss.read_index("service_index.faiss")
with open("service_metadata.pkl", "rb") as f:
    metadata = pickle.load(f)

# 벡터 정규화 및 cosine 인덱스 준비
xb = index.reconstruct_n(0, index.ntotal)
xb = xb / np.linalg.norm(xb, axis=1, keepdims=True)
d = xb.shape[1]
index_cosine = faiss.IndexFlatIP(d)
index_cosine.add(xb)

# Streamlit 상태 변수 초기화
if "conversation_history" not in st.session_state:
    st.session_state.conversation_history = [
        {"role": "system", "content": "당신은 관광기업 상담 전문가 호종이입니다. 모든 답변은 다음 지침을 따릅니다:\n"
                                           "- 답변은 친절한 상담사 말투로 작성해 주세요.\n"
                                           "- 질문에 추천/제시/검색 등의 단어가 있다면 목록 형식, 아니면 서술식으로 답해 주세요.\n"
                                           "- 목록 형식은 반드시 번호와 대시(-)만 사용하고, Markdown 특수문자는 쓰지 마세요.\n"
                                           "- 기업명은 따옴표로 묶고, 기업ID는 괄호로 표기해 주세요.\n"
                                           "- 항목 간 개행 없이 출력해 주세요."}
    ]
if "all_results" not in st.session_state:
    st.session_state.all_results = deque(maxlen=MAX_HISTORY_LEN)
if "excluded_keys" not in st.session_state:
    st.session_state.excluded_keys = set()
if "chat_messages" not in st.session_state:
    st.session_state.chat_messages = []

# 임베딩 생성 함수
def get_embedding(text):
    response = client.embeddings.create(input=[text], model="text-embedding-3-small")
    return np.array(response.data[0].embedding).astype("float32").reshape(1, -1)

# GPT 호출 함수
def ask_gpt(messages):
    reply = client.chat.completions.create(model="gpt-4o", messages=messages)
    return reply.choices[0].message.content

# 추천 여부 판별
def is_best_recommendation_query(query):
    return any(k in query for k in ["강력 추천", "강추", "제시", "추천", "찾아", "검색"])

# 유사도 필터
def is_relevant_question(query, threshold=SIMILARITY_THRESHOLD):
    vec = get_embedding(query)
    D, _ = index_cosine.search(vec / np.linalg.norm(vec), 1)
    return D[0][0] >= threshold

# 추천 함수
def recommend_services(query, top_k=5, exclude_keys=None):
    if exclude_keys is None:
        exclude_keys = set()
    vec = get_embedding(query)
    D, indices = index_cosine.search(vec / np.linalg.norm(vec), 300)

    ranked = [(score, metadata[idx]) for score, idx in zip(D[0], indices[0])]
    seen_keys, filtered = set(), []
    for score, s in ranked:
        key = (s['기업ID'], s.get('서비스유형'), s.get('서비스명'))
        if key in seen_keys or key in exclude_keys:
            continue
        seen_keys.add(key)
        filtered.append(s)
    return filtered[:top_k]

# 출력용 context 생성
def make_context(results):
    return "\n".join([
        f"{i+1}. \"{s['서비스명']}\" - \"{s['기업명']}\" (기업ID: {s['기업ID']})\n"
        f"- 유형: {s.get('서비스유형', '정보 없음')}\n"
        f"- 요약: {s.get('서비스요약', '')}\n"
        f"- 금액: {s.get('서비스금액', '정보 없음')} / 기한: {s.get('서비스기한', '정보 없음')}"
        for i, s in enumerate(results)
    ])

# 요약용 context

def make_summary_context(summary_memory):
    seen, deduped = set(), []
    for result_list in reversed(summary_memory):
        for item in result_list:
            key = (item['서비스명'], item['기업명'])
            if key not in seen:
                seen.add(key)
                deduped.insert(0, item)
    return "\n".join([
        f"{i+1}. {s['서비스명']} ({s['기업명']})\n- 유형: {s.get('서비스유형', '정보 없음')}\n- 요약: {s.get('서비스요약', '')}"
        for i, s in enumerate(deduped)
    ])

# 프롬프트 생성
def make_prompt(query, context, is_best):
    extra = f"\n지금까지 추천된 서비스 목록:\n{make_summary_context(st.session_state.all_results)}" if is_best else ""
    return f"""
당신은 관광수혜기업에게 추천 서비스를 제공하는 AI 상담사 호종이입니다.

사용자 질문:
"{query}"

추천 가능한 서비스 목록:
{context}
{extra}
"""

# ---------- UI 시작 ---------- #
st.markdown("""
<h1 style='text-align: center;'>🎯 혁신이용권 서비스 파인더</h1>
<p style='text-align: center; font-size:14px;'>🤖 호종이에게 관광기업 서비스에 대해 물어보세요.</p>
""", unsafe_allow_html=True)

# 채팅 출력
for msg in st.session_state.chat_messages:
    st.markdown(f"<div style='background:#f1f1f1; padding:10px; margin:10px 0; border-radius:6px;'>" +
                msg['content'].replace("\n", "<br>") + "</div>", unsafe_allow_html=True)

# 사용자 입력
with st.form("chat_form", clear_on_submit=True):
    user_input = st.text_area("", height=80, label_visibility="collapsed")
    submitted = st.form_submit_button("호종이에게 물어보기")

if submitted and user_input.strip():
    st.session_state.chat_messages.append({"role": "user", "content": user_input})
    st.session_state.conversation_history.append({"role": "user", "content": user_input})

    if not is_relevant_question(user_input):
        msg = "❗ 관광기업이나 서비스 관련 질문으로 다시 말씀해 주세요."
        st.session_state.chat_messages.append({"role": "assistant", "content": msg})
    else:
        best_mode = is_best_recommendation_query(user_input)
        exclude = None if best_mode else st.session_state.excluded_keys
        results = recommend_services(user_input, exclude_keys=exclude)

        if not best_mode:
            for s in results:
                st.session_state.excluded_keys.add((s['기업ID'], s.get('서비스유형'), s.get('서비스명')))

        st.session_state.all_results.append(results)
        context = make_context(results)
        prompt = make_prompt(user_input, context, is_best=best_mode)
        st.session_state.conversation_history.append({"role": "user", "content": prompt})

        gpt_reply = ask_gpt(st.session_state.conversation_history)
        st.session_state.conversation_history.append({"role": "assistant", "content": gpt_reply})
        st.session_state.chat_messages.append({"role": "assistant", "content": gpt_reply})

    st.rerun()
