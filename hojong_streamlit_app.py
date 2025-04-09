import streamlit as st
from streamlit_chat import message
import openai
import faiss
import pickle
import numpy as np
from collections import deque
import os
from openai import OpenAI

# OpenAI API 설정
client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])

# FAISS 및 메타데이터 불러오기
index = faiss.read_index("service_index.faiss")
with open("service_metadata.pkl", "rb") as f:
    metadata = pickle.load(f)

xb = index.reconstruct_n(0, index.ntotal)
xb = xb / np.linalg.norm(xb, axis=1, keepdims=True)
d = xb.shape[1]
index_cosine = faiss.IndexFlatIP(d)
index_cosine.add(xb)

# 상태 초기화
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "last_results" not in st.session_state:
    st.session_state.last_results = []
if "excluded_company_ids" not in st.session_state:
    st.session_state.excluded_company_ids = set()
if "all_results" not in st.session_state:
    st.session_state.all_results = deque(maxlen=3)

SIMILARITY_THRESHOLD = 0.30

# 함수 정의
def get_embedding(text, model="text-embedding-3-small"):
    response = client.embeddings.create(input=[text], model=model)
    return np.array(response.data[0].embedding).astype('float32')

def is_best_recommendation_query(query):
    keywords = ["가장", "최고", "제일", "1등", "1위", "진짜 추천", "강력 추천", "정말 추천", "최선의 방안"]
    return any(k in query for k in keywords)

def recommend_services(query, top_k=5, exclude_company_ids=None):
    query_vec = get_embedding(query).reshape(1, -1)
    query_vec = query_vec / np.linalg.norm(query_vec)
    D, indices = index_cosine.search(query_vec, 100)
    results, seen_companies = [], set(exclude_company_ids) if exclude_company_ids else set()

    for i in indices[0]:
        service = metadata[i]
        if service["기업ID"] not in seen_companies:
            results.append(service)
            seen_companies.add(service["기업ID"])
        if len(results) == top_k:
            break
    return results

def is_relevant_question(query):
    vec = get_embedding(query).reshape(1, -1)
    vec = vec / np.linalg.norm(vec)
    D, _ = index_cosine.search(vec, 1)
    return D[0][0] >= SIMILARITY_THRESHOLD

def ask_gpt(messages):
    response = client.chat.completions.create(model="gpt-4o", messages=messages)
    return response.choices[0].message.content

def make_context(results):
    return "\n".join([
        f"{i+1}. {s['서비스명']} ({s['기업명']})\n- 유형: {s.get('서비스유형', '정보 없음')}\n- 요약: {s.get('서비스요약', '')}\n- 금액: {s.get('서비스금액', '정보 없음')} / 기한: {s.get('서비스기한', '정보 없음')}"
        for i, s in enumerate(results)
    ])

def make_summary_context(memory):
    seen, deduped = set(), []
    for group in reversed(memory):
        for s in group:
            key = (s['서비스명'], s['기업명'], s.get('서비스금액', ''))
            if key not in seen:
                seen.add(key)
                deduped.insert(0, s)
    return "\n".join([
        f"{i+1}. {s['서비스명']} ({s['기업명']})\n- 유형: {s.get('서비스유형', '정보 없음')}\n- 요약: {s.get('서비스요약', '')}"
        for i, s in enumerate(deduped)
    ])

def make_prompt(query, context, is_best=False):
    if is_best:
        history = make_summary_context(st.session_state.all_results)
        extra = f"지금까지 추천한 서비스 목록은 다음과 같습니다:\n{history}\n이전에 추천된 기업도 포함해서 조건에 가장 부합하는 최고의 조합을 제시해주세요."
    else:
        extra = "이전 추천된 기업과 중복되지 않는 새로운 추천을 최대 5개까지 부탁드립니다."

    return f"""당신은 관광수혜기업에게 추천 서비스를 제공하는 AI 상담사 호종이입니다.

사용자의 질문은 다음과 같습니다:
"{query}"

그리고 관련된 서비스 목록은 아래와 같습니다:
{context}

📌 {extra}

📌 다음 조건을 지켜서 추천해주세요:
1. 사용자의 질문 속 조건이나 목적을 우선 파악하세요.
2. 동일한 회사 또는 서비스는 중복하지 말고, 새로운 서비스 중심으로 추천해주세요.
3. 조건을 일부 완화하거나 유사한 목적을 가진 대체 서비스도 추천 가능합니다.
4. 각 추천은 번호를 붙이고, 기업명, 서비스명, 서비스 유형, 금액, 기한, 장점, 단점, 추천이유를 분석적으로 설명해주세요.
5. 4번의 답변 생성 시 반드시 서비스명과 기업명은 따옴표(\")로 묶어주고, 목록 표기시에는 대시(-) 로만 나열해주세요.
6. 답변 시 불필요하게 특수문자(*, # 등)로 머릿말을 사용 하지 말아주세요.
7. 부드러운 상담사 말투로 정리해주세요."""

# Streamlit 레이아웃
st.markdown("""
<h1 style='text-align: center;'>관광기업 서비스 추천 AI</h1>
<p style='text-align: center;'>서비스 추천을 원하시는 질문을 하시면, 호종이가 도와드립니다!</p>
""", unsafe_allow_html=True)

with st.container():
    # 채팅 출력 영역
    chat_box = st.container()
    with chat_box:
        for i, (q, a) in enumerate(st.session_state.chat_history):
            message(q, is_user=True, key=f"user_{i}")
            message(a, key=f"ai_{i}")

    # 사용자 입력
    user_input = st.text_area("", placeholder="예: 우리 홈페이지에 예약 시스템과 디자인을 개선하고 싶어요", height=80, key="user_text")
    send = st.button("호종이에게 질문하기")

# 하단 상태 출력
status_placeholder = st.empty()

if send and user_input.strip():
    status_placeholder.info("🤖 질문 분석 중...")
    if not is_relevant_question(user_input):
        status_placeholder.warning("❗ 관광기업 서비스와 관련된 질문을 해주세요.")
    else:
        best_mode = is_best_recommendation_query(user_input)
        exclude = None if best_mode else st.session_state.excluded_company_ids

        status_placeholder.info("🔍 관련 서비스 탐색 중...")
        results = recommend_services(user_input, exclude_company_ids=exclude)
        st.session_state.last_results = results

        if not best_mode:
            for s in results:
                st.session_state.excluded_company_ids.add(s['기업ID'])

        st.session_state.all_results.append(results)
        context = make_context(results)
        prompt = make_prompt(user_input, context, is_best=best_mode)

        messages = [
            {"role": "system", "content": "당신은 관광기업 상담 전문가 호종이입니다."},
            {"role": "user", "content": prompt},
        ]

        status_placeholder.info("✍️ 추천 정리 중...")
        reply = ask_gpt(messages)

        # 대화 기록 업데이트
        st.session_state.chat_history.append((user_input, reply))
        st.experimental_rerun()
