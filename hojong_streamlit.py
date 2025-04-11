# ✅ Streamlit 기반 관광기업 추천 챗봇 (원본 프롬프트 및 문구 완전 보존)

import streamlit as st
import faiss
import pickle
import numpy as np
import random
import itertools
from collections import deque
from openai import OpenAI

# ✅ OpenAI 클라이언트 준비
client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])

# ✅ 기본 스타일 정의
st.markdown("""
    <style>
        html, body, [class*="css"] {
            background-color: #000 !important;
            color: #fff !important;
            font-family: 'Noto Sans KR', sans-serif;
        }
        .log-message {
            font-size: 13px;
            margin-top: 10px;
            color: #aaa;
        }
    </style>
""", unsafe_allow_html=True)

# ✅ 전역 상수
SIMILARITY_THRESHOLD = 0.30
MAX_HISTORY_LEN = 10

# ✅ 상태 변수 초기화
if "conversation_history" not in st.session_state:
    st.session_state.conversation_history = deque(maxlen=MAX_HISTORY_LEN + 1)
    st.session_state.conversation_history.append({"role": "system", "content": "당신은 관광기업 상담 전문가 호종이입니다. 모든 답변은 아래 지침을 따라야 합니다:\n\n- 답변은 친절한 상담사 말투로 작성해 주세요.\n- 사용자 질문에 \"추천\", \"제시\", \"찾아\", \"검색해\" 라는 단어가 포함되면 목록 형식으로 작성하고, 각 항목에 기업명, 서비스명, 기업ID, 서비스유형, 금액, 기한 정보를 포함해 주세요. 만일 그렇지 않으며, 서술식일 경우 자연스럽고 포괄적인 설명으로 구성해 주세요.\n- 목록으로 출력할 경우 반드시 아래 형식으로 출력하세요:\n\n  1. \"서비스명\"\n     - \"기업명\" (기업ID: XXXX)\n     - 유형: ...\n     - 금액: ...\n     - 기한: ...\n     - 요약: ...\n     -...\n\n- 반드시 위 형식을 지켜주세요. 마크다운 스타일(**, ## 등)은 사용하지 마세요.\n- 불릿은 항상 대시(-)만 사용해 주세요.\n- 항목 간 개행을 넣지 마세요.\n- 기업명/서비스명이 언급되면 반드시 이유도 함께 제시해 주세요.\n- 기업ID는 반드시 괄호 안에 표기해 주세요. 예: \"제이어스\" (기업ID: 12345)"})
if "excluded_keys" not in st.session_state:
    st.session_state.excluded_keys = set()
if "all_results" not in st.session_state:
    st.session_state.all_results = deque(maxlen=MAX_HISTORY_LEN)
if "user_query_history" not in st.session_state:
    st.session_state.user_query_history = []

# ✅ 유틸 로그 출력
log_messages = []
def log(msg):
    log_messages.append(msg)

# ✅ FAISS index 로딩
index = faiss.read_index("service_index.faiss")
with open("service_metadata.pkl", "rb") as f:
    metadata = pickle.load(f)

xb = index.reconstruct_n(0, index.ntotal)
xb = xb / np.linalg.norm(xb, axis=1, keepdims=True)
d = xb.shape[1]
index_cosine = faiss.IndexFlatIP(d)
index_cosine.add(xb)

# ✅ 임베딩
embedding_cache = {}
def get_embedding(text, model="text-embedding-3-small"):
    if text in embedding_cache:
        log(f"[CACHE] 임베딩 캐시 사용: '{text}'")
        return embedding_cache[text]
    log(f"[EMBED] OpenAI 임베딩 생성: '{text}'")
    res = client.embeddings.create(input=[text], model=model)
    embedding = res.data[0].embedding
    embedding_cache[text] = embedding
    return embedding

# ✅ GPT 호출
def ask_gpt(messages):
    res = client.chat.completions.create(model="gpt-4o", messages=messages)
    return res.choices[0].message.content

# ✅ 서비스 목록 생성

def make_context(results):
    return "\n".join([
        f"{i+1}. \"{s['서비스명']}\" (제공기업: \"{s['기업명']}\", 기업ID: {s['기업ID']})\n"
        f"- 유형: {s.get('서비스유형', '정보 없음')}\n"
        f"- 요약: {s.get('서비스요약', '')}\n"
        f"- 금액: {s.get('서비스금액', '정보 없음')} / 기한: {s.get('서비스기한', '정보 없음')}\n"
        f"- 법인여부: {s.get('기업의 법인여부', '정보 없음')}\n"
        f"- 위치: {s.get('기업 위치', '정보 없음')}\n"
        f"- 핵심역량: {s.get('기업 핵심역량', '정보 없음')}\n"
        f"- 3개년 평균 매출: {s.get('기업 3개년 평균 매출', '정보 없음')}\n"
        f"- 해당분야업력: {s.get('기업 해당분야업력', '정보 없음')}\n"
        f"- 주요사업내용: {s.get('기업 주요사업내용', '정보 없음')}\n"
        f"- 인력현황: {s.get('기업 인력현황', '정보 없음')}"
        for i, s in enumerate(results)
    ])

# ✅ 추천 프롬프트 생성 (기존 원본 그대로)
def make_prompt(query, context, is_best=False):
    if is_best:
        history = make_summary_context(st.session_state.all_results)
        extra = f"지금까지 추천한 서비스 목록은 다음과 같습니다:\n\n{history}\n\n이전에 추천된 기업도 포함해서 조건에 가장 부합하는 최고의 조합을 제시해주세요."
    else:
        extra = ""

    return f"""당신은 관광수혜기업에게 추천 서비스를 제공하는 AI 상담사 호종이입니다.

사용자의 질문은 다음과 같습니다:
"{query}"

그리고 관련된 서비스 목록은 아래와 같습니다:
{context}

📌 {extra}
"""

# ✅ 요약 컨텍스트

def make_summary_context(summary_memory):
    seen = set()
    deduped = []
    for result_list in reversed(summary_memory):
        for item in result_list:
            key = (item['서비스명'], item['기업명'], item.get('서비스금액', ''))
            if key not in seen:
                seen.add(key)
                deduped.insert(0, item)
    return "\n".join([
        f"{i+1}. {s['서비스명']} ({s['기업명']})\n- 유형: {s.get('서비스유형', '정보 없음')}\n- 요약: {s.get('서비스요약', '')}"
        for i, s in enumerate(deduped)
    ])

# ✅ 추천 로직 (로컬 로직 그대로 유지)
def recommend_services(query, top_k=5, exclude_keys=None, use_random=True):
    if exclude_keys is None:
        exclude_keys = set()

    query_vec = get_embedding(query)
    query_vec = np.array(query_vec).astype('float32').reshape(1, -1)
    query_vec = query_vec / np.linalg.norm(query_vec)
    D, indices = index_cosine.search(query_vec, 300)
    ranked = [(score, metadata[idx]) for score, idx in zip(D[0], indices[0])]

    seen_keys = set()
    filtered = []
    for score, service in ranked:
        key = (service["기업ID"], service.get("서비스유형"), service.get("서비스명"))
        if key in exclude_keys or key in seen_keys:
            continue
        seen_keys.add(key)
        filtered.append((score, service))

    filtered.sort(key=lambda x: x[0], reverse=True)
    if use_random:
        top_10 = filtered[:10]
        selected = random.sample(top_10, min(len(top_10), top_k))
        return [s for _, s in selected]
    return [s for _, s in filtered[:top_k]]

# ✅ 사용자 인터페이스
st.title("🎯 관광기업 서비스 추천 챗봇 (호종이)")
st.markdown("""<small>관광 수혜기업을 위한 맞춤형 추천을 도와드려요.</small>""", unsafe_allow_html=True)

with st.form("chat_form", clear_on_submit=True):
    user_input = st.text_area("호종이에게 물어보세요", height=100)
    submitted = st.form_submit_button("물어보기")

if submitted and user_input:
    best_mode = any(k in user_input for k in ["강력 추천", "강추"])
    exclude = None if best_mode else st.session_state.excluded_keys

    st.session_state.user_query_history.append(user_input)
    results = recommend_services(user_input, exclude_keys=exclude, use_random=not best_mode)

    st.session_state.all_results.append(results)
    context = make_context(results)
    gpt_prompt = make_prompt(user_input, context, is_best=best_mode)

    st.session_state.conversation_history.append({"role": "user", "content": gpt_prompt})
    try:
        reply = ask_gpt(list(st.session_state.conversation_history))
    except Exception as e:
        reply = f"❗ GPT 응답 실패: {e}"

    st.session_state.conversation_history.append({"role": "assistant", "content": reply})

    # GPT 응답에서 언급된 항목 필터링
    mentioned_keys = {
        (s["기업ID"], s.get("서비스유형"), s.get("서비스명"))
        for s in results
        if (s["기업ID"] in reply and s["서비스명"] in reply)
    }
    st.session_state.excluded_keys.update(mentioned_keys)

    # 출력
    st.markdown("---")
    st.markdown("<b>🤖 호종이 추천:</b>", unsafe_allow_html=True)
    st.markdown(reply.replace("\n", "  "), unsafe_allow_html=True)

# ✅ 디버그 로그 출력
if log_messages:
    st.markdown("""<div class='log-message'>""" + "<br>".join(log_messages) + "</div>", unsafe_allow_html=True)
