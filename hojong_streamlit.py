# ✅ Streamlit 기반 최종 통합 버전 (UI + 로직 통합)

import streamlit as st

# ✅ 스타일 및 반응형 CSS 추가
st.markdown("""
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Noto+Sans+KR&display=swap');
        
        html, body, .stApp {
            background-color: #000000 !important;
            color: #FFFFFF !important;
            font-family: 'Noto Sans KR', sans-serif !important;
        }

        /* ✅ 입력창 스타일 */
        textarea {
            background-color: #f0f0f0 !important;
            color: #000000 !important;
            border-radius: 6px !important;
            padding: 10px !important;
            border: 1px solid #666 !important;
        }

        /* ✅ 물어보기 버튼 스타일 */
        button[title="물어보기"] {
            background-color: #000000 !important;
            color: white !important;
            border: 1px solid #ccc !important;
            border-radius: 6px !important;
        }

        /* ✅ 사용자/챗봇 말풍선 */
        .user-msg-box {
            text-align: right;
        }
        .chatbot-msg-box {
            text-align: left;
        }
        .user-msg {
            display: inline-block; 
            text-align: left; 
            background-color: #FFEB3B; 
            color: #000000; 
            padding: 10px 14px; 
            border-radius: 12px 0px 12px 12px; 
            margin: 0 0 30px 0; 
            max-width: 66%;
        }
        .chatbot-msg {
            display: inline-block; 
            text-align: left; 
            background-color: #FFFFFF; 
            color: #000000; 
            padding: 10px 14px; 
            border-radius: 12px 0px 12px 12px; 
            margin: 0 0 30px 0; 
            max-width: 66%;
        }

        @media screen and (max-width: 768px) {
            .input-row {
                flex-direction: column;
                align-items: stretch;
            }
        }
    </style>
    <script>
        // 버튼에 title 강제 지정
        window.addEventListener('load', function () {
            const buttons = parent.document.querySelectorAll('button');
            for (const btn of buttons) {
                if (btn.innerText === "물어보기") {
                    btn.title = "물어보기";
                }
            }
        });
    </script>
""", unsafe_allow_html=True)


import faiss
import pickle
import numpy as np
import random
import itertools
from collections import deque
from openai import OpenAI


# ✅ 상태 초기화
if "conversation_history" not in st.session_state:
    st.session_state.conversation_history = [
        {"role": "system", "content": "당신은 관광기업 상담 전문가 호종이입니다. 모든 답변은 아래 지침을 따라야 합니다:\n\n- 답변은 친절한 상담사 말투로 작성해 주세요.\n- 사용자 질문에 '추천', '제시', '찾아', '검색해' 라는 단어가 포함되면 목록 형식으로 작성하고, 각 항목에 기업명, 서비스명, 기업ID, 서비스유형, 금액, 기한 정보를 포함해 주세요.\n- 목록은 아래 형식을 따르세요:\n  1. \"서비스명\"\n     - \"기업명\" (기업ID: XXXX)\n     - 유형: ...\n     - 금액: ...\n     - 기한: ...\n     - 요약: ...\n- 특수문자(**, ## 등)는 사용하지 말고, 불릿은 대시(-)로만 통일해 주세요. 항목 간 개행 없이 이어서 작성해 주세요.\n- 기업명/서비스명이 나오면 반드시 이유도 함께 설명해 주세요.\n- 기업ID는 반드시 괄호 안에 표기해 주세요. 예: \"제이어스\" (기업ID: 12345)"}
    ]
if "all_results" not in st.session_state:
    st.session_state.all_results = deque(maxlen=5)
if "excluded_company_ids" not in st.session_state:
    st.session_state.excluded_company_ids = set()
if "last_results" not in st.session_state:
    st.session_state.last_results = []
if "chat_messages" not in st.session_state:
    st.session_state.chat_messages = []

# ✅ GPT & FAISS 준비
client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])
index = faiss.read_index("service_index.faiss")
with open("service_metadata.pkl", "rb") as f:
    metadata = pickle.load(f)

xb = index.reconstruct_n(0, index.ntotal)
xb = xb / np.linalg.norm(xb, axis=1, keepdims=True)
d = xb.shape[1]
index_cosine = faiss.IndexFlatIP(d)
index_cosine.add(xb)

# ✅ 함수 정의
def get_embedding(text, model="text-embedding-3-small"):
    response = client.embeddings.create(input=[text], model=model)
    return response.data[0].embedding

def ask_gpt(messages):
    response = client.chat.completions.create(model="gpt-4o", messages=messages)
    return response.choices[0].message.content

def is_best_recommendation_query(query):
    return any(k in query for k in ["추천", "강추", "제시", "찾아", "검색해"])

def is_relevant_question(query, threshold=0.3):
    query_vec = get_embedding(query)
    query_vec = np.array(query_vec).astype('float32').reshape(1, -1)
    query_vec /= np.linalg.norm(query_vec, axis=1, keepdims=True)
    D, _ = index_cosine.search(query_vec, 1)
    return D[0][0] >= threshold

def recommend_services(query, top_k=5, exclude_company_ids=None):
    query_vec = get_embedding(query)
    query_vec = np.array(query_vec).astype('float32').reshape(1, -1)
    query_vec /= np.linalg.norm(query_vec, axis=1, keepdims=True)
    D, indices = index_cosine.search(query_vec, 100)
    
    candidate_services = {}
    for score, idx in zip(D[0], indices[0]):
        service = metadata[idx]
        cid = service["기업ID"]
        if exclude_company_ids and cid in exclude_company_ids:
            continue
        key = (cid, service.get("서비스유형"))
        candidate_services.setdefault(key, []).append((score, service))

    best_services = [random.choice(v) for v in candidate_services.values()]
    best_services.sort(key=lambda x: x[0], reverse=True)
    return [s for _, s in best_services[:top_k]]

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
        f"{i+1}. {s['서비스명']} ({s['기업명']})\n- 유형: {s.get('서비스유형')}\n- 요약: {s.get('서비스요약')}"
        for i, s in enumerate(deduped)
    ])

def make_prompt(query, context, is_best):
    extra = f"지금까지 추천한 서비스 목록은 다음과 같습니다:\n\n{make_summary_context(st.session_state.all_results)}\n\n이전에 추천된 기업도 포함해서 조건에 가장 부합하는 최고의 조합을 제시해주세요." if is_best else ""
    return f"""
당신은 관광수혜기업에게 추천 서비스를 제공하는 AI 상담사 호종이입니다.

사용자의 질문은 다음과 같습니다:
"{query}"

그리고 관련된 서비스 목록은 아래와 같습니다:
{context}

📌 {extra}
"""

# ✅ UI 출력 영역
st.markdown("""
    <h1 style='text-align: center;'>혁신바우처 서비스 파인더</h1>
    <p style='text-align: center; font-size:14px;'>🤖 호종이에게 관광기업 서비스에 대해 물어보세요.</p>
""", unsafe_allow_html=True)

for msg in st.session_state.chat_messages:
    if msg["role"] == "user":
        st.markdown(f"""
        <div class="user-msg-box">
            <div class="user-msg">
                {msg["content"].replace(chr(10), "<br>")}
            </div>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown(f"""
        <div class="chatbot-msg-box">
            <div class="chatbot-msg"> 
                {msg["content"].replace(chr(10), "<br>")}
            </div>
        </div>
        """, unsafe_allow_html=True)

with st.form("chat_form", clear_on_submit=True):
    st.markdown("<div class='input-row'>", unsafe_allow_html=True)
    user_input = st.text_area("", height=100, label_visibility="collapsed")
    submitted = st.form_submit_button("물어보기")
    st.markdown("</div>", unsafe_allow_html=True)

st.markdown("""
    <div style='font-size: 14px; margin-top: 4px; color: #CCCCCC; text-align: left;'>
        ℹ️ 사용법 안내:<br>
        •&nbsp;<b>"자세히 기업명"</b>을 입력하면 해당 기업의 상세 정보를 확인할 수 있어요.<br>
        •&nbsp;<b>"강력 추천"</b> 을 포함하여 질문하면 앞서 제시된 내용들을 포함한 전체 추천을 받아볼 수 있어요.<br>
        •&nbsp;복합적인 조건들을 이용한 질문으로 편리하게 사용해 보세요.<br>
        예를들어 "우리 회사는 외국인에게 국내 유명 관광지역을 소개하고 숙박을 연결해주는 서비스를 하고 있어. 회사 홈페이지를 디자인 중심으로 개편하고 싶고, 참 다국어는 필수고, 숙박지를 예약하고 결제하는 쇼핑몰 기능이 반드시 필요해. 또한 인스타그램으로 홍보도 잘 하는 것도 필수고. 이런걸 만족시킬 수 있는 조합을 만들어줘. 단, 예산은 합쳐서 5,000만원까지이고, 기간은 3.5개월안에는 마쳐야 해. 많은 소통을 위해 가급적 수도권 지역에 있는 회사였으면 좋겠고, 매출도 30억 이상되며 인원도 많아서 안정적인 지원도 받았으면 하고. 이런 회사들로 찾아봐줘. 또 어떻게 이들을 조합하면 되는지, 왜 추천했는지도 상세히 알려줘."
    </div>
""", unsafe_allow_html=True)   

if submitted and user_input.strip():
    st.session_state.conversation_history.append({"role": "user", "content": user_input})
    st.session_state.chat_messages.append({"role": "user", "content": user_input})

    if user_input.startswith("자세히"):
        keyword = user_input.replace("자세히", "").strip()
        all_results = list(itertools.chain.from_iterable(st.session_state.all_results))
        matches = [s for s in all_results if keyword in s["기업명"]]
        if not matches:
            reply = "❗ 해당 키워드를 포함한 기업명이 없습니다."
        elif len(matches) > 1:
            reply = "❗ 여러 개의 기업명이 일치합니다:<br>" + "<br>".join(f"- {s['기업명']}" for s in matches)
        else:
            s = matches[0]
            details = [f"• {k}: {v}" for k, v in s.items()]
            reply = "<br>".join(details)
        st.session_state.chat_messages.append({"role": "assistant", "content": reply})
        st.rerun()
    else:
        if not is_relevant_question(user_input):
            msg = "❗ 관광기업이나 서비스 관련 질문으로 다시 말씀해 주세요."
            st.session_state.chat_messages.append({"role": "assistant", "content": msg})
            st.rerun()

        best_mode = is_best_recommendation_query(user_input)
        exclude = None if best_mode else st.session_state.excluded_company_ids
        results = recommend_services(user_input, exclude_company_ids=exclude)

        if not best_mode:
            for s in results:
                st.session_state.excluded_company_ids.add(s["기업ID"])

        st.session_state.last_results = results
        st.session_state.all_results.append(results)

        context = make_context(results)
        prompt = make_prompt(user_input, context, is_best=best_mode)
        st.session_state.conversation_history.append({"role": "user", "content": prompt})

        gpt_reply = ask_gpt(st.session_state.conversation_history)
        st.session_state.conversation_history.append({"role": "assistant", "content": gpt_reply})
        st.session_state.chat_messages.append({"role": "assistant", "content": gpt_reply})

        st.rerun()
