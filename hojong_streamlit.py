# hojong_streamlit.py (반응형 개선 버전)
import streamlit as st
import faiss
import pickle
import numpy as np
import random
import itertools
from collections import deque
from openai import OpenAI

# ✅ 스타일 및 반응형 CSS 추가
st.markdown("""
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Noto+Sans+KR&display=swap');

        html, body, [class*="css"] {
            background-color: #000000 !important;
            color: #FFFFFF !important;
            font-family: 'Noto Sans KR', sans-serif !important;
        }

        .message-box {
            background-color: #F2F2FF;
            color: #000000;
            padding: 12px;
            border-radius: 8px;
            margin-bottom: 20px;
            word-break: break-word;
        }

        .chat-row {
            display: flex;
            flex-direction: column;
            gap: 12px;
        }

        .input-row {
            display: flex;
            flex-wrap: wrap;
            gap: 10px;
            align-items: flex-end;
            margin-top: 12px;
        }

        .input-row textarea {
            flex-grow: 1;
            min-width: 250px;
            background-color: #1c1c1c;
            color: #fff;
            border-radius: 6px;
            padding: 10px;
            border: 1px solid #444;
        }

        .stButton button {
            background-color: #444;
            color: white;
            padding: 8px 16px;
            border-radius: 6px;
            border: none;
        }

        @media screen and (max-width: 768px) {
            .input-row {
                flex-direction: column;
                align-items: stretch;
            }
        }
    </style>
""", unsafe_allow_html=True)

# OpenAI client 초기화
client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])

# FAISS 로딩
index = faiss.read_index("service_index.faiss")
with open("service_metadata.pkl", "rb") as f:
    metadata = pickle.load(f)

# 상태 초기화
if "conversation_history" not in st.session_state:
    st.session_state.conversation_history = [
        {"role": "system", "content": "당신은 관광기업 상담 전문가 호종이입니다."}
    ]
if "all_results" not in st.session_state:
    st.session_state.all_results = deque(maxlen=5)
if "excluded_company_ids" not in st.session_state:
    st.session_state.excluded_company_ids = set()
if "last_results" not in st.session_state:
    st.session_state.last_results = []
if "chat_messages" not in st.session_state:
    st.session_state.chat_messages = []

# 벡터 정규화 함수
def normalize(vecs):
    norms = np.linalg.norm(vecs, axis=1, keepdims=True)
    return vecs / norms

# cosine index 준비
xb = index.reconstruct_n(0, index.ntotal)
xb = normalize(xb)
d = xb.shape[1]
index_cosine = faiss.IndexFlatIP(d)
index_cosine.add(xb)
SIMILARITY_THRESHOLD = 0.30

# 임베딩 함수
def get_embedding(text, model="text-embedding-3-small"):
    response = client.embeddings.create(input=[text], model=model)
    return response.data[0].embedding

def ask_gpt(messages):
    response = client.chat.completions.create(model="gpt-4o", messages=messages)
    return response.choices[0].message.content

def is_best_recommendation_query(query):
    keywords = ["추천", "강추", "소개해", "안내해", "제시해"]
    return any(k in query for k in keywords)

def recommend_services(query, top_k=5, exclude_company_ids=None):
    query_vec = get_embedding(query)
    query_vec = np.array(query_vec).astype('float32').reshape(1, -1)
    query_vec = normalize(query_vec)
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

def is_relevant_question(query, threshold=SIMILARITY_THRESHOLD):
    query_vec = get_embedding(query)
    query_vec = np.array(query_vec).astype('float32').reshape(1, -1)
    query_vec = normalize(query_vec)
    D, _ = index_cosine.search(query_vec, 1)
    return D[0][0] >= threshold

def make_context(results):
    return "\n".join([
        f"{i+1}. {s['서비스명']} ({s['기업명']})\n"
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
    deduplicated = []
    for result_list in reversed(summary_memory):
        for item in result_list:
            key = (item['서비스명'], item['기업명'], item.get('서비스금액', '없음'))
            if key not in seen:
                seen.add(key)
                deduplicated.insert(0, item)
    return "\n".join([
        f"{i+1}. {s['서비스명']} ({s['기업명']})\n- 유형: {s.get('서비스유형', '정보 없음')}\n- 요약: {s.get('서비스요약', '')}"
        for i, s in enumerate(deduplicated)
    ])

def make_prompt(query, context, is_best=False):
    style_instruction = (
        "- 답변은 목록 형식으로 출력해 주세요. 각 추천 항목은 번호를 붙이고, 기업명, 서비스명, 서비스 유형, 금액, 기한, 법인여부, 위치, 핵심역량, 3개년 평균 매출, 해당분야업력, 주요사업내용, 인력현황을 상세하게 기술해 주세요.\n"
        "- 서비스명과 기업명은 반드시 따옴표로 묶고, 목록은 대시(-)로 나열해 주세요."
    ) if "추천" in query else "- 답변은 서술식으로 작성해 주세요. 기업 정보도 자연스럽게 설명해 주세요."

    extra = f"\n지금까지 추천한 서비스 목록:\n{make_summary_context(st.session_state.all_results)}" if is_best else ""

    return f"""
당신은 관광수혜기업에게 추천 서비스를 제공하는 AI 상담사 호종이입니다.

사용자의 질문은 다음과 같습니다:
"{query}"

관련 서비스 목록:
{context}

📌 조건:
{style_instruction}
{extra}
- 동일한 기업의 서비스가 여러 개일 경우, 하나만 선택해 주세요.
- 특수문자는 사용하지 말고, 부드러운 상담사 말투로 답변해 주세요.
"""

# ----------------------- UI ----------------------- #
st.markdown("""
    <h1 style='text-align: center;'>혁신이용권 서비스 파인더</h1>
    <p style='text-align: center; font-size:14px;'>🤖 호종이에게 관광기업 서비스에 대해 물어보세요.</p>
""", unsafe_allow_html=True)

for msg in st.session_state.chat_messages:
    role_class = "message-box"
    st.markdown(f"<div class='{role_class}'>{msg['content'].replace(chr(10), '<br>')}</div>", unsafe_allow_html=True)

with st.form("chat_form", clear_on_submit=True):
    st.markdown("<div class='input-row'>", unsafe_allow_html=True)
    user_input = st.text_area("", height=100, label_visibility="collapsed", key="input_box")
    submitted = st.form_submit_button("물어보기")
    st.markdown("</div>", unsafe_allow_html=True)
    
st.markdown("""
    <div style='font-size: 14px; margin-top: 4px; color: #CCCCCC; text-align: left;'>
        ℹ️ 사용법 안내:<br>
        • <b>"자세히 기업명"</b>을 입력하면 해당 기업의 상세 정보를 확인할 수 있어요.<br>
        • <b>"추천"</b> 으로 질문하면 종합 추천을 받아볼 수 있어요.<br>
        • 최근 추천받은 기업은 자동으로 중복 제외돼요.<br>
        • 복합적인 조건들을 이용한 질문으로 편리하게 사용해 보세요.<br>
        &nbsp;&nbsp;예를들어 "우리 회사는 국내 유명 관광지역을 소개하고 숙박을 연결해주는 서비스를 하고 있어. 회사 홈페이지를 디자인 중심으로 개편하고 싶고, 숙박지를 예약하고 결제하는 쇼핑몰 기능이 반드시 필요해. 이 2가지를 만족시킬 수 있는 조합을 만들어줘. 단, 예산은 합쳐서 5,000만원까지이고, 기간은 3.5개월안에는 마쳐야 해. 많은 소통을 위해 가급적 수도권 지역에 있는 회사였으면 좋겠고, 매출도 30억 이상되며 인원도 많아서 안정적인 지원도 받았으면 하고. 이런 회사를 좀 추천 해줘."

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
            service_link = f"https://www.tourvoucher.or.kr/user/svcManage/svc/BD_selectSvc.do?svcNo={s['서비스번호']}"
            company_link = f"https://www.tourvoucher.or.kr/user/entrprsManage/provdEntrprs/BD_selectProvdEntrprs.do?entrprsId={s['기업ID']}"
            details = []
            for k, v in s.items():
                if k == "기업 3개년 평균 매출":
                    try: v = format(int(v), ",") + "원"
                    except: pass
                elif k == "기업 인력현황":
                    try: v = f"{int(float(v))}명"
                    except: pass
                elif k == "기업 핵심역량":
                    v = v.replace("_x000D_", ", ")
                if k == "기업명":
                    details.append(f"• <b>{k}: {v}</b>")
                else:
                    details.append(f"• {k}: {v}")
            reply = "<br>".join(details) + f"<br>🔗 서비스 링크: <a href='{service_link}' target='_blank'>{service_link}</a><br>🏢 기업 링크: <a href='{company_link}' target='_blank'>{company_link}</a>"
        st.session_state.chat_messages.append({"role": "assistant", "content": reply})

    else:
        if not is_relevant_question(user_input):
            msg = "❗ 관광기업이나 서비스 관련 질문으로 다시 말씀해 주세요."
            st.session_state.chat_messages.append({"role": "assistant", "content": msg})
        else:
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
