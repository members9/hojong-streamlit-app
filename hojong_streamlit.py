# service_recommender.py
import streamlit as st
import openai
import faiss
import pickle
import numpy as np
import random  # 랜덤 선택을 위한 모듈 임포트
import itertools  # 여러 리스트를 합치기 위해 사용
from collections import deque
from openai import OpenAI

# OpenAI 클라이언트 초기화
client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])

# FAISS 인덱스와 메타데이터 로드 (앱 시작 시 한 번만 로드)
index = faiss.read_index("service_index.faiss")
with open("service_metadata.pkl", "rb") as f:
    metadata = pickle.load(f)

def normalize(vecs):
    norms = np.linalg.norm(vecs, axis=1, keepdims=True)
    return vecs / norms

# 저장된 임베딩 벡터 재구성 및 정규화
xb = index.reconstruct_n(0, index.ntotal)
xb = normalize(xb)
d = xb.shape[1]
index_cosine = faiss.IndexFlatIP(d)
index_cosine.add(xb)

SIMILARITY_THRESHOLD = 0.30

# ----------------------- 함수 정의 ----------------------- #
def get_embedding(text, model="text-embedding-3-small"):
    response = openai.Embedding.create(input=[text], model=model)
    return response['data'][0]['embedding']

def is_best_recommendation_query(query):
    keywords = ["가장", "최고", "제일", "1등", "1위", "진짜 추천", "강력 추천", "정말 추천", "최선"]
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
        service_type = service.get("서비스유형", None)
        key = (cid, service_type)
        candidate_services.setdefault(key, []).append((score, service))
        
    best_services = []
    for candidate_list in candidate_services.values():
        chosen_score, chosen_service = random.choice(candidate_list)
        best_services.append((chosen_score, chosen_service))
    best_services.sort(key=lambda x: x[0], reverse=True)
    results = [service for score, service in best_services[:top_k]]
    return results

def is_relevant_question(query, threshold=SIMILARITY_THRESHOLD):
    query_vec = get_embedding(query)
    query_vec = np.array(query_vec).astype('float32').reshape(1, -1)
    query_vec = normalize(query_vec)
    D, _ = index_cosine.search(query_vec, 1)
    max_similarity = D[0][0]
    return max_similarity >= threshold

def ask_gpt(messages):
    response = openai.ChatCompletion.create(model="gpt-4o", messages=messages)
    return response['choices'][0]['message']['content']

def make_context(results):
    """목록 형식으로 기업의 상세정보 포함"""
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
    if "추천" in query:
        style_instruction = (
            "- 답변은 목록 형식으로 출력해 주세요. 각 추천 항목은 번호를 붙이고, "
            "기업명, 서비스명, 서비스 유형, 금액, 기한, 법인여부, 위치, 핵심역량, 3개년 평균 매출, "
            "해당분야업력, 주요사업내용, 인력현황을 상세하게 기술해 주세요.\n"
            "- 답변 시 반드시 서비스명과 기업명은 따옴표로 묶어주시고, 목록 표기 시에는 대시(-)로만 나열해 주세요."
        )
    else:
        style_instruction = (
            "- 답변은 서술식으로 작성해 주세요. 기업 정보(매출, 인원, 법인여부 등)도 포함하여 자연스럽게 설명해 주세요."
        )
    
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

📌 다음 조건을 지켜서 답변해주세요:
{style_instruction}
- 만약 동일한 서비스유형과 동일한 회사의 서비스명이 복수개일 경우에는 해당 회사의 단 1개의 서비스만 선택해 주세요.
- 불필요한 특수문자(**, ## 등)은 사용하지 말아 주세요.
- 부드러운 상담사 말투로 정리해 주세요.
"""

# ----------------------- Streamlit 상태 초기화 ----------------------- #
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
    st.session_state.chat_messages = []  # 사용자와 AI의 채팅을 저장

# ----------------------- 화면 구성 ----------------------- #
st.markdown("<h1 style='text-align: center;'>관광공사 서비스 가이드 AI</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center; font-size:14px;'>🤖 호종이에게 어떤 서비스나 기업을 찾으시는지 물어보세요.</p>", unsafe_allow_html=True)

chat_container = st.container()
with chat_container:
    # 채팅창 표시 (스크롤바 있는 영역)
    for msg in st.session_state.chat_messages:
        if msg["role"] == "user":
            st.markdown(f"<p style='background-color:#DCF8C6; padding:8px; border-radius:5px; text-align:right;'>{msg['content']}</p>", unsafe_allow_html=True)
        else:
            st.markdown(f"<p style='background-color:#FFFFFF; padding:8px; border-radius:5px; text-align:left;'>{msg['content']}</p>", unsafe_allow_html=True)

# 하단 메시지 안내
st.markdown("<p style='text-align:center; font-size:12px;'>""자세히 기업명"" 을 입력하시면 보다 상세한 정보를 얻을 수 있습니다.</p>", unsafe_allow_html=True)

# ----------------------- 채팅 입력 폼 ----------------------- #
with st.form(key="chat_form", clear_on_submit=True):
    user_input = st.text_area("메시지 입력", height=80)  # 3줄 정도의 높이
    submitted = st.form_submit_button("물어보기")

if submitted and user_input.strip() != "":
    # 사용자가 입력한 메시지를 대화 이력 및 채팅 메시지에 추가
    st.session_state.conversation_history.append({"role": "user", "content": user_input})
    st.session_state.chat_messages.append({"role": "user", "content": user_input})
    
    # '자세히' 명령 처리: 전체 저장 내역(all_results)에서 검색
    if user_input.startswith("자세히"):
        all_stored_results = list(itertools.chain.from_iterable(st.session_state.all_results))
        if not all_stored_results:
            detail_msg = "저장된 추천 내역이 없습니다."
            st.session_state.chat_messages.append({"role": "assistant", "content": detail_msg})
        else:
            keyword = user_input.replace("자세히", "").strip()
            matches = [s for s in all_stored_results if keyword in s["기업명"]]
            if not matches:
                detail_msg = "해당 키워드를 포함한 기업명이 없습니다."
                st.session_state.chat_messages.append({"role": "assistant", "content": detail_msg})
            elif len(matches) > 1:
                multiple_msg = "여러 개의 기업명이 일치합니다. 더 구체적으로 입력해 주세요.\n"
                for s in matches:
                    multiple_msg += f"- {s['기업명']}\n"
                st.session_state.chat_messages.append({"role": "assistant", "content": multiple_msg})
            else:
                s = matches[0]
                service_link = f"https://www.tourvoucher.or.kr/user/svcManage/svc/BD_selectSvc.do?svcNo={s['서비스번호']}"
                company_link = f"https://www.tourvoucher.or.kr/user/entrprsManage/provdEntrprs/BD_selectProvdEntrprs.do?entrprsId={s['기업ID']}"
                detail_lines = []
                for k, v in s.items():
                    if k == "기업 3개년 평균 매출":
                        try:
                            num = float(v)
                            v = format(int(num), ",") + "원"
                        except Exception:
                            pass
                    elif k == "기업 인력현황":
                        try:
                            num = float(v)
                            v = f"{int(num)}명"
                        except Exception:
                            pass
                    elif k == "기업 핵심역량":
                        try:
                            v = v.replace("_x000D_", "")
                        except Exception:
                            pass
                    detail_lines.append(f"{k}: {v}")
                detail_msg = "\n".join(detail_lines)
                detail_msg += f"\n🔗 서비스 링크: {service_link}\n🏢 기업 링크: {company_link}"
                st.session_state.chat_messages.append({"role": "assistant", "content": detail_msg})
    else:
        # 일반 질문 처리
        if not is_relevant_question(user_input):
            msg = "죄송하지만, 질문의 내용을 조금 더 관광기업이나 서비스와 관련된 내용으로 다시 해 주세요."
            st.session_state.chat_messages.append({"role": "assistant", "content": msg})
        else:
            best_mode = is_best_recommendation_query(user_input)
            exclude = None if best_mode else st.session_state.excluded_company_ids
            last_results = recommend_services(user_input, exclude_company_ids=exclude)
            # 제외 대상 업데이트
            if not best_mode:
                for s in last_results:
                    st.session_state.excluded_company_ids.add(s['기업ID'])
            st.session_state.last_results = last_results
            st.session_state.all_results.append(last_results)
            context = make_context(last_results)
            prompt_text = make_prompt(user_input, context, is_best=best_mode)
            st.session_state.conversation_history.append({"role": "user", "content": prompt_text})
            
            # AI에게 전체 대화 이력을 전달하여 응답 생성
            gpt_reply = ask_gpt(st.session_state.conversation_history)
            try:
                gpt_reply = gpt_reply.replace("\n\n", "\n")
            except Exception:
                pass
            st.session_state.conversation_history.append({"role": "assistant", "content": gpt_reply})
            st.session_state.chat_messages.append({"role": "assistant", "content": gpt_reply})
    
    # 강제로 페이지 재실행하여 최신 대화가 반영되도록 함.
    st.experimental_rerun()
