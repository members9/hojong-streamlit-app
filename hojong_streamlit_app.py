import faiss
import pickle
import numpy as np
from collections import deque
import streamlit as st
from openai import OpenAI

# OpenAI 클라이언트 초기화
client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])

# FAISS 및 메타데이터 로드
index = faiss.read_index("service_index.faiss")
with open("service_metadata.pkl", "rb") as f:
    metadata = pickle.load(f)

def normalize(vecs):
    norms = np.linalg.norm(vecs, axis=1, keepdims=True)
    return vecs / norms

xb = index.reconstruct_n(0, index.ntotal)
xb = normalize(xb)
d = xb.shape[1]
index_cosine = faiss.IndexFlatIP(d)
index_cosine.add(xb)

SIMILARITY_THRESHOLD = 0.30

# 세션 상태 초기화
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "last_results" not in st.session_state:
    st.session_state.last_results = []
if "excluded_company_ids" not in st.session_state:
    st.session_state.excluded_company_ids = set()
if "all_results" not in st.session_state:
    st.session_state.all_results = deque(maxlen=3)
if "user_input" not in st.session_state:
    st.session_state.user_input = ""
if "selected_service" not in st.session_state:
    st.session_state.selected_service = None

def get_embedding(text, model="text-embedding-3-small"):
    response = client.embeddings.create(input=[text], model=model)
    return response.data[0].embedding

def is_best_recommendation_query(query):
    keywords = ["가장", "최고", "제일", "1등", "1위", "진짜 추천", "강력 추천", "정말 추천", "최선의 방안"]
    return any(k in query for k in keywords)

def recommend_services(query, top_k=5, exclude_company_ids=None):
    query_vec = get_embedding(query)
    query_vec = np.array(query_vec).astype('float32').reshape(1, -1)
    query_vec = normalize(query_vec)

    D, indices = index_cosine.search(query_vec, 100)
    results = []
    seen_companies = set(exclude_company_ids) if exclude_company_ids else set()

    for i in indices[0]:
        service = metadata[i]
        cid = service["기업ID"]
        if cid not in seen_companies:
            results.append(service)
            seen_companies.add(cid)
        if len(results) == top_k:
            break
    return results

def is_relevant_question(query, threshold=SIMILARITY_THRESHOLD):
    query_vec = get_embedding(query)
    query_vec = np.array(query_vec).astype('float32').reshape(1, -1)
    query_vec = normalize(query_vec)
    D, _ = index_cosine.search(query_vec, 1)
    max_similarity = D[0][0]
    st.session_state.similarity_score = max_similarity
    return max_similarity >= threshold

def ask_gpt(messages):
    response = client.chat.completions.create(model="gpt-4o", messages=messages)
    return response.choices[0].message.content

def make_context(results):
    return "\n".join([
        f"{i+1}. {s['서비스명']} ({s['기업명']})\n- 유형: {s.get('서비스유형', '정보 없음')}\n- 요약: {s.get('서비스요약', '')}\n- 금액: {s.get('서비스금액', '정보 없음')} / 기한: {s.get('서비스기한', '정보 없음')}"
        for i, s in enumerate(results)
    ])

def make_summary_context(summary_memory):
    seen = set()
    deduplicated = []
    for batch in reversed(summary_memory):
        for item in batch:
            key = (item['서비스명'], item['기업명'], item.get('서비스금액', '없음'))
            if key not in seen:
                seen.add(key)
                deduplicated.insert(0, item)
    return "\n".join([
        f"{i+1}. {s['서비스명']} ({s['기업명']})\n- 유형: {s.get('서비스유형', '')}\n- 요약: {s.get('서비스요약', '')}"
        for i, s in enumerate(deduplicated)
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
5. 4번의 답변 생성 시 반드시 서비스명과 기업명은 따옴표(")로 묶어주고, 목록 표기시에는 반드시 대시(-) 로만 나열해주세요.
6. 답변 시 불필요하게 특수문자(**, ## 등)로 머릿말을 사용 하지 말아주세요.
7. 부드러운 상담사 말투로 정리해주세요.
"""

# UI 구성
st.title("관광기업 서비스 추천 AI 🤖")
st.markdown("서비스 추천을 원하시는 질문을 하시면, 호종이가 도와드립니다!")

# 대화 출력 영역
st.markdown("---")
scroll_container = st.container()
with scroll_container:
    for user_msg, ai_msg in st.session_state.chat_history:
        st.markdown(f"**🙋 사용자 질문:** {user_msg}")
        st.markdown(ai_msg, unsafe_allow_html=True)
    st.markdown("ℹ️ 각 추천 서비스에 대해 더 알고 싶으면 '자세히 기업명'처럼 입력하세요.")

# 입력창
with st.form("input_form", clear_on_submit=True):
    user_input = st.text_area("질문을 입력하세요", key="user_input", height=80, label_visibility="collapsed")
    submitted = st.form_submit_button("질문하기", use_container_width=True)

    if submitted and user_input:
        if user_input.startswith("자세히") and st.session_state.all_results:
            keyword = user_input.replace("자세히", "").strip()
            all_services_flat = [s for batch in st.session_state.all_results for s in batch]
            matches = [s for s in all_services_flat if keyword in s["기업명"]]
            st.markdown(f"✅ 현재 keyword: {keyword}")
            st.markdown("✅ 현재 기업명 목록:\n" + "\n".join([s["기업명"] for s in all_services_flat]))

            if not matches:
                st.warning("❌ 해당 키워드를 포함한 기업명이 없습니다.")
            elif len(matches) > 1:
                st.warning("⚠️ 여러 개의 기업명이 일치합니다. 더 구체적으로 입력해주세요.")
                for m in matches:
                    st.markdown(f"- {m['기업명']}")
            else:
                s = matches[0]
                service_link = f"https://www.tourvoucher.or.kr/user/svcManage/svc/BD_selectSvc.do?svcNo={s['서비스번호']}"
                company_link = f"https://www.tourvoucher.or.kr/user/entrprsManage/provdEntrprs/BD_selectProvdEntrprs.do?entrprsId={s['기업ID']}"
                with st.expander("🔍 선택한 서비스 자세히 보기", expanded=True):
                    for k, v in s.items():
                        st.markdown(f"**{k}**: {v}")
                    st.markdown(f"[🔗 서비스 링크]({service_link})")
                    st.markdown(f"[🏢 기업 링크]({company_link})")
        else:
            if not is_relevant_question(user_input):
                st.warning("⚠️ 질문의 내용을 조금 더 관광기업이나 서비스와 관련된 내용으로 다시 작성해주세요.")
            else:
                st.info(f"🔍 질문과 관광기업 서비스간 유사도: {st.session_state.similarity_score:.4f}")
                best_mode = is_best_recommendation_query(user_input)
                exclude = None if best_mode else st.session_state.excluded_company_ids

                last_results = recommend_services(user_input, exclude_company_ids=exclude)
                st.session_state.last_results = last_results

                if not best_mode:
                    for s in last_results:
                        st.session_state.excluded_company_ids.add(s['기업ID'])

                st.session_state.all_results.append(last_results)

                context = make_context(last_results)
                gpt_prompt = make_prompt(user_input, context, is_best=best_mode)

                chat_history = [
                    {"role": "system", "content": "당신은 관광기업 상담 전문가 호종이입니다."},
                    {"role": "user", "content": gpt_prompt}
                ]
                reply = ask_gpt(chat_history)

                st.session_state.chat_history.append((user_input, reply))
                st.rerun()
