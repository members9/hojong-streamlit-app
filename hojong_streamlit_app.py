
import openai
import faiss
import pickle
import numpy as np
from collections import deque
from openai import OpenAI
import streamlit as st

# 환경 설정
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

# 상태 변수 초기화
SIMILARITY_THRESHOLD = 0.30
if "excluded_company_ids" not in st.session_state:
    st.session_state.excluded_company_ids = set()
if "all_results" not in st.session_state:
    st.session_state.all_results = deque(maxlen=3)
if "last_results" not in st.session_state:
    st.session_state.last_results = []

# 임베딩
def get_embedding(text, model="text-embedding-3-small"):
    response = client.embeddings.create(input=[text], model=model)
    return response.data[0].embedding

# 유사도 필터링
def is_relevant_question(query):
    query_vec = get_embedding(query)
    query_vec = np.array(query_vec).astype("float32").reshape(1, -1)
    query_vec = normalize(query_vec)
    D, _ = index_cosine.search(query_vec, 1)
    st.session_state.similarity = float(D[0][0])
    return st.session_state.similarity >= SIMILARITY_THRESHOLD

# 추천 여부 판단
def is_best_recommendation_query(query):
    keywords = ["가장", "최고", "제일", "1등", "1위", "진짜 추천", "강력 추천", "정말 추천", "최선의 방안"]
    return any(k in query for k in keywords)

# 서비스 추천
def recommend_services(query, top_k=5, exclude_company_ids=None):
    query_vec = get_embedding(query)
    query_vec = np.array(query_vec).astype("float32").reshape(1, -1)
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

# 대화 생성
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
    for item in reversed(summary_memory):
        key = (item['서비스명'], item['기업명'], item.get('서비스금액', '없음'))
        if key not in seen:
            seen.add(key)
            deduplicated.insert(0, item)

    return "\n".join([
        f"{i+1}. {s['서비스명']} ({s['기업명']})\n- 유형: {s.get('서비스유형', '정보 없음')}\n- 요약: {s.get('서비스요약', '')}"
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
5. 4번의 답변 생성 시 반드시 서비스명과 기업명은 따옴표(")로 묶어주고, 목록 표기시에는 대시(-) 로만 나열해주세요.
6. 답변 시 불필요하게 특수문자(*, # 등)로 머릿말을 사용하지 말아주세요.
7. 부드러운 상담사 말투로 정리해주세요.
"""

# UI 구성
st.title("관광기업 서비스 추천 챗봇 🧳")
st.markdown("서비스 추천을 원하시는 질문을 하시면, 호종이가 도와드립니다!")

# 답변 출력 섹션
if "chat_log" not in st.session_state:
    st.session_state.chat_log = []

for i, chat in enumerate(reversed(st.session_state.chat_log)):
    st.markdown(f"**질문 {len(st.session_state.chat_log)-i}:** {chat['question']}")
    st.success(chat["answer"])

# 입력창 아래 유사도 메시지
if "similarity" in st.session_state:
    st.caption(f"🔎 입력한 질문과 서비스 데이터 간 유사도: {st.session_state.similarity:.4f}")

# 입력창
with st.form(key="query_form"):
    user_input = st.text_area("💬 질문을 입력하세요", height=80, placeholder="예: 우리 회사에 적합한 숙박 예약 플랫폼을 추천해줘")
    submitted = st.form_submit_button("호종이에게 물어보기")

if submitted and user_input.strip():
    if not is_relevant_question(user_input):
        st.warning("질문이 관광기업 서비스와 관련성이 낮습니다. 다시 입력해 주세요.")
    else:
        best_mode = is_best_recommendation_query(user_input)
        exclude = None if best_mode else st.session_state.excluded_company_ids
        results = recommend_services(user_input, exclude_company_ids=exclude)
        st.session_state.last_results = results

        if not best_mode:
            for s in results:
                st.session_state.excluded_company_ids.add(s["기업ID"])

        st.session_state.all_results.append(results)
        context = make_context(results)
        prompt = make_prompt(user_input, context, is_best=best_mode)

        messages = [
            {"role": "system", "content": "당신은 관광기업 상담 전문가 호종이입니다."},
            {"role": "user", "content": prompt}
        ]
        gpt_reply = ask_gpt(messages)

        st.session_state.chat_log.append({
            "question": user_input,
            "answer": gpt_reply
        })

        st.rerun()
