# ✅ 최신 로직 통합된 Streamlit UI 코드 (프롬프트 포함 재정비)

import streamlit as st
import faiss
import pickle
import numpy as np
import random
import itertools
from collections import deque
from openai import OpenAI
from datetime import datetime
from zoneinfo import ZoneInfo

# ✅ 전역 상수
SIMILARITY_THRESHOLD = 0.30
MAX_HISTORY_LEN = 5

# ✅ 사용자 질문 전체 히스토리 저장 리스트 (무한 저장)
user_query_history = []

# ✅ 상태 초기화
if "conversation_history" not in st.session_state:
    st.session_state.conversation_history = [{
        "role": "system",
        "content": "당신은 관광기업 상담 전문가 호종이입니다. 모든 답변은 아래 지침을 따라야 합니다:\n\n- 답변은 친절한 상담사 말투로 작성해 주세요.\n- 사용자 질문에 \"추천\", \"제시\", \"찾아\", \"검색해\" 라는 단어가 포함되면 목록 형식으로 작성하고, 각 항목에 기업명, 서비스명, 기업ID, 서비스유형, 금액, 기한 정보를 포함해 주세요.\n- 목록은 아래 형식을 따르세요:\n  1. \"서비스명\"\n     - \"기업명\" (기업ID: XXXX)\n     - 유형: ...\n     - 금액: ...\n     - 기한: ...\n     - 요약: ...\n- 특수문자(**, ## 등)는 사용하지 말고, 불릿은 대시(-)로만 통일해 주세요. 항목 간 개행 없이 이어서 작성해 주세요.\n- 기업명/서비스명이 나오면 반드시 이유도 함께 설명해 주세요.\n- 기업ID는 반드시 괄호 안에 표기해 주세요. 예: \"제이어스\" (기업ID: 12345)"
    }]
if "all_results" not in st.session_state:
    st.session_state.all_results = deque(maxlen=MAX_HISTORY_LEN)
if "excluded_keys" not in st.session_state:
    st.session_state.excluded_keys = set()
if "last_results" not in st.session_state:
    st.session_state.last_results = []
if "chat_messages" not in st.session_state:
    st.session_state.chat_messages = []
if "embedding_cache" not in st.session_state:
    st.session_state.embedding_cache = {}

# ✅ GPT 및 FAISS 세팅
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

# ✅ 보완된 상태 변수
if "user_query_history" not in st.session_state:
    st.session_state.user_query_history = deque(maxlen=5)
if "embedding_query_text" not in st.session_state:
    st.session_state.embedding_query_text = None

# ✅ GPT 후속 질문 판단 캐시 저장소 (전역)
followup_cache = {}

# ✅ 후속 질문 여부 판단
def is_followup_question(prev, current):
    key = (prev.strip(), current.strip())  # 전처리된 질문 쌍을 캐시 키로 사용

    if key in followup_cache:
        st.writer(f"⚠️ [CACHE HIT] Cache에 후속 질문 여부 판단 완료: {key}")
        return followup_cache[key]

    st.writer(f"🧠 [CACHE MISS] ChatGPT에 후속 질문 여부 판단 중: {key}")
    messages = [
        {"role": "system", "content": "다음 사용자 질문이 이전 질문에 대한 후속 질문인지 아닌지를 판단해 주세요. 후속이면 YES, 아니면 NO로만 답해 주세요."},
        {"role": "user", "content": f"이전 질문: {prev}\n현재 질문: {current}"}
    ]
    try:
        reply = openai.ChatCompletion.create(
            model="gpt-4o",
            messages=messages
        )
        answer = reply['choices'][0]['message']['content'].strip().lower()
        result = "yes" in answer  # 'yes' 포함 여부로 판단
        followup_cache[key] = result  # ✅ 캐시 저장
        return result
    except Exception as e:
        st.writer(f"[❌ GPT 오류] 후속 질문 판단 실패: {e}")
        return True  # 오류 시 기본은 후속 질문으로 간주

# ✅ 결과가 충분한지 판단
def is_related_results_enough(results):
    st.writer("⚠️ [INFO] 추천 결과의 연관성이 낮아 GPT 호출을 생략합니다.")
    return results and len(results) >= 3

# ✅ GPT 응답에서 실제 언급된 키만 추출
def parse_referenced_keys(response_text, result_list):
    referenced = set()
    for s in result_list:
        if str(s["기업ID"]) in response_text and s["서비스명"] in response_text:
            referenced.add((s["기업ID"], s.get("서비스유형"), s.get("서비스명")))
    return referenced

def get_kst_time():
    return datetime.now(ZoneInfo("Asia/Seoul")).strftime("%p %I:%M")

def get_embedding(text, model="text-embedding-3-small"):
    if text in st.session_state.embedding_cache:
        return st.session_state.embedding_cache[text]
    response = client.embeddings.create(input=[text], model=model)
    embedding = response.data[0].embedding
    st.session_state.embedding_cache[text] = embedding
    return embedding

def ask_gpt(messages):
    response = client.chat.completions.create(model="gpt-4o", messages=messages)
    return response.choices[0].message.content

def is_best_recommendation_query(query):
    return any(k in query for k in ["강력 추천", "강추"])

def is_relevant_question(query, threshold=SIMILARITY_THRESHOLD):
    vec = np.array(get_embedding(query)).astype('float32').reshape(1, -1)
    vec /= np.linalg.norm(vec, axis=1, keepdims=True)
    D, _ = index_cosine.search(vec, 1)
    return D[0][0] >= threshold

def recommend_services(query, top_k=5, exclude_keys=None, use_random=True):
    if exclude_keys is None:
        exclude_keys = set()
    vec = np.array(get_embedding(query)).astype('float32').reshape(1, -1)
    vec /= np.linalg.norm(vec, axis=1, keepdims=True)
    D, indices = index_cosine.search(vec, 300)

    ranked = [(score, metadata[idx]) for score, idx in zip(D[0], indices[0])]
    
    # 📌 STEP 1: 유사도 기준 정렬된 원본 상위 30개 출력
    st.write(f"\n📌 [STEP 1] 유사도 기준 정렬된 원본 상위 30개:")
    for i, (score, s) in enumerate(ranked[:30]):
        st.write(f"{i+1}. [{score:.4f}] {s['기업명']} / {s.get('서비스유형')} / {s.get('서비스명')}")

    # ✅ 4. 제외할 키 (기업ID + 서비스유형 + 서비스명) 정의
    if exclude_keys:
        st.write(f"\n🚫 [STEP 2] 제외 대상 키 수: {len(exclude_keys)}")
        for i, key in enumerate(list(exclude_keys)[:10]):
            st.write(f" - 제외 {i+1}: 기업ID={key[0]} / {key[1]} / {key[2]}")
    else:
        st.write("\n🚫 [STEP 2] 제외 대상 없음")

    seen_keys = set()
    filtered = []
    for score, s in ranked:
        key = (s["기업ID"], s.get("서비스유형"), s.get("서비스명"))
        if key in exclude_keys or key in seen_keys:
            continue
        seen_keys.add(key)
        filtered.append((score, s))

    filtered.sort(key=lambda x: x[0], reverse=True)
    
    # ✅ 상위 30개까지 출력 (디버깅 또는 로그 확인용)
    st.write(f"\n✅ [STEP 3] 필터링 후 상위 30개:")
    for i, (score, s) in enumerate(filtered[:30]):
        st.write(f"{i+1}. [{score:.4f}] {s['기업명']} / {s.get('서비스유형')} / {s.get('서비스명')}")

    if use_random:
        top_10 = filtered[:10]
        selected = random.sample(top_10, min(len(top_10), top_k))
        return [s for _, s in selected]
    return [s for _, s in filtered[:top_k]]

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
            key = (item['서비스명'], item['기업명'], item.get('서비스금액', '없음'))
            if key not in seen:
                seen.add(key)
                deduped.insert(0, item)
    return "\n".join([
        f"{i+1}. {s['서비스명']} ({s['기업명']})\n- 유형: {s.get('서비스유형', '정보 없음')}\n- 요약: {s.get('서비스요약', '')}"
        for i, s in enumerate(deduped)
    ])

def make_prompt(query, context, is_best=False):
    recent_queries = "\n".join(f"- {q}" for q in st.session_state.user_query_history)
    style_instruction = (
        "- 답변은 목록 형식으로 출력해 주세요. 각 추천 항목은 번호를 붙이고,"
        "기업명, 서비스명, 서비스 유형, 금액, 기한, 법인여부, 위치, 핵심역량, 3개년 평균 매출, 해당분야업력, 주요사업내용, 인력현황을 상세하게 기술해 주세요.\n"
        "- 답변 시 반드시 서비스명과 기업명은 따옴표로 묶어주시고, 목록 표기 시에는 대시(-)로만 나열해 주세요.\n"
        "- 만약 사용자 질문과 충분한 연관성이 없다고 판단되면, 직접적인 추천이 어렵다고 먼저 말씀해 주세요.\n"
        "- 단, 유사한 키워드나 참고가 될 만한 서비스가 있다면 최대 1~2개만 예시로 소개해 주세요.\n"
        "- 불필요한 특수문자(**, ## 등)은 사용하지 말아 주세요.\n"
        "- 각 추천 항목 설명 시 반드시 기업ID를 괄호 안에 반드시 형태로 명시해 주세요. 예: \"기업명\" (기업ID: 1234)"
    )
    extra = f"지금까지 추천한 서비스 목록:\n\n{make_summary_context(st.session_state.all_results)}\n\n이전에 추천된 기업도 포함해서 조건에 가장 부합하는 최고의 조합을 제시해주세요." if is_best else ""

    return f"""
당신은 관광수혜기업에게 추천 서비스를 제공하는 AI 상담사 호종이입니다.

사용자의 질문:
"{query}"

관련된 서비스 목록:
{context}

📌 {extra}
📌 최근 사용자 질문 이력:\n" + recent_queries + "\n\n📌 다음 조건을 지켜서 답변해주세요:
{style_instruction}
"""

# ✅ UI 렌더링 및 입력 처리
st.title("혁신바우처 서비스 파인더")
st.write("🤖 호종이에게 관광기업 서비스에 대해 물어보세요.")

for msg in st.session_state.chat_messages:
    role_class = "user-msg" if msg["role"] == "user" else "chatbot-msg"
    time_class = "user-msg-time" if msg["role"] == "user" else "chatbot-msg-time"
    box_class = "user-msg-box" if msg["role"] == "user" else "chatbot-msg-box"
    st.markdown(f"""
    <div class='{box_class}'>
        <div class='{role_class}'>
            {msg['content'].replace(chr(10), '<br>')}
            <div class='{time_class}'>{msg['timestamp']}</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

with st.form("chat_form", clear_on_submit=True):
    user_input = st.text_area("", height=100, label_visibility="collapsed")
    submitted = st.form_submit_button("물어보기")

if submitted and user_input.strip():
    if user_input == st.session_state.get("embedding_query_text"):
        reply = "⚠️ 동일한 질문이 반복되어 GPT 응답을 생략합니다. 질문을 바꿔 주세요."
        st.session_state.chat_messages.append({"role": "assistant", "content": reply, "timestamp": get_kst_time()})
        st.rerun()

    st.session_state.embedding_query_text = user_input
    st.session_state.user_query_history.append(user_input)

    st.session_state.chat_messages.append({"role": "user", "content": user_input, "timestamp": get_kst_time()})

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
            details = []
            for k, v in s.items():
                if k == "기업 3개년 평균 매출":
                    try:
                        num = float(v)
                        v = f"{int(num):,}원"
                    except:
                        pass
                elif k == "기업 인력현황":
                    try:
                        num = float(v)
                        v = f"{int(num)}명"
                    except:
                        pass
                elif k == "기업 핵심역량":
                    try:
                        v = v.replace("_x000D_", "")
                    except:
                        pass
                details.append(f"• {k}: {v}")

            service_link = f"https://www.tourvoucher.or.kr/user/svcManage/svc/BD_selectSvc.do?svcNo={s['서비스번호']}"
            company_link = f"https://www.tourvoucher.or.kr/user/entrprsManage/provdEntrprs/BD_selectProvdEntrprs.do?entrprsId={s['기업ID']}"
            details.append(f"🔗 <b>서비스 링크:</b> <a href='{service_link}' target='_blank'>{service_link}</a>")
            details.append(f"🏢 <b>기업 링크:</b> <a href='{company_link}' target='_blank'>{company_link}</a>")
            reply = "<br>".join(details)

        st.session_state.chat_messages.append({"role": "assistant", "content": reply, "timestamp": get_kst_time()})
        st.rerun()

    elif not is_relevant_question(user_input):
        reply = "❗ 관광기업이나 서비스 관련 질문으로 다시 말씀해 주세요."
        st.session_state.chat_messages.append({"role": "assistant", "content": reply, "timestamp": get_kst_time()})
        st.rerun()

    else:
        
            # ✅ 후속 질문 판단: 이전 질문이 있을 때만 수행
        if user_query_history:
            previous_input = user_query_history[-1]
            if not is_followup_question(previous_input, user_input):
                st.writer("🔁 [INFO] 독립된 질문입니다. 기준 임베딩 갱신.")
                embedding_query_text = user_input
            else:
                st.writer("➡️ [INFO] 후속 질문입니다. 기준 임베딩 유지.")
        else:
            st.writer("🌱 [INFO] 최초 질문입니다. 기준 임베딩 설정.")
            embedding_query_text = user_input

        # 사용자 입력을 대화 이력과 히스토리에 각각 추가
        #conversation_history.append({"role": "user", "content": user_input})
        user_query_history.append(user_input)  # ✅ 무한 저장 리스트에 추가
        
        
        best_mode = is_best_recommendation_query(user_input)
        exclude = None if best_mode else st.session_state.excluded_keys
        results = recommend_services(user_input, exclude_keys=exclude, use_random=not best_mode)

        if not results:
            reply = "❗ 관련된 추천 결과가 충분하지 않습니다. 질문을 더 구체적으로 해 주세요."
            st.session_state.chat_messages.append({"role": "assistant", "content": reply, "timestamp": get_kst_time()})
            st.rerun()

        st.session_state.last_results = results
        st.session_state.all_results.append(results)

        context = make_context(results)
        prompt = make_prompt(user_input, context, is_best=best_mode)
        st.session_state.conversation_history.append({"role": "user", "content": prompt})
        gpt_reply = ask_gpt(st.session_state.conversation_history)
        st.session_state.conversation_history.append({"role": "assistant", "content": gpt_reply})
        st.session_state.chat_messages.append({"role": "assistant", "content": gpt_reply, "timestamp": get_kst_time()})

        mentioned_keys = parse_referenced_keys(gpt_reply, results)
        st.session_state.excluded_keys.update(mentioned_keys)
        st.rerun()