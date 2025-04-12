# ✅ 최신 로직 + Streamlit UI/UX 통합 버전
# CLI 환경의 최신 로직(13_service_recommender.py)을 Streamlit UI에 통합

import streamlit as st
import faiss
import pickle
import numpy as np
import random
from collections import deque
import itertools
from datetime import datetime
from zoneinfo import ZoneInfo  # Python 3.9 이상
import openai
from sentence_transformers import SentenceTransformer

# ✅ 스타일 및 반응형 CSS 추가
st.markdown("""
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Noto+Sans+KR&display=swap');
        
        html, body, .stApp {
            background-color: #FFFFFF !important;
            color: #0c0c0c !important;
            font-family: 'Noto Sans KR', sans-serif !important;
        }

        /* ✅ 입력창 */
        .stTextArea textarea {
            background-color: #f0f0f0 !important;
            color: #0c0c0c !important;
            border-radius: 6px !important;
            padding: 10px !important;
            border: 1px solid #DDDDDD !important;
        }

        /* ✅ 버튼 */
        .stButton > button {
            background-color: #0c0c0c !important;
            color: #FFFFFF !important;
            padding: 10px 16px !important;
            border-radius: 6px !important;
            border: 2px solid #FFFFFF !important;
        }

        /* ✅ 사용자/챗봇 말풍선 */
        .user-msg-box {
            text-align: right !important;
        }
        .user-msg {
            display: inline-block !important;
            text-align: left !important; 
            background-color: #FFEB3B !important; 
            color: #0c0c0c !important;
            padding: 10px 14px !important; 
            border-radius: 12px 0px 12px 12px; 
            margin: 0 0 30px 0 !important; 
            max-width: 66% !important;
        }
        .user-msg-time {
            text-align: left !important;
            font-size: 11px;
            color: #666;
            margin-top: 2px;
            width: 100%;
        }    
        .chatbot-msg-box {
            text-align: left !important;
        }
        .chatbot-msg {
            display: inline-block !important; 
            text-align: left !important !important; 
            background-color: #bacee0 !important; 
            color: #0c0c0c !important !important;
            padding: 10px 14px !important; 
            border-radius: 12px 0px 12px 12px !important; 
            margin: 0 0 30px 0 !important; 
            max-width: 66% !important;
        }
        .chatbot-msg-time {
            text-align: right !important; 
            font-size: 11px;
            color: #666;
            margin-top: 2px;
            width: 100%;
        }   
        /* ✅ 기타 */
        .responsive-title {
            font-size: clamp(40px, 5vw, 60px) !important;
            font-weight: 700 !important;
            color: #0c0c0c !important;  
            text-align: center !important;
            white-space: nowrap !important;
            overflow: hidden !important;
            width: 100% !important;
            padding-bottom: 2px !important;
            margin-top: -52px !important;
        }
        .responsive-subtitle {
            font-size: clamp(14px, 5vw, 12px) !important;
            color: #0c0c0c !important;  
            text-align: center !important;
            white-space: nowrap !important;
            overflow: hidden !important;
            width: 100% !important;
            padding-bottom: 2px !important;
        }
        .user-guide {
            font-size: 14px !important;
            margin-top: 4px !important; 
            color: #0c0c0c !important; 
            text-align: left !important;
            line-height: 1.4 !important;
        }
        
        @media screen and (max-width: 768px) {
            .input-row {
                flex-direction: column !important;
                align-items: stretch !important;
            }
        }
        
        .main .block-container {
            padding-top: 1rem !important;
        }
    </style>
""", unsafe_allow_html=True)

# ✅ 설정 변수 (13_service_recommender.py와 일치하도록 유지)
USE_OPENAI_EMBEDDING = True  # 🔁 여기서 스위칭 가능 (True: OpenAI, False: 로컬 모델)
SIMILARITY_THRESHOLD = 0.30
MAX_HISTORY_LEN = 5  # 질문과 답변 히스로리 저장 컨텍스트 개수

# ✅ OpenAI API 키 설정
openai.api_key = st.secrets["OPENAI_API_KEY"]

# ✅ 로컬 모델 초기화 (필요 시)
if not USE_OPENAI_EMBEDDING:
    local_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

# ✅ KST 시간 가져오기
def get_kst_time():
    return datetime.now(ZoneInfo("Asia/Seoul")).strftime("%p %I:%M")

# ✅ 세션 상태 초기화
if "conversation_history" not in st.session_state:
    st.session_state.conversation_history = deque(maxlen=MAX_HISTORY_LEN + 1)  # system 메시지 포함을 위해 +1
    st.session_state.conversation_history.append({
        "role": "system",
        "content": "당신은 관광기업 상담 전문가 호종이입니다. "
                "모든 답변은 아래 지침을 따라야 합니다:\n\n"
                "- 답변은 친절한 상담사 말투로 작성해 주세요.\n"
                "- 사용자 질문에 \"추천\", \"제시\", \"찾아\", \"검색해\" 라는 단어가 포함되면 목록 형식으로 작성하고,"
                " 각 항목에 기업명, 서비스명, 기업ID, 서비스유형, 금액, 기한 정보를 포함해 주세요."
                " 만일 그렇지 않으며, 서술식일 경우 자연스럽고 포괄적인 설명으로 구성해 주세요.\n"
                "- 목록으로 출력할 경우 반드시 아래 형식으로 출력하세요:\n"
                "\n"
                "  1. \"서비스명\"\n"
                "     - \"기업명\" (기업ID: XXXX)\n"
                "     - 유형: ...\n"
                "     - 금액: ...\n"
                "     - 기한: ...\n"
                "     - 요약: ...\n"
                "     -...\n"
                "\n"
                "- 반드시 위 형식을 지켜주세요. 마크다운 스타일(**, ## 등)은 사용하지 마세요.\n"
                "- 불릿은 항상 대시(-)만 사용해 주세요.\n"
                "- 항목 간 개행을 넣지 마세요.\n"
                "- 기업명/서비스명이 언급되면 반드시 이유도 함께 제시해 주세요.\n"
                "- 기업ID는 반드시 괄호 안에 표기해 주세요. 예: \"제이어스\" (기업ID: 12345)"
    })
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
if "followup_cache" not in st.session_state:
    st.session_state.followup_cache = {}
if "user_query_history" not in st.session_state:
    st.session_state.user_query_history = []
if "embedding_query_text" not in st.session_state:
    st.session_state.embedding_query_text = None

# ✅ FAISS 인덱스 및 메타데이터 로드
@st.cache_resource
def load_index_and_metadata():
    index = faiss.read_index("service_index.faiss")
    with open("service_metadata.pkl", "rb") as f:
        metadata = pickle.load(f)
    
    # 저장된 임베딩 벡터를 재구성하고 정규화
    xb = index.reconstruct_n(0, index.ntotal)
    xb = normalize(xb)
    d = xb.shape[1]
    index_cosine = faiss.IndexFlatIP(d)
    index_cosine.add(xb)
    
    return index, metadata, index_cosine

index, metadata, index_cosine = load_index_and_metadata()

# ✅ 유틸리티 함수들
def normalize(vecs):
    norms = np.linalg.norm(vecs, axis=1, keepdims=True)
    return vecs / norms

def get_embedding(text, model="text-embedding-3-small"):
    if text in st.session_state.embedding_cache:
        return st.session_state.embedding_cache[text]

    if USE_OPENAI_EMBEDDING:
        response = openai.Embedding.create(input=[text], model=model)
        embedding = response['data'][0]['embedding']
    else:
        embedding = local_model.encode([text])[0].tolist()

    st.session_state.embedding_cache[text] = embedding
    return embedding

def is_followup_question(prev, current):
    key = (prev.strip(), current.strip())  # 전처리된 질문 쌍을 캐시 키로 사용

    if key in st.session_state.followup_cache:
        return st.session_state.followup_cache[key]

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
        st.session_state.followup_cache[key] = result  # ✅ 캐시 저장
        return result
    except Exception as e:
        return True  # 오류 시 기본은 후속 질문으로 간주

def is_best_recommendation_query(query):
    keywords = ["강력 추천", "강추"]
    return any(k in query for k in keywords)

def is_relevant_question(query, threshold=SIMILARITY_THRESHOLD):
    query_vec = get_embedding(query)
    query_vec = np.array(query_vec).astype('float32').reshape(1, -1)
    query_vec = normalize(query_vec)
    D, _ = index_cosine.search(query_vec, 1)
    max_similarity = D[0][0]
    return max_similarity >= threshold

def is_related_results_enough(ranked_results, threshold=0.35, top_n=3):
    """
    벡터 유사도 기반 추천 결과 중 상위 N개의 평균 유사도가 threshold 이상인지 확인.
    관련도가 낮으면 False 반환 → GPT 호출 방지 가능.
    """
    if not ranked_results or len(ranked_results) < top_n:
        return False
    top_scores = [score for score, _ in ranked_results[:top_n]]
    avg_score = sum(top_scores) / len(top_scores)
    return avg_score >= threshold

def recommend_services(query, top_k=5, exclude_keys=None, use_random=True):
    # ✅ exclude_keys가 None이면 빈 집합으로 초기화
    if exclude_keys is None:
        exclude_keys = set()
    
    # 1. 질의에 대한 임베딩 생성 및 정규화
    query_vec = get_embedding(query)
    query_vec = np.array(query_vec).astype('float32').reshape(1, -1)
    query_vec = normalize(query_vec)

    # ✅ 2. 유사한 서비스 300개 검색 (기존 100개 → 확장)
    D, indices = index_cosine.search(query_vec, 300)

    # 3. 유사도 높은 순서로 (score, service) 목록 생성
    ranked = [(score, metadata[idx]) for score, idx in zip(D[0], indices[0])]
    # ⛔ 유사도 낮을 경우 GPT 호출도 생략할 수 있도록 빈 리스트 반환
    if not is_related_results_enough(ranked):
        return []

    # 4. 중복 제거 및 제외 대상 필터링
    seen_keys = set()
    filtered = []
    for score, service in ranked:
        key = (service["기업ID"], service.get("서비스유형"), service.get("서비스명"))
        if key in exclude_keys or key in seen_keys:
            continue
        seen_keys.add(key)
        filtered.append((score, service))

    # 5. 유사도 내림차순 정렬 (이미 정렬돼 있으나 안전 차원에서 재정렬)
    filtered.sort(key=lambda x: x[0], reverse=True)

    # 6. 상위 10개 중 랜덤 선택 or top_k개 선택
    if use_random:
        top_10 = filtered[:10]
        selected = random.sample(top_10, min(len(top_10), top_k))
        results = [service for _, service in selected]
    else:
        results = [service for _, service in filtered[:top_k]]

    return results

def ask_gpt(messages):
    response = openai.ChatCompletion.create(model="gpt-4o", messages=messages)
    return response['choices'][0]['message']['content']

def make_context(results):
    """추천 결과 목록을 목록 형식으로 출력하도록 구성 (기업의 상세정보 포함)"""
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
    deduplicated = []
    for result_list in reversed(summary_memory):
        for item in result_list:
            key = (item['서비스명'], item['기업명'], item.get('서비스금액', '없음'))
            if key not in seen:
                seen.add(key)
                deduplicated.insert(0, item)  # 원래 순서 유지
    return "\n".join([
        f"{i+1}. {s['서비스명']} ({s['기업명']})\n- 유형: {s.get('서비스유형', '정보 없음')}\n- 요약: {s.get('서비스요약', '')}"
        for i, s in enumerate(deduplicated)
    ])

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

# ✅ UI 출력 영역
st.markdown("""
    <div class="responsive-title">혁신바우처 서비스 파인더</div>
    <p class="responsive-subtitle">🤖 호종이에게 관광기업 서비스에 대해 물어보세요.</p>
""", unsafe_allow_html=True)

# 채팅 메시지 표시
for msg in st.session_state.chat_messages:
    if msg["role"] == "user":
        st.markdown(f"""
        <div class="user-msg-box">
            <div class="user-msg">
                {msg["content"].replace(chr(10), "<br>")}
                <div class="user-msg-time">{msg['timestamp']}</div>
            </div>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown(f"""
        <div class="chatbot-msg-box">
            <div class="chatbot-msg"> 
                {msg["content"].replace(chr(10), "<br>")}
                <div class="chatbot-msg-time">{msg['timestamp']}</div>
            </div>
        </div>
        """, unsafe_allow_html=True)

# 입력 폼
with st.form("chat_form", clear_on_submit=True):
    st.markdown("<div class='input-row'>", unsafe_allow_html=True)
    user_input = st.text_area("", height=100, label_visibility="collapsed")
    submitted = st.form_submit_button("물어보기")
    st.markdown("</div>", unsafe_allow_html=True)

# 사용 안내 표시
st.markdown("""
    <div class="user-guide">
        ℹ️ 사용법 안내:<br>
        •&nbsp;<b>"자세히 기업명"</b>을 입력하면 해당 기업의 상세 정보를 확인할 수 있어요.<br>
        •&nbsp;<b>"강력 추천"</b> 을 포함하여 질문하면 앞서 제시된 내용들을 포함한 전체 추천을 받아볼 수 있어요.<br>
        •&nbsp;<b>"초기화"</b>를 입력하면 대화 내용과 추천 기록을 모두 지울 수 있어요.<br>
        •&nbsp;복합적인 조건들을 이용한 질문으로 편리하게 사용해 보세요.
    </div>
""", unsafe_allow_html=True)

# 메시지 처리 로직
if submitted and user_input.strip():
    # 시간대 설정
    current_time = get_kst_time()
    
    # 사용자 메시지 저장
    st.session_state.chat_messages.append({"role": "user", "content": user_input, "timestamp": current_time})
    
    # 초기화 명령 처리
    if user_input.lower() == "초기화":
        st.session_state.embedding_query_text = None
        st.session_state.excluded_keys.clear()
        st.session_state.all_results.clear()
        st.session_state.conversation_history.clear()
        st.session_state.conversation_history.append({
            "role": "system", 
            "content": "당신은 관광기업 상담 전문가 호종이입니다."
        })
        st.session_state.user_query_history = []
        
        # 초기화 응답 메시지 추가
        st.session_state.chat_messages.append({
            "role": "assistant", 
            "content": "🤖 호종이는 잠시 레드썬하고 다시 돌아왔습니다.", 
            "timestamp": current_time
        })
        st.rerun()
    
    # '자세히' 명령 처리
    elif user_input.startswith("자세히"):
        keyword = user_input.replace("자세히", "").strip()
        all_stored_results = list(itertools.chain.from_iterable(st.session_state.all_results))
        
        if not all_stored_results:
            reply = "ℹ️ 저장된 추천 내역이 없습니다."
        else:
            matches = [s for s in all_stored_results if keyword in s["기업명"]]
            if not matches:
                reply = "ℹ️ 해당 키워드를 포함한 기업명이 없습니다."
            elif len(matches) > 1:
                reply = "ℹ️ 여러 개의 기업명이 일치합니다. 더 구체적으로 입력해 주세요.\n" + "\n".join([f"- {s['기업명']}" for s in matches])
            else:
                s = matches[0]
                service_link = f"https://www.tourvoucher.or.kr/user/svcManage/svc/BD_selectSvc.do?svcNo={s['서비스번호']}"
                company_link = f"https://www.tourvoucher.or.kr/user/entrprsManage/provdEntrprs/BD_selectProvdEntrprs.do?entrprsId={s['기업ID']}"
                
                # 상세 정보 형식화
                details = []
                for k, v in s.items():
                    # 기업 3개년 평균 매출: 숫자를 정수, 콤마 구분 후 "원" 추가
                    if k == "기업 3개년 평균 매출":
                        try:
                            num = float(v)
                            v = format(round(num), ",") + "원"
                        except Exception:
                            pass
                    # 기업 인력현황: 정수로 표기 후 "명" 추가
                    elif k == "기업 인력현황":
                        try:
                            num = float(v)
                            v = f"{int(num)}명"
                        except Exception:
                            pass
                    # 기업 핵심역량: _x000D_ 제거
                    elif k == "기업 핵심역량":
                        try:
                            v = v.replace("_x000D_", "")
                        except Exception:
                            pass
                    details.append(f"{k}: {v}")
                
                links = [
                    f"🔗 서비스 링크: {service_link}",
                    f"🏢 기업 링크: {company_link}"
                ]
                reply = "📄 서비스 상세정보\n" + "\n".join(details) + "\n\n" + "\n".join(links)
        
        # 상세 정보 응답 메시지 추가
        st.session_state.chat_messages.append({
            "role": "assistant", 
            "content": reply, 
            "timestamp": current_time
        })
        st.rerun()
    
    # 일반 질문 처리
    else:
        # 대화 이력에 사용자 입력 추가
        st.session_state.conversation_history.append({"role": "user", "content": user_input})
        
        # 질문 관련성 확인
        if not is_relevant_question(user_input):
            reply = "ℹ️ 죄송하지만, 질문의 내용을 조금 더 관광기업이나 서비스와 관련된 내용으로 다시 해 주세요."
            st.session_state.chat_messages.append({
                "role": "assistant", 
                "content": reply, 
                "timestamp": current_time
            })
            st.rerun()
        
        # 후속 질문 판단
        if st.session_state.user_query_history:
            previous_input = st.session_state.user_query_history[-1]
            if not is_followup_question(previous_input, user_input):
                st.session_state.embedding_query_text = user_input
            # 후속 질문이면 이전 임베딩 유지
        else:
            # 최초 질문인 경우
            st.session_state.embedding_query_text = user_input
        
        # 질문 히스토리에 추가
        st.session_state.user_query_history.append(user_input)
        
        # 추천 모드 설정 및 서비스 추천
        best_mode = is_best_recommendation_query(user_input)
        exclude = None if best_mode else st.session_state.excluded_keys
        last_results = recommend_services(
            st.session_state.embedding_query_text, 
            exclude_keys=exclude, 
            use_random=not best_mode
        )
        
        # 추천 결과가 없을 경우
        if not last_results:
            reply = "🧭 관련된 추천 결과가 충분하지 않습니다.\n\n관광기업이나 서비스와 관련된 질문을 조금 더 구체적으로 해 주시면 감사하겠습니다!"
            st.session_state.chat_messages.append({
                "role": "assistant", 
                "content": reply, 
                "timestamp": current_time
            })
            st.rerun()
        
        # 추천 결과 기반 응답 생성
        unique_last_results = [
            s for s in last_results
            if (s["기업ID"], s.get("서비스유형"), s.get("서비스명")) not in st.session_state.excluded_keys
        ]
        context = make_context(unique_last_results)
        gpt_prompt = make_prompt(user_input, context, is_best=best_mode)
        
        # GPT에 프롬프트 전달
        st.session_state.conversation_history.append({"role": "user", "content": gpt_prompt})
        
        try:
            gpt_reply = ask_gpt(list(st.session_state.conversation_history))
        except Exception as e:
            gpt_reply = f"⚠️ 응답 생성 중 오류가 발생했습니다. 다시 시도해주세요: {str(e)}"
            
        # 응답을 저장하고 대화 이력에 추가
        st.session_state.conversation_history.append({"role": "assistant", "content": gpt_reply})
        
        # 이번 추천에서 언급된 서비스들 제외 대상으로 등록
        mentioned_keys = {
            (s["기업ID"], s.get("서비스유형"), s.get("서비스명"))
            for s in last_results
            if (
                str(s["기업ID"]) in gpt_reply and
                s["서비스명"] in gpt_reply
            )
        }
        
        # 제외 대상 업데이트
        st.session_state.excluded_keys.update(mentioned_keys)
        
        # 추천 결과 저장
        st.session_state.last_results = last_results
        st.session_state.all_results.append(last_results)
        
        # 챗봇 응답 메시지 추가
        st.session_state.chat_messages.append({
            "role": "assistant", 
            "content": gpt_reply, 
            "timestamp": current_time
        })
        
        st.rerun()