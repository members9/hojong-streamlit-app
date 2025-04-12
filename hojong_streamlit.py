
import streamlit as st
import streamlit.components.v1 as components
import faiss
import pickle
import numpy as np
import random
from collections import deque
import itertools
from datetime import datetime
from zoneinfo import ZoneInfo  # Python 3.9 이상
from sentence_transformers import SentenceTransformer

# ✅ 진입 암호 입력 로직 (4자리 숫자 예: 7299)
if "authenticated" not in st.session_state:
    st.session_state.authenticated = False

if not st.session_state.authenticated:
    st.markdown("## 🔐 접근 권한이 필요합니다")
    password_input = st.text_input("4자리 숫자 비밀번호를 입력하세요:", type="password")
    if password_input and password_input.strip() == "7299":
        st.session_state.authenticated = True
        st.rerun()
    elif password_input:
        st.error("❌ 비밀번호가 틀렸습니다.")
    st.stop()

# ✅ 인증 이후 실행
st.set_page_config(layout="wide")

from openai import OpenAI
client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])

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
            word-wrap: break-word !important;
            overflow-wrap: break-word !important;
            word-break: break-all !important;
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
            word-wrap: break-word !important;
            overflow-wrap: break-word !important;
            word-break: break-all !important;
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
        
        /* 링크 스타일 */
        .link-button {
            display: inline-block;
            background-color: #f8f9fa;
            color: #0366d6;
            padding: 5px 10px;
            border-radius: 4px;
            margin: 5px 0;
            text-decoration: none;
            border: 1px solid #ddd;
            white-space: normal;
            word-break: break-all;
            max-width: 100%;
        }
        
        @media screen and (max-width: 768px) {
            .input-row {
                flex-direction: column !important;
                align-items: stretch !important;
            }
            .user-msg, .chatbot-msg {
                font-size: 12px !important;
                line-height: 1.3 !important;
            }
            .user-msg-time, chatbot-msg-time {
                font-size: 9x;
            }   
            .user-guide {
                font-size: 11px !important;
                line-height: 1.3 !important;
            }
        }
        
        .main .block-container {
            padding-top: 1rem !important;
        }
    </style>
""", unsafe_allow_html=True)

# ✅ 설정 변수 (13_service_recommender.py와 일치하도록 유지)
USE_OPENAI_EMBEDDING = True  # 🔁 여기서 스위칭 가능 (True: OpenAI, False: 로컬 모델)
Q_SIMILARITY_THRESHOLD = 0.30
A_SIMILARITY_THRESHOLD = 0.45
MAX_HISTORY_LEN = 5  # 질문과 답변 히스로리 저장 컨텍스트 개수

# ✅ 세션 상태에 디버그 모드 변수 추가
if "debug_mode" not in st.session_state:
    st.session_state.debug_mode = False

# # 사이드바에 디버그 모드 토글 추가
# with st.sidebar:
#     st.title("개발자 설정")
#     debug_toggle = st.checkbox("디버그 모드", value=st.session_state.debug_mode)
#     if debug_toggle != st.session_state.debug_mode:
#         st.session_state.debug_mode = debug_toggle
#         st.rerun()

# 디버그 정보 표시 함수
def debug_info(message, level="info", pin=False):
    """디버그 모드일 때만 표시 + 핀 메시지는 입력창 위에 고정"""
    if st.session_state.debug_mode:
        if pin:
            st.session_state.debug_pinned_message = message  # ✅ 고정 메시지로 등록
        if level == "info":
            st.info(message)
        elif level == "warning":
            st.warning(message)
        elif level == "error":
            st.error(message)
        elif level == "success":
            st.success(message)
        else:
            st.write(message)


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
                "- 사용자 질문에 \"추천\", \"제시\", \"찾아\", \"검색해\" 라는 단어가 포함되면 반드시 목록 형식으로 작성해주세요.\n"
                " 만일 그런 경우가 아니라면, 서술식일 경우 자연스럽고 포괄적인 설명으로 구성해 주세요. 마크다운 스타일(**, ## 등)은 사용하지 마세요.\n"
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


# ✅ 유틸리티 함수들
def normalize(vecs):
    norms = np.linalg.norm(vecs, axis=1, keepdims=True)
    return vecs / norms

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

# 기업 ID를 키로, 기업명을 값으로 하는 딕셔너리 생성
@st.cache_resource
def create_company_lookup():
    company_dict = {}
    for item in metadata:
        if "기업ID" in item and "기업명" in item:
            company_dict[str(item["기업ID"])] = item["기업명"]
    return company_dict

# ⚠️ 이 호출은 함수 정의 후에 배치
index, metadata, index_cosine = load_index_and_metadata()
company_lookup = create_company_lookup()


def get_embedding(text, model="text-embedding-3-small"):
    if text in st.session_state.embedding_cache:
        return st.session_state.embedding_cache[text]

    if USE_OPENAI_EMBEDDING:
        response = client.embeddings.create(input=[text], model=model)
        embedding = response.data[0].embedding  # 수정된 부분: 딕셔너리 접근이 아닌 객체 속성 접근
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
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=messages
        )
        answer = response.choices[0].message.content.strip().lower() 
        result = "yes" in answer  # 'yes' 포함 여부로 판단
        st.session_state.followup_cache[key] = result  # ✅ 캐시 저장
        return result
    except Exception as e:
        return True  # 오류 시 기본은 후속 질문으로 간주

def is_best_recommendation_query(query):
    keywords = ["강력 추천", "강추"]
    return any(k in query for k in keywords)

def is_relevant_question(query, threshold=Q_SIMILARITY_THRESHOLD):
    query_vec = get_embedding(query)
    query_vec = np.array(query_vec).astype('float32').reshape(1, -1)
    query_vec = normalize(query_vec)
    D, _ = index_cosine.search(query_vec, 1)
    max_similarity = D[0][0]
    return max_similarity >= threshold

def is_related_results_enough(ranked_results, threshold=A_SIMILARITY_THRESHOLD, top_n=MAX_HISTORY_LEN):
    """
    벡터 유사도 기반 추천 결과 중 상위 N개의 평균 유사도가 threshold 이상인지 확인.
    관련도가 낮으면 False 반환 → GPT 호출 방지 가능.
    """
    if not ranked_results or len(ranked_results) < top_n:
        return False
    top_scores = [score for score, _ in ranked_results[:top_n]]
    avg_score = sum(top_scores) / len(top_scores)
    debug_info(f"📊 상위 {top_n}개 평균 유사도: {avg_score:.4f}", pin=True)
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
        debug_info("⚠️ [INFO] 추천 결과의 연관성이 낮아 GPT 호출을 생략합니다.", "warning")
        return []
    
    # 📌 STEP 1: 유사도 기준 정렬된 원본 상위 30개 출력
    debug_info(f"\n📌 [STEP 1] 유사도 기준 정렬된 원본 상위 30개:")
    for i, (score, s) in enumerate(ranked[:30]):
        debug_info(f"{i+1}. [{score:.4f}] {s['기업명']} / {s.get('서비스유형')} / {s.get('서비스명')}")
        
    # ✅ 4. 제외할 키 (기업ID + 서비스유형 + 서비스명) 정의
    if exclude_keys:
        debug_info(f"\n\n🚫 [STEP 2] 제외 대상 키 수: {len(exclude_keys)}")
        for i, key in enumerate(list(exclude_keys)[:10]):
            company_name = company_lookup.get(str(key[0]), "알 수 없음")
            debug_info(f" - 제외 {i+1}: 기업ID={key[0]} / 기업명={company_name} / {key[1]} / {key[2]}")
    else:
        debug_info("\n\n🚫 [STEP 2] 제외 대상 없음")

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
    
    # ✅ 상위 30개까지 출력 (디버깅 또는 로그 확인용)
    debug_info(f"\n✅ [STEP 3] 필터링 후 상위 30개:")
    for i, (score, s) in enumerate(filtered[:30]):
        debug_info(f"{i+1}. [{score:.4f}] {s['기업명']} / {s.get('서비스유형')} / {s.get('서비스명')}")

    # 6. 상위 10개 중 랜덤 선택 or top_k개 선택
    if use_random:
        top_10 = filtered[:10]
        selected = random.sample(top_10, min(len(top_10), top_k))
        results = [service for _, service in selected]
    else:
        results = [service for _, service in filtered[:top_k]]

    return results

def ask_gpt(messages):
    response = client.chat.completions.create(model="gpt-4o", messages=messages)
    return response.choices[0].message.content

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

if st.session_state.debug_mode and "debug_pinned_message" in st.session_state:
    st.markdown(f"""
        <div style="background-color:#fff3cd; border-left: 6px solid #ffeeba; padding:10px; margin-bottom:10px;">
            "{st.session_state.debug_pinned_message}"
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
        •&nbsp;<b>"강력 추천"</b> 을 포함하여 입력하면 앞서 제시된 서비스들을 포함한 전체 추천을 받아볼 수 있어요.<br>
        •&nbsp;<b>"초기화"</b>를 입력하면 대화 내용과 추천 기록을 모두 지울 수 있어요.<br>
        •&nbsp;<b>"디버그"</b>를 입력하면 디버그 모드로 전환할 수 있어요.<br>
        •&nbsp;복합적인 조건들을 이용한 질문으로 편리하게 사용해 보세요.<br>
        예를들어 "우리 회사는 외국인에게 국내 유명 관광지역을 소개하고 숙박을 연결해주는 서비스를 하고 있어. 회사 홈페이지를 디자인 중심으로 개편하고 싶고, 참 다국어는 필수고, 숙박지를 예약하고 결제하는 쇼핑몰 기능이 반드시 필요해. 또한 인스타그램으로 홍보도 잘 하는 것도 필수고. 이런걸 만족시킬 수 있는 조합을 만들어줘. 단, 예산은 합쳐서 5,000만원까지이고, 기간은 3.5개월안에는 마쳐야 해. 많은 소통을 위해 가급적 수도권 지역에 있는 회사였으면 좋겠고, 매출도 30억 이상되며 인원도 많아서 안정적인 지원도 받았으면 하고. 이런 회사들로 찾아봐줘. 또 어떻게 이들을 조합하면 되는지, 왜 추천했는지도 상세히 알려줘."
    </div>
""", unsafe_allow_html=True)

# 메시지 처리 로직
if submitted and user_input.strip():
    # 시간대 설정
    current_time = get_kst_time()
    
    # 사용자 메시지 저장
    st.session_state.chat_messages.append({"role": "user", "content": user_input, "timestamp": current_time})
    
    # 디버그 모드 토글 명령 처리
    if user_input.strip().lower() == "디버그":
        st.session_state.debug_mode = not st.session_state.debug_mode
        mode_status = "활성화" if st.session_state.debug_mode else "비활성화"
        st.session_state.chat_messages.append({
            "role": "assistant", 
            "content": f"🛠️ 디버그 모드가 {mode_status}되었습니다.", 
            "timestamp": current_time
        })
        st.rerun()
    
    # 초기화 명령 처리
    elif user_input.strip().lower() == "초기화":
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
    elif user_input.strip().startswith("자세히"):
        keyword = user_input.replace("자세히", "").strip()
        all_stored_results = list(itertools.chain.from_iterable(st.session_state.all_results))
        
        if not all_stored_results:
            reply = "⚠️ 저장된 추천 내역이 없습니다."
        else:
            keywords = keyword.split()
            matches = [
                s for s in all_stored_results
                if all(any(k in s.get(field, "") for field in ["기업명", "서비스명"]) for k in keywords)
            ]
            
            if not matches:
                reply = "⚠️ 해당 키워드를 포함한 기업명이 없습니다."
            elif len(matches) > 1:
                reply = "⚠️ 여러 개의 항목이 일치합니다. 더 구체적으로 입력해 주세요.\n" + "\n".join([f"- {s['기업명']} / {s['서비스명']}" for s in matches])
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
                    details.append(f"•&nbsp;{k}: {v}")
                
                # 링크 버튼 형식으로 변경
                links = [
                    f'<a href="{service_link}" target="_blank" class="link-button">🔗 서비스 상세 보기</a>',
                    f'<a href="{company_link}" target="_blank" class="link-button">🏢 기업 정보 보기</a>'
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
        
        debug_info("\n🤖 호종이가 질문을 분석 중입니다...")
        # 질문 관련성 확인
        if not is_relevant_question(user_input):
            reply = "⚠️ 죄송하지만, 질문의 내용을 조금 더 관광기업이나 서비스와 관련된 내용으로 다시 해 주세요."
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
                debug_info("➡️ [INFO] 독립된 질문입니다. 기준 임베딩 갱신.")
                st.session_state.embedding_query_text = user_input
            else:
                # 후속 질문이면 이전 임베딩 유지
                debug_info("➡️ [INFO] 후속 질문입니다. 기준 임베딩 유지.")
        else:
            # 최초 질문인 경우
            debug_info("➡️ [INFO] 최초 질문입니다. 기준 임베딩 설정.")
            st.session_state.embedding_query_text = user_input
        
        # 질문 히스토리에 추가
        st.session_state.user_query_history.append(user_input)
        
        debug_info("🤖 호종이가 관련 서비스를 찾는 중입니다...")
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
            reply = "⚠️ 추천 결과가 충분하지 않아 관련된 업체나 서비스를 제공드리기가 어렵습니다. n관광기업이나 서비스와 관련된 질문을 조금 더 구체적으로 해 주시면 감사하겠습니다!"
            st.session_state.chat_messages.append({
                "role": "assistant", 
                "content": reply, 
                "timestamp": current_time
            })
            st.rerun()
        
        debug_info("🤖 호종이가 추천 내용을 정리 중입니다...")
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
        
        debug_info(f"[❗DEBUG] GPT 응답에서 언급된 키: {mentioned_keys}")
        for s in last_results:
            기업ID = s.get('기업ID')
            기업명 = s.get('기업명')
            서비스명 = s.get('서비스명')
            debug_info(f"🔍 비교 중: {기업ID} / {기업명} / {서비스명}")
            debug_info(f"    → 기업ID 비교: {기업ID} in GPT? {'YES' if str(기업ID) in gpt_reply else 'NO'}")
            debug_info(f"    → 기업명 비교: {기업명} in GPT? {'YES' if 기업명 in gpt_reply else 'NO'}")
            debug_info(f"    → 서비스명 비교: {서비스명} in GPT? {'YES' if 서비스명 in gpt_reply else 'NO'}")
        debug_info(f"\n🔍 excluded_keys 키 수: {len(st.session_state.excluded_keys)}")
        
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