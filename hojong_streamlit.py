
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
import time
import json

# ✅ 진입 암호 입력 로직 (4자리 숫자 예: 7299)
if "authenticated" not in st.session_state:
    st.session_state.authenticated = False

if not st.session_state.authenticated:
    st.markdown("## 🔐 접근 권한")
    password_input = st.text_input("4자리 숫자 비밀번호를 입력하세요:", type="password")
    if password_input and password_input.strip() == "7299":
        st.session_state.authenticated = True
        st.rerun()
    elif password_input:
        st.error("❌ 비밀번호가 틀렸습니다.")
    st.stop()

# ✅ 인증 이후 실행
st.set_page_config(layout="wide")

import openai
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
            border-radius: 0px 12px 12px 12px !important; 
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
        .info-msg {
            background-color: #fff3cd !important; 
            font-size: 14px !important;
            border-left: 6px solid #ffeeba !important;
            border-radius: 6px 6px 6px 6px !important; 
            padding: 10px !important;
            margin-bottom: 10px !important;
            text-align: center !important;
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
                margin: 0 0 15px 0 !important; 
            }
            .user-msg-time, chatbot-msg-time {
                font-size: 9x;
            }   
            .user-guide {
                font-size: 11px !important;
                line-height: 1.3 !important;
            }
            .info-msg {
                font-size: 11px !important;
                padding: 5px !important;
                margin-bottom: 5px !important;
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
FALLBACK_ATTEMPT_NUM = 2

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
        
    """디버그 모드일 때만 표시, 핀 메시지는 입력창 위에 고정"""
    if st.session_state.debug_mode:        
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
    if pin:
        st.session_state.debug_pinned_message = message  # ✅ 고정 메시지로 등록

def pause_here(message="⏸️ 디버깅 지점입니다. 계속하려면 버튼을 누르세요."):
    if "pause_continue" not in st.session_state:
        st.session_state.pause_continue = False

    if not st.session_state.pause_continue:
        st.warning(message)
        if st.button("👉 계속 실행하기", key=f"btn_{len(st.session_state.chat_messages)}"):
            st.session_state.pause_continue = True
            st.rerun()
        else:
            st.stop()
            
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
                "[주의: 모든 답변은 아래 지침을 따라야 합니다]\n\n"
                "- 답변은 반드시 사용자 질문의 요약으로 시작해 주세요.\n"
                "- 반드시 아래 형식으로 시작해 주세요: '질문 요약: \"...\"'\n"
                "- 이후 이어서 본문을 작성해 주세요.\n"
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
                "- 반드시 위 형식을 지켜주세요.\n"
                "- 절대로 마크다운 스타일(예: **굵은 글씨**, *기울임*, ## 제목 등)을 사용하지 마세요."
                "- 예) 잘못된 예: **\"삼성전자\"**, ##\"홈페이지 서비스\"##\n"
                "- 예) 올바른 예: \"삼성전자\", \"홈페이지 서비스\"\n"
                "- 불릿은 항상 대시(-)만 사용해 주세요.\n"
                "- 항목 간 개행을 넣지 마세요.\n"
                "- 기업명/서비스명이 언급되면 반드시 이유도 함께 제시해 주세요.\n"
                "- 기업ID는 반드시 괄호 안에 표기해 주세요. 예: \"삼성전자\" (기업ID: 12345)\n"
                "- 위 지침들을 어기면 시스템 오류로 간주됩니다."
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
if "embedding_query_text_summary" not in st.session_state:
    st.session_state.embedding_query_text_summary = None
if "embedding_query_vector" not in st.session_state:
    st.session_state.embedding_query_vector = None  # 벡터 캐싱 초기화
if "pending_fallback" not in st.session_state:
    st.session_state.pending_fallback = False
if "fallback_attempt" not in st.session_state:
    st.session_state.fallback_attempt = 0
if "A_SIMILARITY_THRESHOLD" not in st.session_state:
    st.session_state.A_SIMILARITY_THRESHOLD = A_SIMILARITY_THRESHOLD  # 기본값 사용
if "TOP_N" not in st.session_state:
    st.session_state.TOP_N = MAX_HISTORY_LEN
if "is_processing" not in st.session_state:
    st.session_state.is_processing = False    


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
        {"role": "system", "content": """당신은 AI 질문 분석가입니다.
                다음에 제시된 두 질문을 비교해서, 두 번째 질문이 첫 번째 질문의 응답을 전제로 한 후속 질문인지 판단해 주세요.\n
                - 후속 질문이란 이전 질문에 대한 **추가 요청**, **관련 조건의 확대**, **구체화**, **계속된 탐색** 등이 포함된 경우입니다.\n
                - 단순히 유사하거나 키워드가 같은 것이 아니라 **전제 맥락 없이 의미가 부족한 문장**도 후속 질문으로 간주됩니다.\n
                - yes 또는 no로만 답해 주세요."""},
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
    
    
# ✅ 요약 함수 추가 (GPT 사용)
def summarize_query(query):
    """
    긴 사용자 질문을 유사도 임베딩에 적합하도록 요약
    """
    prompt = f"""사용자의 질문이 다음과 같습니다:\n\n{query}\n\n
                이 질문을 벡터 임베딩에 적합하도록 핵심 키워드 중심으로 요약해 주세요. 
                불필요한 서사나 예시는 제거하고, 핵심 목적/조건/희망사항만 정리해 주세요.
                출력은 (질문의 총 길이를 100으로 나눈 수)만큼의 문장 수로 작성 해주세요."""
    
    try:
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": "당신은 AI 질의 요약 도우미입니다."},
                {"role": "user", "content": prompt}
            ]
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        debug_info(f"❌ 요약 실패: {str(e)}", level="error")
        return query  # 실패하면 원본 사용

# ✅ get_embedding 호출 전 요약 처리 (벡터 검색 이전)
def get_embedding_with_optional_summary(text, model="text-embedding-3-small"):
    # 너무 긴 경우만 요약
    if len(text) > 150:
        debug_info("📌 질문이 길어 GPT로 요약 후 벡터화합니다.", pin=True)
        text = summarize_query(text)
        st.session_state.embedding_query_text_summary = text
        debug_info(f"📌 gpt 요약: " + text)
    return get_embedding(text, model)

def is_best_recommendation_query(query):
    keywords = ["강력 추천", "강추"]
    return any(k in query for k in keywords)

def is_relevant_question(query, threshold=Q_SIMILARITY_THRESHOLD): 
    query_vec = get_embedding_with_optional_summary(query)
    st.session_state.embedding_query_vector = query_vec
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
    threshold = threshold or st.session_state.A_SIMILARITY_THRESHOLD
    top_n = top_n or st.session_state.TOP_N
    debug_info(f"📌 threshold : " + str(threshold))
    debug_info(f"📌 top_n : " + str(top_n))
    if not ranked_results or len(ranked_results) < top_n:
        debug_info(f"📌 관련도가 낮으면 False 반환 → GPT 호출 방지 가능.")
        return False
    top_scores = [score for score, _ in ranked_results[:top_n]]
    avg_score = sum(top_scores) / len(top_scores)
    debug_info(f"🤖 분석 결과 상위 {top_n}개 평균 유사도는 {avg_score:.4f} 입니다.", pin=True)
    
    return avg_score >= threshold

def recommend_services(query, top_k=5, exclude_keys=None, use_random=True):
    # ✅ exclude_keys가 None이면 빈 집합으로 초기화
    if exclude_keys is None:
        exclude_keys = set()
    
    # 벡터를 재사용하려 했으나, 하기 이유 발생으로 재사용 안함.
    # 이전 질문: 홈페이지 구축 업체 알려줘 → 벡터 A
    # 지금 질문: 디자인 업체도 알려줘 → 벡터 A 그대로 사용
    # → "디자인"이 강조되어야 할 텍스트에 "홈페이지" 벡터를 쓰게 됨
    # if "embedding_query_vector" in st.session_state and st.session_state.embedding_query_vector is not None:
    #     debug_info(f"\n📌 이전 생성된 벡터 재사용")
    #     query_vec = st.session_state.embedding_query_vector
    # else:
    #     debug_info(f"\n📌 새로운 쿼리를 기준으로 요약 후 새로운 벡터 생성")

    query_vec = get_embedding_with_optional_summary(query)
    st.session_state.embedding_query_vector = query_vec
    query_vec = np.array(query_vec).astype('float32').reshape(1, -1)
    query_vec = normalize(query_vec)

    # ✅ 2. 유사한 서비스 300개 검색 (기존 100개 → 확장)
    D, indices = index_cosine.search(query_vec, 300)

    # 3. 유사도 높은 순서로 (score, service) 목록 생성
    ranked = [(score, metadata[idx]) for score, idx in zip(D[0], indices[0])]
    
    # 📌 STEP 1: 유사도 기준 정렬된 원본 상위 30개 출력
    debug_info(f"\n📌 [STEP 1] 유사도 기준 정렬된 원본 상위 30개:")
    for i, (score, s) in enumerate(ranked[:30]):
        debug_info(f"{i+1}. [{score:.4f}] {s['기업명']} / {s.get('서비스유형')} / {s.get('서비스명')}")
    
    # ⛔ 유사도 낮을 경우 GPT 호출도 생략할 수 있도록 빈 리스트 반환
    debug_info(f"✅ 파라미터 조정되었는지 확인: 임계값={st.session_state.A_SIMILARITY_THRESHOLD}, TOP_N={st.session_state.TOP_N}", "success")
    if not is_related_results_enough(ranked, st.session_state.A_SIMILARITY_THRESHOLD, st.session_state.TOP_N):
        debug_info("📌 추천 결과의 연관성이 낮아 fallback 루프로 진입합니다.", "warning")
        st.session_state.pending_fallback = True
        return []
    

        
    # ✅ 4. 제외할 키 (기업ID + 서비스유형 + 서비스명) 정의
    if exclude_keys:
        debug_info(f"\n\n📌 [STEP 2] 제외 대상 키 수: {len(exclude_keys)}")
        for i, key in enumerate(list(exclude_keys)[:10]):
            company_name = company_lookup.get(str(key[0]), "알 수 없음")
            debug_info(f" - 제외 {i+1}: 기업ID={key[0]} / 기업명={company_name} / {key[1]} / {key[2]}")
    else:
        debug_info("\n\n📌 [STEP 2] 제외 대상 없음")

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
    debug_info(f"\n📌 [STEP 3] 필터링 후 상위 30개:")
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
    summarized = st.session_state.embedding_query_text_summary or query
    if is_best:
        history = make_summary_context(st.session_state.all_results)
        extra = f"지금까지 추천한 서비스 목록은 다음과 같습니다:\n\n{history}\n\n이전에 추천된 기업도 포함해서 조건에 가장 부합하는 최고의 조합을 제시해주세요."
    else:
        extra = ""
        
    return f"""[주의: 아래 지침에 따라 절대 마크다운이나 강조 표시 없이 일반 텍스트로만 응답하세요.]

당신은 관광수혜기업에게 추천 서비스를 제공하는 AI 상담사입니다.

질문 요약: "{summarized}" ← 반드시 이 문장으로 답변을 시작해 주세요.

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

# if st.session_state.get("is_processing", False):
if "debug_pinned_message" in st.session_state:
    st.markdown(f"""
        <div class="info-msg">
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
        •&nbsp;<b>"자세히 기업명 서비스명"</b>을 입력하면 해당 상세 정보를 확인할 수 있어요. (기업명과 서비스명은 단어 일부만 입력하셔도 되요.)<br>
        •&nbsp;<b>"강력 추천"</b> 을 포함하여 입력하면 앞서 제시된 서비스들을 포함한 전체 추천을 받아볼 수 있어요.<br>
        •&nbsp;<b>"초기화"</b>를 입력하면 대화 내용과 추천 기록을 모두 지울 수 있어요.<br>
        •&nbsp;<b>"복합적인 조건들"</b>을 이용한 질문이나 <b>"의미적으로 유사한 정보"</b> 검색으로 편리하게 사용해 보세요.<br>
        - 예를들어 "우리 회사는 외국인에게 국내 유명 관광지역을 소개하고 숙박을 연결해주는 서비스를 하고 있어. 회사 홈페이지를 디자인 중심으로 개편하고 싶고, 참 다국어는 필수고, 숙박지를 예약하고 결제하는 쇼핑몰 기능이 반드시 필요해. 또한 인스타그램으로 홍보도 잘 하는 것도 필수고. 이런걸 만족시킬 수 있는 조합을 만들어줘. 단, 예산은 합쳐서 5,000만원까지이고, 기간은 3.5개월안에는 마쳐야 해. 많은 소통을 위해 가급적 수도권 지역에 있는 회사였으면 좋겠고, 매출도 30억 이상되며 인원도 많아서 안정적인 지원도 받았으면 하고. 이런 회사들로 찾아봐줘. 또 어떻게 이들을 조합하면 되는지, 왜 추천했는지도 상세히 알려줘.<br>
        - 또는 "삼성전자와 유사한 서비스를 제공하는 다른 기업을 추천해줘." 
    </div>
""", unsafe_allow_html=True)


if st.session_state.get("is_processing", False):
    user_input = st.session_state.pending_input
    del st.session_state.pending_input
    submitted = True
    # 의도적인 딜레이 (벡터에서만 검색해서 너무 빠르니 찾아보는거 같지 않아서)
    if st.session_state.pending_fallback:
        time.sleep(1)

# 메시지 처리 로직
if submitted and user_input.strip():
    
    # 시간대 설정
    current_time = get_kst_time()
    
    if not st.session_state.get("is_processing", False):
        # 사용자 메시지 아예 여기서 저장해버림 (그래야 버튼 누르고 바로 보여줄 수 있음.)
        st.session_state.chat_messages.append({"role": "user", "content": user_input, "timestamp": current_time})
        
        st.session_state.pending_input = user_input
        st.session_state.is_processing = True  # 분석 중 상태 True 설정
        debug_info("🤖 잠시만 기다려 주세요. 최적의 답변을 준비 중입니다...", pin=True)
        st.rerun()
    else:
        debug_info("🤖 잠시만 기다려 주세요. 최적의 답변을 준비 중입니다...", pin=True)
        st.session_state.is_processing = False  
    
    # ✅ fallback 상황인지 먼저 체크하고, 사용자 입력을 아직 저장하지 않음
    if st.session_state.pending_fallback:
        debug_info("📚 fallback 상태 감지됨 : " + str(st.session_state.fallback_attempt), "success")
        
        if user_input.strip().lower() == "네" and st.session_state.fallback_attempt < FALLBACK_ATTEMPT_NUM:
            # 파라미터 조정
            st.session_state.fallback_attempt += 1
            st.session_state.A_SIMILARITY_THRESHOLD = max(0.1, st.session_state.A_SIMILARITY_THRESHOLD - 0.03)
            st.session_state.TOP_N = max(2, st.session_state.TOP_N - 1)
            
            # 이제 사용자 입력 저장
            # st.session_state.chat_messages.append({
            #     "role": "user",
            #     "content": user_input,
            #     "timestamp": current_time
            # })
            
            debug_info(f"📚 파라미터 조정됨: 임계값={st.session_state.A_SIMILARITY_THRESHOLD}, TOP_N={st.session_state.TOP_N}", "success")
            
            # 이전 질문으로 기준 임베딩 복원
            #if st.session_state.user_query_history:
            #    st.session_state.embedding_query_text += ("," + st.session_state.user_query_history[-1])
            
            debug_info(f"📚 embedding_query_text : " + str(st.session_state.embedding_query_text))
            # pause_here("🧪 001 last_results : " + str(st.session_state.embedding_query_text))
            
            # 검색 로직 직접 실행
            best_mode = is_best_recommendation_query(st.session_state.embedding_query_text)
            
            # pause_here("🧪 002 best_mode : " + str(best_mode))
            
            exclude = None if best_mode else st.session_state.excluded_keys
            last_results = recommend_services(
                st.session_state.embedding_query_text,
                exclude_keys=exclude,
                use_random=not best_mode
            )
            
            # pause_here("🧪 003 last_results : " + str(last_results))
            
            # 결과 처리
            if not last_results:
                # 여전히 결과가 없음 - 다시 fallback 상태로
                st.session_state.pending_fallback = True
                if st.session_state.fallback_attempt == 1:
                    reply = "⚠️ 조금 더 포괄적인 범위로 다시 찾아보겠습니다. 진행을 원하시면 '네' 라고 답해주세요."
                elif st.session_state.fallback_attempt == 2:
                    reply = "⚠️ 여전히 서비스를 찾기 어렵습니다. 마지막으로 더 찾아보겠습니다. 진행을 원하시면 '네' 라고 답해주세요."
                else:
                    reply = "⚠️ 정보 제공이 불가능하여 죄송합니다. 다른 질문을 입력해 주시면 다시 찾아보겠습니다."        
                st.session_state.chat_messages.append({
                    "role": "assistant", 
                    "content": reply, 
                    "timestamp": current_time
                })
                debug_info(f"🤖 ...", pin=True)
                # pause_here("🧪 004-1 last_results is null")/
                # st.rerun()
            else:
                
                # pause_here("🧪 004-2 last_results is not null") 
                
                # 결과 찾음 - 처리 진행
                context = make_context(last_results)
                gpt_prompt = make_prompt(st.session_state.embedding_query_text, context, is_best=best_mode)
                
                # GPT 호출
                st.session_state.conversation_history.append({"role": "user", "content": gpt_prompt})
                try:
                    gpt_reply = ask_gpt(list(st.session_state.conversation_history))
                    # pause_here("🧪 005-1 gpt_reply : " + gpt_reply)
                    
                except Exception as e:
                    gpt_reply = f"⚠️ 응답 생성 중 오류가 발생했습니다. 다시 시도해주세요: {str(e)}"
                    # pause_here("🧪 005-2 gpt_reply is error! ")
                    
                # 응답 저장
                st.session_state.conversation_history.append({"role": "assistant", "content": gpt_reply})
                
                # 결과 처리
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
                
                # 결과 저장
                st.session_state.last_results = last_results
                st.session_state.all_results.append(last_results)
                
                # 챗봇 응답 추가
                st.session_state.chat_messages.append({
                    "role": "assistant", 
                    "content": gpt_reply, 
                    "timestamp": current_time
                })
                # fallback 상태 초기화
                st.session_state.pending_fallback = False
                st.session_state.fallback_attempt = 0
                st.session_state.A_SIMILARITY_THRESHOLD = A_SIMILARITY_THRESHOLD
                st.session_state.TOP_N = MAX_HISTORY_LEN
                st.session_state.user_query_history = []
                
                # st.rerun()  # 화면 업데이트

        else:
            # fallback 취소
            reply = "⛔ 죄송합니다만 검색이 취소되었습니다. 다른 질문을 입력해 주시면 다시 찾아보겠습니다."
            # st.session_state.chat_messages.append({
            #     "role": "user",
            #     "content": user_input,
            #     "timestamp": current_time
            # })
            st.session_state.chat_messages.append({
                "role": "assistant",
                "content": reply,
                "timestamp": current_time
            })
            # fallback 상태 초기화
            st.session_state.pending_fallback = False
            st.session_state.fallback_attempt = 0
            st.session_state.A_SIMILARITY_THRESHOLD = A_SIMILARITY_THRESHOLD
            st.session_state.TOP_N = MAX_HISTORY_LEN
            st.session_state.user_query_history = []
            st.session_state.embedding_query_text = None
            st.session_state.embedding_query_text_summary = None
            st.session_state.embedding_query_vector = None  # 벡터 캐싱 초기화
            # st.rerun()
        
        if st.session_state.debug_mode:
            pause_here()
        else: 
            st.rerun()

    # 사용자 메시지 저장
    # st.session_state.chat_messages.append({"role": "user", "content": user_input, "timestamp": current_time})
    
    # 디버그 모드 토글 명령 처리
    if user_input.strip().lower() == "디버그":
        st.session_state.debug_mode = not st.session_state.debug_mode
        mode_status = "활성화" if st.session_state.debug_mode else "비활성화"
        st.session_state.chat_messages.append({
            "role": "assistant", 
            "content": f"🛠️ 디버그 모드가 {mode_status}되었습니다.", 
            "timestamp": current_time
        })
        debug_info(f"🤖 앗! 비밀이 해제되었습니다. 유용하게 사용하세요.", pin=True)
        st.rerun()
    
    # 초기화 명령 처리
    elif user_input.strip().lower() == "초기화":
    
        st.session_state.is_processing = False
        st.session_state.pending_fallback = False
        st.session_state.fallback_attempt = 0
        st.session_state.A_SIMILARITY_THRESHOLD = A_SIMILARITY_THRESHOLD  # 기본값 사용
        st.session_state.TOP_N = MAX_HISTORY_LEN
        st.session_state.embedding_cache = {}
        st.session_state.followup_cache = {}        
        st.session_state.embedding_query_text = None
        st.session_state.embedding_query_text_summary = None
        st.session_state.embedding_query_vector = None  # 벡터 캐싱 초기화
        st.session_state.excluded_keys.clear()
        st.session_state.all_results.clear()
        st.session_state.last_results = []
        st.session_state.user_query_history = []
        st.session_state.conversation_history.clear()
        st.session_state.conversation_history.append({
            "role": "system", 
            "content": "당신은 관광기업 상담 전문가 AI입니다."
        })

        # 초기화 응답 메시지 추가
        st.session_state.chat_messages = []
        st.session_state.chat_messages.append({
            "role": "assistant", 
            "content": "🤖 잠시 머리 좀 비우고 다시 돌아왔습니다.", 
            "timestamp": current_time
        })
        debug_info(f"🤖 자~ 이제 다시 관광기업 서비스에 대해 물어보세요.", pin=True)
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
                reply = "⚠️ 여러 개의 유사 항목이 일치합니다. 더 구체적으로 입력해 주세요.\n" + "\n".join([f"- {s['기업명']} : {s['서비스명']}" for s in matches])
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
        debug_info(f"🤖 링크를 누르시면 관광공사 홈페이지 기업 및 서비스 화면으로 이동합니다.", pin=True)
        st.rerun()
    
    # 일반 질문 처리
    else:
    
        # 대화 이력에 사용자 입력 추가
        st.session_state.conversation_history.append({"role": "user", "content": user_input})
        
        # 지금 질문한 내용이 사업과 관련이 없어.
        if not is_relevant_question(user_input):
            
             # 지금한 질문이 사업과 관련없고, 최초 대화가 아닌 경우 경우임. 단, 지금한 질문은 사업과 관련성이 없어도 이전 대화와의 연계성을 검토
            if st.session_state.user_query_history:
                previous_input = st.session_state.user_query_history[-1]
                
                # 지금한 질문이 사업과 관련없고, 이전과 지금이 서로 진짜 관련없는 상황임 --> 에러!
                if not is_followup_question(previous_input, user_input):
                    debug_info("📚 1. 지금 질문한 내용이 사업과 관련이 없어. 더구나 이전한 얘기와도 연계가 없어.")
                    st.session_state.user_query_history.append(user_input)
                    reply = "⚠️ 죄송하지만, 질문의 내용을 조금 더 관광기업이나 서비스와 관련된 내용으로 다시 해 주세요."
                    st.session_state.chat_messages.append({
                        "role": "assistant", 
                        "content": reply, 
                        "timestamp": current_time
                    })
                    st.rerun()
                # 지금한 질문이 사업과 관련없으나, 이전과 후속 대화인 경우임. --> 후속 대화로 인지 --> 단, 검색 시 내용은 이전걸로 넣어줘야 해.
                # ex. 이전 : 홈페이지 구축 업체 알려줘.
                #     지금 : 다른 사례는 없어? 
                else:
                    debug_info("📚 2. 지금 질문한 내용이 사업과 관련이 없어. 하지만 지금 얘기한건 이전에 얘기와는 연관되어 있어.")
                    st.session_state.embedding_query_text = "[이전 질문 : ]" + previous_input + "\n[지금 질문 : ]" + user_input
                    
            # 지금한 질문이 사업과 관련없고, 최초 대화인 경우임 또는 Fallback 후 초기화 된 이후임. --> 에러!
            else: 
                debug_info("📚 3. 지금 질문한 내용이 사업과 관련이 없어. 하지만 최초부터 이런 관련없는 얘기하면 안되는거야.")
                st.session_state.user_query_history.append(user_input)
                reply = "⚠️ 죄송하지만, 질문의 내용을 조금 더 관광기업이나 서비스와 관련된 내용으로 다시 해 주세요."
                st.session_state.chat_messages.append({
                    "role": "assistant", 
                    "content": reply, 
                    "timestamp": current_time
                })
                st.rerun()
                
        # 지금한 질문한 내용이 사업과 관련이 있음.
        else:
            
            # 지금한 질문이 사업과 관련있고, 최초 대화가 아닌 경우임. 다만 지금한 질문이 이전과 관련성을 검토한다.
            if st.session_state.user_query_history:
                
                previous_input = st.session_state.user_query_history[-1]
                # 지금한 질문이 사업과 관련 있으나, 이전과 지금이 서로 연속성은 없는 경우. --> 신규 대화로 전환     
                # ex. 이전 : 홈페이지 구축 업체 알려줘.
                #     지금 : 아니야. 디자인 홍보 업체를 새로 알려줘. 
                if not is_followup_question(previous_input, user_input):
                    debug_info("📚 4. 지금 질문한 내용이 사업과는 관련있지만, 앞에서 얘기한 사업이랑은 전혀 관련이 없어. 하지만 새롭게 다른 사업과 관련있는 얘기하면 좋은거야.")
                    # st.session_state.embedding_query_text = user_input
                    st.session_state.embedding_query_text = "[이전 질문 : ]" + previous_input + "\n[지금 질문 : ]" + user_input
                    
                # 지금한 질문이 사업과 관련 있고, 이전의 대화와고 관련이 있음.   --> 후속 대화로 인지    
                # ex. 이전 : 홈페이지 구축 업체 알려줘.
                #     지금 : 홈페이지 구축 업체를 추가로 알려줘.
                else:
                    debug_info("📚 5. 지금 질문한 내용이 사업과는 관련도 있고, 앞에서 얘기한 사업과 관련이 있어. 그리고 지금 얘기한 것도 구체적으로 사업과 관련이 되어 있어.")
                    st.session_state.embedding_query_text = "[이전 질문 : ]" + previous_input + "\n[지금 질문 : ]" + user_input
                    
            # 지금한 질문이 사업과 관련있고, 최초 대화한 경우 또는 Fallback 후 초기화 된 이후임. --> 신규 대화로 인지
            else:
                debug_info("📚 6. 지금 질문한 내용이 사업과 관련이 있어. 그리고 지금 최초로 얘기한 것도 사업과 관련이 되어 있어.")
                st.session_state.embedding_query_text = user_input
                
    
        
    
    
        # # 후속 질문 판단
        # if st.session_state.user_query_history:
        #     previous_input = st.session_state.user_query_history[-1]
        #     if not is_followup_question(previous_input, user_input):
        #         debug_info("🤖 신규 질문으로 인식하고 관련 서비스를 찾는 중입니다...", pin=False)
        #         st.session_state.embedding_query_text = user_input
        #     else:
        #         # 후속 질문이면 이전 임베딩 유지
        #         debug_info("🤖 후속 질문으로 인식하고 관련 서비스를 찾는 중입니다...", pin=False)
        # else:    
        #     # 최초 질문인 경우
        #     debug_info("🤖 최초 질문으로 인식하고 관련 서비스를 찾는 중입니다...", pin=False)
        #     st.session_state.embedding_query_text = user_input
        
        
        
        # # 질문 히스토리에 추가
        st.session_state.user_query_history.append(user_input)
        debug_info("📚 embedding_query_text = " + st.session_state.embedding_query_text, pin=False)
        
        debug_info("🤖 관련 서비스를 찾는 중입니다...", pin=False)
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
            st.session_state.pending_fallback = True
            reply = "⚠️ 질문 의도 파악이 쉽지 않거나 원하시는 추천 결과가 충분하지 않아 관련된 업체나 서비스를 제공드리기가 어렵스럽습니다. 한 번 더 진행을 원하시면 '네' 라고 답해주세요."
            st.session_state.chat_messages.append({
                "role": "assistant", 
                "content": reply, 
                "timestamp": current_time
            })
            debug_info(f"🤖 ...", pin=True)
            if st.session_state.debug_mode:
                pause_here()
            else: 
                st.rerun()
        
        debug_info("🤖 추천 내용을 정리 중입니다...", pin=False)
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
                
        if st.session_state.debug_mode:
            if "embedding_query_text" in st.session_state:
                debug_info("📚 embedding_query_text =\n" + st.session_state.embedding_query_text)
            if "unique_last_results" in st.session_state:
                debug_info("📚 unique_last_results = " + json.dumps(st.session_state.unique_last_results, ensure_ascii=False, indent=2))
            if "context" in locals():
                debug_info("📚 context =\n" + context)
            if "gpt_prompt" in locals():
                debug_info("📚 gpt_prompt =\n" + gpt_prompt)
            if "gpt_reply" in locals():
                debug_info("📚 gpt_reply =\n" + gpt_reply)
            if "conversation_history" in st.session_state:
                debug_info("📚 conversation_history = " + json.dumps(list(st.session_state.conversation_history), ensure_ascii=False, indent=2))
            if "last_results" in st.session_state:
                debug_info("📚 last_results = " + json.dumps(st.session_state.last_results, ensure_ascii=False, indent=2))
            if "all_results" in st.session_state:
                debug_info("📚 all_results = " + json.dumps(list(st.session_state.all_results), ensure_ascii=False, indent=2))
            pause_here()
        else:    
            st.rerun()
        