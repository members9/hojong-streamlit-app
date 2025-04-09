import openai
import faiss
import pickle
import numpy as np
from collections import deque
import os
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")
client = OpenAI(api_key=api_key)

def normalize(vecs):
    norms = np.linalg.norm(vecs, axis=1, keepdims=True)
    return vecs / norms

index = faiss.read_index("service_index.faiss")
with open("service_metadata.pkl", "rb") as f:
    metadata = pickle.load(f)

xb = index.reconstruct_n(0, index.ntotal)
xb = normalize(xb)
d = xb.shape[1]
index_cosine = faiss.IndexFlatIP(d)
index_cosine.add(xb)

SIMILARITY_THRESHOLD = 0.30
last_results = []
excluded_company_ids = set()
all_results = deque(maxlen=3)

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
    print(f"🤖 질문과 관광기업 서비스간 유사도 확인: {max_similarity:.4f}")
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
        history = make_summary_context(all_results)
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
6. 답변 시 불필요하게 특수문자(**, ## 등)로 머릿말을 사용 하지 말아주세요.
7. 부드러운 상담사 말투로 정리해주세요.
"""

print("\n🤖 안녕하세요? 관광기업 서비스 가이드 호종이입니다. 'exit'을 입력하면 종료됩니다.\n")

while True:
    user_input = input("\n🤖 호종이에게 어떤 서비스나 기업을 찾으시는지 물어보세요: \n")

    if user_input.lower() == "exit":
        print("🤖 호종이는 이만 물러가겠습니다. 언제든지 다시 불러주세요!")
        break

    if user_input.startswith("자세히") and last_results:
        try:
            keyword = user_input.replace("자세히", "").strip()
            matches = [s for s in last_results if keyword in s["기업명"]]
            if not matches:
                print("ℹ️  해당 키워드를 포함한 기업명이 없습니다.")
                continue
            elif len(matches) > 1:
                print("ℹ️  여러 개의 기업명이 일치합니다. 더 구체적으로 입력해주세요.")
                for s in matches:
                    print(f"- {s['기업명']}")
                continue

            s = matches[0]
            service_link = f"https://www.tourvoucher.or.kr/user/svcManage/svc/BD_selectSvc.do?svcNo={s['서비스번호']}"
            company_link = f"https://www.tourvoucher.or.kr/user/entrprsManage/provdEntrprs/BD_selectProvdEntrprs.do?entrprsId={s['기업ID']}"
            print("\n📄 서비스 상세정보")
            for k, v in s.items():
                print(f"{k}: {v}")
            print(f"🔗 서비스 링크: {service_link}")
            print(f"🏢 기업 링크: {company_link}")
        except:
            print("❌ 기업명으로 조회 중 오류가 발생했습니다. 예: '자세히 트립'")
        continue

    print("\n🤖 호종이가 질문을 분석 중입니다...")
    if not is_relevant_question(user_input):
        print("ℹ️  죄송하지만, 질문의 내용을 조금 더 관광기업이나 서비스와 관련된 내용으로 다시 해주세요.")
        continue

    best_mode = is_best_recommendation_query(user_input)
    exclude = None if best_mode else excluded_company_ids

    print("🤖 호종이가 관련 서비스를 찾는 중입니다...")
    last_results = recommend_services(user_input, exclude_company_ids=exclude)

    if not best_mode:
        for s in last_results:
            excluded_company_ids.add(s['기업ID'])

    all_results.append(last_results)

    print("🤖 호종이가 추천 내용을 정리 중입니다...")
    context = make_context(last_results)
    gpt_prompt = make_prompt(user_input, context, is_best=best_mode)

    chat_history = [
        {"role": "system", "content": "당신은 관광기업 상담 전문가 호종이입니다."},
        {"role": "user", "content": gpt_prompt}
    ]
    gpt_reply = ask_gpt(chat_history)

    print("\n🤖 호종이 추천:")
    print(gpt_reply)
    print("\nℹ️  각 추천 서비스에 대해 더 알고 싶으면 '자세히 기업명' 처럼 입력하세요.")
