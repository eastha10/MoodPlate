# LLM2.py — 웹/CLI 겸용 그래프 통합본

from __future__ import annotations
from typing import TypedDict, List, Tuple, Optional, Dict, Any
import os, re, difflib
from pathlib import Path
import pandas as pd

from dotenv import load_dotenv
from langchain.prompts import PromptTemplate
from langchain_naver import ChatClovaX
from langgraph.graph import StateGraph, END
from langchain_community.tools.tavily_search import TavilySearchResults

# =========================
# 0) 환경설정 (.env 로드)
# =========================
# 프로젝트 구조 예시:
#   project_root/.env
#   project_root/script/LLM2.py  ← 이 파일
CURRENT_DIR = Path(__file__).resolve().parent
BASE_DIR = CURRENT_DIR.parent
ENV_PATH = BASE_DIR / ".env"
load_dotenv(dotenv_path=str(ENV_PATH))  # settings.py에서 이미 로드하면 생략 가능

# =========================
# 1) CSV 로드 (메뉴 DB)
# =========================
# CSV가 본 파일과 같은 폴더에 있다고 가정
DEFAULT_CSV_NAME = "food_preprocessed_v23_richer_descriptions_utf8bom.csv"
MENU_CSV_PATH = CURRENT_DIR / DEFAULT_CSV_NAME
if not MENU_CSV_PATH.exists():
    # 노트북/서버 등에서 별도 경로에 있을 때 대비
    alt_path = Path("/mnt/data") / DEFAULT_CSV_NAME
    if alt_path.exists():
        MENU_CSV_PATH = alt_path

def _safe_get(rec: pd.Series, col: str) -> str:
    v = rec.get(col, "")
    return (str(v).strip() if pd.notna(v) else "")

def load_menu_db(csv_path: Path) -> Dict[str, Dict[str, Any]]:
    df = pd.read_csv(csv_path, encoding="utf-8-sig")
    db: Dict[str, Dict[str, Any]] = {}
    for _, r in df.iterrows():
        name = _safe_get(r, "name")
        if not name:
            continue
        tags = [t for t in (_safe_get(r, "tag1"), _safe_get(r, "tag2"), _safe_get(r, "tag3")) if t]
        db[name] = {
            "desc": _safe_get(r, "description"),
            "tags": tags,
            "difficulty": _safe_get(r, "difficulty"),
            "cat_text": _safe_get(r, "cat_text"),
            "baemin_category": _safe_get(r, "baemin_category"),
            # 호환 키
            "category": _safe_get(r, "baemin_category"),
        }
    return db

menu_db = load_menu_db(MENU_CSV_PATH)

# 정규화 인덱스(이름 근사 매칭)
def normalize(s: str) -> str:
    return re.sub(r"\s+", "", str(s)).strip().casefold()

norm_index = {normalize(k): k for k in menu_db.keys()}

def resolve_menu_name(name: str) -> str:
    n = normalize(name)
    if n in norm_index:
        return norm_index[n]
    candidates = difflib.get_close_matches(n, norm_index.keys(), n=1, cutoff=0.8)
    return norm_index[candidates[0]] if candidates else name.strip()

# =========================
# 2) LLM 구성 (HCX-005)
# =========================
llm = ChatClovaX(
    model="HCX-005",
    temperature=0.7,
    # base_url="https://clovastudio.stream.ntruss.com/v1/openai",  # 기본값
)

# =========================
# 3) 프롬프트
# =========================
summary_prompt = PromptTemplate.from_template(
    "다음은 사용자의 오늘 기분/상황/식사 선호입니다:\n\n{user_input}\n\n이를 간단히 요약해줘."
)
recommend_prompt = PromptTemplate.from_template(
    "요약된 식사 상황: {summary}\n\n이 상황에 어울리는 식사 메뉴 3가지를 한 단어의 메뉴명과 10~20단어의 간단한 설명으로 추천해줘."
)
refine_prompt = PromptTemplate.from_template(
    "이전 추천 메뉴:\n{recommendations}\n\n사용자 피드백:\n{feedback}\n\n상황 요약:\n{summary}\n\n"
    "피드백을 반영하여 새로운 식사 메뉴 3가지를 추천해줘. 한 단어의 메뉴명과 10~20단어의 간단한 설명 포함."
)
feedback_check_prompt = PromptTemplate.from_template(
    "사용자 피드백:\n{feedback}\n\n이 피드백이 만족을 의미하거나 좋다라는 의미면 'finish', 아니면 'refine' 이라고 단답으로 답해줘."
)
recipe_prompt = PromptTemplate.from_template(
    "추천받은 식사의 레시피를 받아 이를 알려줄 거야."
    "아래는 '{menu_name}'에 대한 웹 검색 결과야.\n\n"
    "{web_content}\n\n"
    "출처: {source}\n\n"
    "이 레시피를 요약해서 설명해 주세요. 출처도 간단히 언급해 주세요."
)

# =========================
# 4) 체인
# =========================
summary_chain = summary_prompt | llm
recommend_chain = recommend_prompt | llm
refine_chain = refine_prompt | llm
feedback_check_chain = feedback_check_prompt | llm
recipe_chain = recipe_prompt | llm

# =========================
# 5) 상태 타입
# =========================
class MealState(TypedDict, total=False):
    user_input: str
    summary: str
    recommendations: str
    feedback: str
    recipe_menus: list[str]
    recipe_menu: str
    recipes: Dict[str,str]
    recipe: str
    next_step: str
    force_finish: bool
    accepted_menus: list[str]
    accepted_menu: str


# =========================
# 6) 유틸 (메타 구성)
# =========================
STOPWORDS = {"및"}

def _unique_preserve(seq):
    seen = set(); out = []
    for x in seq:
        if x not in seen:
            seen.add(x); out.append(x)
    return out

def _split_tokens(s: str):
    if not s:
        return []
    tokens = re.split(r"[,\s/|]+", str(s))
    return [t for t in (tok.strip() for tok in tokens) if t and t not in STOPWORDS]

def format_with_meta(name: str, base_desc: str) -> str:
    info = menu_db.get(name)
    if not info:
        return base_desc

    # tag1~3
    tags = [t for t in info.get("tags", []) if str(t).strip()]
    tags_hash = [(t if str(t).startswith("#") else f"#{t}") for t in _unique_preserve(tags)]

    # cat_text / baemin_category
    cat_text_list = _split_tokens(info.get("cat_text", ""))
    baemin_raw = info.get("baemin_category", "") or info.get("category", "")
    baemin_list = _split_tokens(baemin_raw)

    diff = str(info.get("difficulty", "")).strip()

    parts = []
    if tags_hash:
        parts.append("태그: " + ", ".join(tags_hash))
    if cat_text_list:
        parts.append("분류(cat_text): " + ", ".join(cat_text_list))
    if baemin_list:
        parts.append("배민카테고리: " + ", ".join(baemin_list))
    if diff:
        parts.append(f"난이도: {diff}")

    return f"{base_desc} ({'; '.join(parts)})" if parts else base_desc

def extract_menus_and_descriptions(text: str) -> List[Tuple[str, str]]:
    lines = text.splitlines()
    result: List[Tuple[str, str]] = []
    for line in lines:
        # "1. 김치찌개 - 설명" 등 다양한 구분자 지원
        m = re.match(r"\s*\d+\.\s*([^\-–:·\)]+)[\-–:·\)]\s*(.+)", line)
        if m:
            menu, desc = m.groups()
            result.append((menu.strip(), desc.strip()))
    return result

# =========================
# 7) 노드 함수
# =========================
def summarize(state: MealState) -> MealState:
    summary = summary_chain.invoke({"user_input": state["user_input"]}).content
    return {**state, "summary": summary}

def recommend(state: MealState) -> MealState:
    raw = recommend_chain.invoke({"summary": state["summary"]}).content
    menu_entries = extract_menus_and_descriptions(raw)

    enriched_lines = []
    for idx, (menu, desc) in enumerate(menu_entries[:3], 1):
        key = resolve_menu_name(menu.strip())
        final_desc = menu_db.get(key, {}).get("desc", desc.strip())
        final_desc = format_with_meta(key, final_desc)
        enriched_lines.append(f"{idx}. {key} - {final_desc}")

    return {**state, "recommendations": "\n".join(enriched_lines)}

def collect_feedback(state: MealState) -> MealState:
    """
    웹/CLI 겸용: feedback이 없으면 입력 대기 상태로 종료 (await_feedback)
    """
    if state.get("force_finish"):
        return {**state, "next_step": "finish"}
    fb = (state.get("feedback") or "").strip()
    if not fb:
        return {**state, "next_step": "await_feedback"}
    return state

def check_feedback(state: MealState) -> MealState:
    if state.get("force_finish"):
        return {**state, "next_step": "finish"}
    
    result = feedback_check_chain.invoke({"feedback": state["feedback"]}).content.strip()
    next_step = "finish" if "finish" in result.lower() else "refine"
    return {**state, "next_step": next_step}

def refine_recommendation(state: MealState) -> MealState:
    raw = refine_chain.invoke({
        "summary": state["summary"],
        "recommendations": state["recommendations"],
        "feedback": state["feedback"]
    }).content
    menu_entries = extract_menus_and_descriptions(raw)

    enriched_lines = []
    for idx, (menu, desc) in enumerate(menu_entries[:3], 1):
        key = resolve_menu_name(menu.strip())
        final_desc = menu_db.get(key, {}).get("desc", desc.strip())
        final_desc = format_with_meta(key, final_desc)
        enriched_lines.append(f"{idx}. {key} - {final_desc}")

    return {**state, "recommendations": "\n".join(enriched_lines)}

def _extract_recommended_names(rec_text: str) -> List[str]:
    names = []
    for line in rec_text.splitlines():
        m = re.match(r"\s*\d+\.\s*([^-]+)", line)
        if m:
            names.append(m.group(1).strip())
    return names

def show_recipe(state: MealState) -> MealState:
    """
    - recipe_menus(list) 또는 recipe_menu(str)를 입력으로 받아 여러 개 레시피 생성
    - 추천 목록에 포함된 메뉴만 허용
    - 유효한 것이 하나도 없으면 await_recipe로 대기
    - 결과는 state['recipes'] = {name: text}, 하위호환으로 state['recipe'] = 첫 번째 항목 텍스트
    """
    # 1) 요청된 메뉴들 수집 (문자열/리스트 모두 허용)
    requested: List[str] = []
    if isinstance(state.get("recipe_menus"), list):
        requested.extend([str(x).strip() for x in state["recipe_menus"] if str(x).strip()])
    single = (state.get("recipe_menu") or "").strip()
    if single:
        requested.append(single)

    # 없으면 대기
    if not requested:
        return {**state, "next_step": "await_recipe"}

    # 2) 추천 목록에 있는지 검증
    rec_names = _extract_recommended_names(state.get("recommendations", ""))
    # 추천에 포함된 메뉴만 추려냄
    valid = [name for name in requested if name in rec_names]

    if not valid:
        # 모두 추천에 없으면 대기
        return {**state, "next_step": "await_recipe"}

    # 3) 레시피 생성 루프
    recipes: Dict[str, str] = {}
    search = TavilySearchResults()

    for name in valid:
        # Tavily 검색
        docs = search.invoke({"query": f"{name} 레시피"})
        top_doc = docs[0] if docs else None

        if top_doc:
            content = top_doc.get("content", "")
            url = top_doc.get("url", "")
            title = top_doc.get("title", "검색결과")
            recipe_text = recipe_chain.invoke({
                "menu_name": name,
                "web_content": content,
                "source": f"{title} ({url})"
            }).content.strip()
        else:
            recipe_text = llm.invoke(
                f"{name} 레시피를 4단계로 요약해줘. 출처도 포함해줘."
            ).content.strip()

        recipes[name] = recipe_text

    # 4) state에 저장 (하위호환: 첫 번째 항목을 recipe에도 넣어줌)
    first_recipe = next(iter(recipes.values())) if recipes else None
    return {**state, "recipes": recipes, "recipe": first_recipe}


def finish(state: MealState) -> MealState:
    return state

# =========================
# 8) 그래프 구성
# =========================
builder = StateGraph(MealState)
builder.add_node("summarize", summarize)
builder.add_node("recommend", recommend)
builder.add_node("collect_feedback", collect_feedback)
builder.add_node("check_feedback", check_feedback)
builder.add_node("refine_recommend", refine_recommendation)
builder.add_node("show_recipe", show_recipe)
builder.add_node("finish", finish)

builder.set_entry_point("summarize")
builder.add_edge("summarize", "recommend")
builder.add_edge("recommend", "collect_feedback")

def route_after_collect(state: MealState):
    # ✅ 버튼 종료가 최우선
    if state.get("force_finish"):
        return "finish_now"
    fb = (state.get("feedback") or "").strip()
    return "check" if fb else "await"

builder.add_conditional_edges(
    "collect_feedback",
    route_after_collect,
    {
        "finish_now": "show_recipe",  # 바로 레시피 단계로 이동(원치 않으면 'finish')
        "check": "check_feedback",
        "await": END,
    },
)


builder.add_conditional_edges(
    "check_feedback",
    lambda s: s["next_step"],    # 'refine' or 'finish'
    {
        "refine": "refine_recommend",
        "finish": "show_recipe",
    },
)

builder.add_edge("refine_recommend", "collect_feedback")

def route_after_show(state: MealState):
    if state.get("recipe"):
        return "done"
    return "await"

builder.add_conditional_edges(
    "show_recipe",
    route_after_show,
    {
        "done": "finish",
        "await": END,  # 입력 대기 종료(next_step='await_recipe')
    },
)

builder.set_finish_point("finish")
app = builder.compile()

# =========================
# 9) 웹에서 사용할 헬퍼 (옵션)
# =========================
def step_once(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    웹 서버에서 한 번 호출할 때 사용:
    - 현재 state로 app.invoke 실행
    - 결과 state 반환
    """
    return app.invoke(state)

# =========================
# 10) CLI 테스트 (옵션)
# =========================
def run_graph_cli():
    """
    CLI에서 대화형으로 테스트할 수 있는 루프.
    웹에서는 사용하지 않아도 됩니다.
    """
    state: MealState = {"user_input": input("오늘 식사 관련 기분/상황을 말해주세요: ").strip()} #오늘은 비가 오고 우울해서 따뜻하고 든든한 식사가 먹고 싶어.
    while True:
        state = app.invoke(state)
        print("\n=== 상태 ===")
        # if state.get("summary"):
            # print("요약:", state["summary"])
        if state.get("recommendations"):
            print(state["recommendations"])
        if state.get("recipes"):
            print("\n레시피(다중):")
            for name, text in state["recipes"].items():
                print(f"\n▶ {name}\n{text}")

        step = state.get("next_step", "")
        if step == "await_feedback":
            fb = input("\n🗣️ 피드백을 입력(엔터로 건너뛰기): ").strip()
            if not fb:
                print("종료합니다."); break
            state["feedback"] = fb
        elif step == "await_recipe":
            menus_raw = input("레시피가 궁금한 메뉴들(쉼표로 구분): ").strip()
            if not menus_raw:
                print("종료합니다."); break
            menus = [m.strip() for m in menus_raw.split(",") if m.strip()]
            state.pop("recipe_menu", None)      # 혹시 단일값 남아있으면 제거
            state["recipe_menus"] = menus
        else:
            # finish 도달 또는 대기 없음
            print("\n플로우 종료.")
            break

if __name__ == "__main__":
    # CLI 테스트용
    run_graph_cli()
