

from langgraph.graph import StateGraph, END
from langchain_ollama import OllamaLLM
from langdetect import detect
import sqlite3
import json
import re
from datetime import datetime
from typing import TypedDict, Dict, Any
import pandas as pd

# ====================
# Shared State Definition
# ====================
class AgentState(TypedDict):
    query: str
    intent: str
    parameters: Dict[str, Any]
    language: str
    data: Any
    response: str
    template: str
    desired_language: str

# ====================
# Node 1: Enhanced Intent Recognition
# ====================
class IntentRecognizer:
    def __init__(self):
        self.llm = OllamaLLM(model="deepseek-r1:1.5b", temperature=0.1)
        self.categories = [
            "fetch_daily_metrics", "fetch_top_pages", "fetch_traffic_sources",
            "fetch_device_usage", "fetch_geography", "fetch_events",
            "compare", "trend", "summary"
        ]

    def _detect_language(self, text: str) -> str:
        try:
            code = detect(text)
        except:
            code = ""
        if re.search(r'[\u4e00-\u9fff]', text) and not code.startswith("en"):
            # heuristic: Cantonese often uses “台” vs Mandarin
            return "cantonese" if "台" in text else "mandarin"
        return {"en": "english", "zh-cn": "mandarin", "zh-tw": "cantonese"}.get(code.lower(), "english")

    def process(self, state: AgentState) -> AgentState:
        query = state["query"]
        language = self._detect_language(query)
        prompt = (
            f"Classify this business query and extract intent + parameters.\n"
            f"Query: '{query}'\n"
            f"Options: {', '.join(self.categories)}\n"
            f"Respond ONLY in JSON: {{\"intent\":..., \"parameters\":{{...}}}}"
        )
        try:
            resp = self.llm.invoke(prompt)
            parsed = json.loads(re.search(r'\{.*\}', resp, re.DOTALL).group())
            intent = parsed.get("intent", "unknown")
            params = parsed.get("parameters", {})
        except:
            intent, params = "unknown", {}
        return {**state, "intent": intent, "parameters": params, "language": language}

# ====================
# Node 2: Dynamic Template Generation
# ====================
class TemplateMapper:
    def __init__(self):
        self.conn = sqlite3.connect('template_memory.db')
        self.conn.execute(
            '''CREATE TABLE IF NOT EXISTS query_templates (
               intent TEXT PRIMARY KEY,
               template TEXT,
               usage_count INTEGER DEFAULT 0,
               last_used TEXT)''' 
        )
        self.conn.commit()
        self.llm = OllamaLLM(model="deepseek-r1:1.5b")

    def _generate(self, intent: str, params: dict) -> str:
        prompt = (
            f"Generate a SINGLE-LINE Pandas command for intent '{intent}' with params {params}.\n"
            f"Available DataFrames: daily_metrics_df, top_pages_df, traffic_sources_df.\n"
            f"Respond ONLY with valid Python code."
        )
        code = self.llm.invoke(prompt).strip()
        return code if re.match(r'^\w+_df\.', code) else "top_pages_df.sort_values('screenPageViews', ascending=False).head(5)"

    def process(self, state: AgentState) -> AgentState:
        intent = state["intent"]
        if intent == "unknown":
            return {**state, "template": ""}
        cur = self.conn.execute("SELECT template FROM query_templates WHERE intent=?", (intent,))
        row = cur.fetchone()
        ts = datetime.now().isoformat()
        if row:
            template = row[0]
            self.conn.execute(
                "UPDATE query_templates SET usage_count=usage_count+1, last_used=? WHERE intent=?", (ts, intent)
            )
        else:
            template = self._generate(intent, state["parameters"])
            self.conn.execute(
                "INSERT INTO query_templates(intent,template,usage_count,last_used) VALUES(?,?,1,?)",
                (intent, template, ts)
            )
        self.conn.commit()
        return {**state, "template": template}

# ====================
# Node 3: Data Retrieval
# ====================
class DataRetrievalAgent:
    def __init__(self, json_path="llm_module.json"):
        with open(json_path, 'r') as f:
            raw = json.load(f)
        self.dfs = {
            'daily_metrics_df': pd.DataFrame(raw.get('daily_metrics', [])),
            'top_pages_df': pd.DataFrame(raw.get('top_pages', [])),
            'traffic_sources_df': pd.DataFrame(raw.get('traffic_sources', []))
        }

    def process(self, state: AgentState) -> AgentState:
        tmpl = state["template"]
        if not tmpl:
            return {**state, "data": []}
        try:
            res = eval(tmpl, {}, self.dfs)
            if isinstance(res, pd.DataFrame):
                data = res.to_dict(orient='records')
            elif isinstance(res, pd.Series):
                data = [{res.name: v} for v in res.to_dict().values()]
            else:
                data = [{"result": str(res)}]
        except Exception as e:
            data = [{"error": str(e)}]
        return {**state, "data": data}

# ====================
# Node 4: Response Generation with Follow-Up (from node4.py)
# ====================
from langchain_ollama import OllamaLLM as LLM
import pandas as pd
import re

class ResponseGenerationAgent:
    def __init__(self):
        self.llm = LLM(model="qwen:7b-chat")
        self.column_translations = {
            "mandarin": {
                "pagePath": "页面路径",
                "screenPageViews": "页面浏览量",
                "activeUsers": "活跃用户",
                "sessions": "会话数",
                "engagementRate": "参与率"
            },
            "cantonese": {
                "pagePath": "頁面路徑",
                "screenPageViews": "頁面瀏覽量",
                "activeUsers": "活躍用戶",
                "sessions": "會話數",
                "engagementRate": "參與率"
            }
        }

    def _get_example(self, language: str, intent: str) -> str:
        examples = {
            "mandarin": {
                "fetch_top_pages": (
                    "关键分析：首页以520次浏览量领先，显著高于产品页（430次）\n"
                    "建议后续问题：是否需要分析不同流量来源的页面表现？"
                )
            },
            "cantonese": {
                "fetch_top_pages": (
                    "關鍵分析：首頁以520次瀏覽量領先，顯著高於產品頁（430次）\n"
                    "建議後續問題：是否需要分析不同流量來源嘅頁面表現？"
                )
            },
            "english": {
                "fetch_top_pages": (
                    "Key Analysis: Homepage leads with 520 views, significantly higher than products page (430)\n"
                    "Suggested Next Question: Would you like to analyze page performance by traffic source?"
                )
            }
        }
        return examples.get(language, {}).get(intent, "")

    def _remove_english(self, text: str, language: str) -> str:
        if re.search(r'[a-zA-Z]', text):
            correction_prompt = f"""
請將以下內容中的英文完全翻譯為{language}:
{text}

純中文版本:"""
            return self.llm.invoke(correction_prompt).strip()
        return text

    def process(self, state: AgentState) -> AgentState:
        df = pd.DataFrame(state["data"])
        lang = state.get("desired_language") or state["language"]
        lang = lang.lower()
        if lang in self.column_translations:
            df = df.rename(columns=self.column_translations[lang])
        preview = df.head(5).to_markdown(index=False)

        prompt = f"""
你必须严格遵守这些规则！You must strictly follow these rules!
Respond ONLY in {lang.upper()}. Never use English words except numbers.
Focus on business insights from this data:

{preview}

Format:
关键分析 (Key Analysis in Chinese) / Key Analysis (English)
建议后续问题 (Suggested Follow-up in Chinese) / Suggested Next Question (English)

Example:
{self._get_example(lang, state['intent'])}

现在开始/Now begin:
"""
        response = self.llm.invoke(prompt).strip()
        if lang != "english":
            response = self._remove_english(response, lang)
        return {**state, "response": response}

# ====================
# Build & Run the Graph
# ====================
intent_recognizer   = IntentRecognizer()
template_mapper     = TemplateMapper()
data_retriever      = DataRetrievalAgent()
response_generator  = ResponseGenerationAgent()

workflow = StateGraph(AgentState)
workflow.add_node("intent_recognition", intent_recognizer.process)
workflow.add_node("template_mapping", template_mapper.process)
workflow.add_node("data_retrieval", data_retriever.process)
workflow.add_node("response_generation", response_generator.process)
workflow.set_entry_point("intent_recognition")
workflow.add_edge("intent_recognition", "template_mapping")
workflow.add_edge("template_mapping",     "data_retrieval")
workflow.add_edge("data_retrieval",       "response_generation")
workflow.add_edge("response_generation",  END)

app = workflow.compile()

if __name__ == "__main__":
    tests = [
        {"query":"Show top performing pages last week","desired_language":"english"}
       
    ]
    for t in tests:
        state = {
            "query": t["query"],
            "intent": "", "parameters": {}, "language": "",
            "data": [], "template": "", "response": "",
            "desired_language": t["desired_language"]
        }
        out = app.invoke(state)
        print(f"\n--- {t['query']} ({t['desired_language'].upper()}) ---")
        print(out["response"])
        print(pd.DataFrame(out["data"]).head().to_markdown(index=False))
