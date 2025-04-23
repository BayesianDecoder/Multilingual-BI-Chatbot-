
# Node 1: IntentRecognitionAgent
from langchain_ollama import OllamaLLM
import json
import re
import sqlite3
from datetime import datetime

class IntentRecognitionAgent:
    def __init__(self):
        self.llm = OllamaLLM(model="deepseek-r1:1.5b")
        self.available_categories = [
            "fetch_daily_metrics", "fetch_top_pages", "fetch_traffic_sources",
            "fetch_device_usage", "fetch_geography", "fetch_events"
        ]

    def detect_intent(self, query: str) -> dict:
        from langdetect import detect
        
        try:
            lang_code = detect(query)
        except:
            lang_code = "en"

        language_map = {
            "en": "english",
            "zh-cn": "mandarin",
            "zh-tw": "cantonese",
            "zh": "mandarin"
        }
        language = language_map.get(lang_code.lower(), "english")

        prompt = f"""
You are an AI assistant for business analytics.
Classify this query:
- Identify intent from: {' | '.join(self.available_categories)}
- Extract parameters: date_range, metric, dimension, limit.

Respond ONLY in JSON:
{{
  "intent": "...",
  "parameters": {{ "date_range": "...", "metric": "...", "dimension": "...", "limit": ... }}
}}

Query: "{query}"
"""
        response = self.llm.invoke(prompt)
        try:
            json_match = re.search(r'\{.*\}', response, re.DOTALL)
            parsed = json.loads(json_match.group()) if json_match else {"intent": "unknown", "parameters": {}}
        except:
            parsed = {"intent": "unknown", "parameters": {}}

        return {
            "intent": parsed["intent"],
            "parameters": parsed["parameters"],
            "language": language
        }

# Node 2: TemplateMappingAgent
class TemplateMappingAgent:
    def __init__(self):
        self.llm = OllamaLLM(model="deepseek-r1:1.5b")
        self.conn = sqlite3.connect('template_memory.db')
        self._init_db()
        
    def _init_db(self):
        self.conn.execute('''CREATE TABLE IF NOT EXISTS query_templates (
                                intent TEXT PRIMARY KEY,
                                template TEXT,
                                usage_count INTEGER DEFAULT 0,
                                last_used TEXT
                             )''')
        self.conn.commit()

    def _clean_llm_response(self, response: str) -> str:
        response = re.sub(r'<think>.*?</think>', '', response, flags=re.DOTALL).strip()
        response = re.sub(r'```(?:python)?(.*?)```', r'\1', response, flags=re.DOTALL).strip()
        return response

    def _generate_template(self, intent: str, params: dict) -> str:
        prompt = f"""Generate a SINGLE LINE Python Pandas command for:
- Intent: {intent}
- Parameters: {params}
- Available DataFrames: daily_metrics_df, top_pages_df, traffic_sources_df

Respond ONLY with valid Python code."""
        raw_response = self.llm.invoke(prompt)
        cleaned = self._clean_llm_response(raw_response)
        if not re.match(r'^\w+_df\.[\w\(\)\.\[\]\', =<>\"-]+$', cleaned):
            return "top_pages_df.sort_values('screenPageViews', ascending=False).head(5)"
        return cleaned

    def get_template(self, intent: str, params: dict) -> str:
        cursor = self.conn.execute(
            "SELECT template FROM query_templates WHERE intent=?",
            (intent,)
        )
        result = cursor.fetchone()
        now_str = datetime.now().isoformat()

        if result:
            self.conn.execute(
                "UPDATE query_templates SET usage_count = usage_count + 1, last_used = ? WHERE intent=?",
                (now_str, intent)
            )
            template = result[0]
        else:
            template = self._generate_template(intent, params)
            self.conn.execute(
                "INSERT INTO query_templates (intent, template, usage_count, last_used) VALUES (?, ?, ?, ?)",
                (intent, template, 1, now_str)
            )
        self.conn.commit()
        return template

    def __del__(self):
        self.conn.close()

# === Example usage ===
if __name__ == "__main__":
    agent = IntentRecognitionAgent()

    # Test with Mandarin Query
    result = agent.detect_intent("上周我的网站访问量是多少？")
    print(f"\n Final Output:\n{json.dumps(result, indent=2, ensure_ascii=False)}")

    # Test with English Query
    result = agent.detect_intent("Show me the top pages last month")
    print(f"\n Final Output:\n{json.dumps(result, indent=2)}")
