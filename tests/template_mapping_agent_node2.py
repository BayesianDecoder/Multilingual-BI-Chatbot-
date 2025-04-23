import sqlite3
from langchain_ollama import OllamaLLM
import json
import re
from datetime import datetime

class TemplateMappingAgent:
    def __init__(self):
        self.llm = OllamaLLM(model="deepseek-r1:1.5b")
        self.conn = sqlite3.connect('template_memory.db')
        self._init_db()
        
    def _init_db(self):
        """Initialize database with learning capabilities"""
        self.conn.execute('''CREATE TABLE IF NOT EXISTS query_templates (
                                intent TEXT PRIMARY KEY,
                                template TEXT,
                                usage_count INTEGER DEFAULT 0,
                                last_used TEXT
                             )''')
        self.conn.commit()

    def _clean_llm_response(self, response: str) -> str:
        """Removes <think> blocks, markdown, and extra text"""
        response = re.sub(r'<think>.*?</think>', '', response, flags=re.DOTALL).strip()
        response = re.sub(r'```.*?```', '', response, flags=re.DOTALL).strip()
        return response

    def _generate_new_template(self, intent: str, params: dict) -> str:
        """Improved template generation with strict output control"""
        prompt = f"""Generate a SINGLE LINE Python Pandas command for:
- Intent: {intent}
- Parameters: {params}
- Available DataFrames: daily_metrics_df, top_pages_df, traffic_sources_df

Respond ONLY with the Python code. No explanations, no markdown.
Example:
top_pages_df.sort_values('screenPageViews', ascending=False).head(5)
"""
        raw_response = self.llm.invoke(prompt)
        cleaned = self._clean_llm_response(raw_response)
        
        if not re.match(r'^\w+_df\.[\w\(\)\.,=\' ]+$', cleaned):
            return "top_pages_df.sort_values('screenPageViews', ascending=False).head(5)"  # Fallback
        
        return cleaned

    def get_template(self, intent: str, params: dict) -> str:
        """Retrieve existing template or generate a new one"""
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
            self.conn.commit()
            return result[0]
        
        new_template = self._generate_new_template(intent, params)
        
        self.conn.execute(
            "INSERT INTO query_templates (intent, template, last_used) VALUES (?, ?, ?)",
            (intent, new_template, now_str)
        )
        self.conn.commit()
        
        return new_template




    def similarity_check(self, new_query: str) -> str:
       """Suggest similar existing intent or return 'No match' with strict enforcement"""
       cursor = self.conn.execute("SELECT intent FROM query_templates")
       existing_intents = [row[0] for row in cursor.fetchall()]

       if not existing_intents:
          return "No match"

       prompt = f"""
You must pick the closest intent from this list OR reply with "No match":
{', '.join(existing_intents)}

New query: "{new_query}"

Respond EXACTLY with:
- One intent name from the list.
- Or "No match".
No explanations.
"""
       raw_response = self.llm.invoke(prompt)
       cleaned_response = self._clean_llm_response(raw_response)

    # ✅ Final strict validation
       if cleaned_response in existing_intents:
          return cleaned_response
       if cleaned_response.lower() == "no match":
           return "No match"

    # If LLM gives unexpected output, force fallback
       return "No match"


    def list_templates(self):
        """List all stored templates with usage stats"""
        cursor = self.conn.execute("SELECT intent, usage_count, last_used FROM query_templates")
        return cursor.fetchall()

    def __del__(self):
        self.conn.close()

# === Example Usage ===
if __name__ == "__main__":
    agent = TemplateMappingAgent()
    
    # Force clear existing templates for clean run
    agent.conn.execute("DELETE FROM query_templates")
    agent.conn.commit()
    
    # Test template generation
    print(agent.get_template("top_pages_analysis", {"date_range": "last_week"}))
    
    # Test similarity check
    print(agent.similarity_check("Show me traffic by country"))
    
    # Verify storage
    agent.get_template("top_pages_analysis", {})  # Increment usage
    for row in agent.list_templates():
        print(f"{row[0]} | Used: {row[1]} | Last: {row[2]}")
