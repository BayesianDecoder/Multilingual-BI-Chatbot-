from langchain_ollama import OllamaLLM
import pandas as pd
import re

class ResponseGenerationAgent:
    def __init__(self):
        self.llm = OllamaLLM(model="qwen:7b-chat")
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
    
    def generate_response(self, dataframe, intent: str, language: str):
        # Normalize language input
        language = language.lower()
        if language not in ["english", "mandarin", "cantonese"]:
            language = "english"
        
        # Translate dataframe columns if needed
        if language != "english":
            translated_df = dataframe.rename(columns=self.column_translations[language])
            data_preview = translated_df.head(5).to_markdown(index=False)
        else:
            data_preview = dataframe.head(5).to_markdown(index=False)

        # Strict language enforcement prompt
        prompt = f"""
你必须严格遵守这些规则！You must strictly follow these rules!
Respond ONLY in {language.upper()}. 
Never use English words except numbers.
Focus on business insights from this data:

{data_preview}

Format:
关键分析 (Key Analysis in Chinese) / Key Analysis (English)
建议后续问题 (Suggested Follow-up in Chinese) / Suggested Next Question (English)

Example ({'中文' if language != 'english' else 'English'}):
{self._get_example(language, intent)}

现在开始/Now begin:"""
        
        response = self.llm.invoke(prompt).strip()
        
        # Post-process to ensure language compliance
        if language != "english":
            response = self._remove_english(response, language)
            
        return response

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
        """Remove any remaining English characters through LLM correction"""
        if re.search(r'[a-zA-Z]', text):
            correction_prompt = f"""
请将以下内容中的英文完全翻译为{language}:
{text}

纯中文版本:"""
            return self.llm.invoke(correction_prompt)
        return text

# === Example Usage ===
if __name__ == "__main__":
    df = pd.DataFrame({
        "pagePath": ["/home", "/products", "/contact"],
        "screenPageViews": [520, 430, 210]
    })

    agent = ResponseGenerationAgent()

    print("\nMandarin Output:")
    print(agent.generate_response(df, "fetch_top_pages", "mandarin"))

    # print("\nEnglish Output:")
    # print(agent.generate_response(df, "fetch_traffic_sources", "english"))