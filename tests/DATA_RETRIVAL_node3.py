import pandas as pd
import json

class DataRetrievalAgent:
    def __init__(self, json_path):
        # Load JSON data
        with open(json_path, 'r') as f:
            data = json.load(f)
        
        # Convert JSON sections to DataFrames
        self.daily_metrics_df = pd.DataFrame(data.get("daily_metrics", []))
        self.top_pages_df = pd.DataFrame(data.get("top_pages", []))
        self.traffic_sources_df = pd.DataFrame(data.get("traffic_sources", []))
        self.device_category_df = pd.DataFrame(data.get("device_category", []))
        self.geography_df = pd.DataFrame(data.get("geography", []))
        self.events_df = pd.DataFrame(data.get("events", []))
    
    def execute_template(self, template: str):
        """Execute the given Pandas query template"""
        try:
            # Provide DataFrames to eval context
            local_vars = {
                "daily_metrics_df": self.daily_metrics_df,
                "top_pages_df": self.top_pages_df,
                "traffic_sources_df": self.traffic_sources_df,
                "device_category_df": self.device_category_df,
                "geography_df": self.geography_df,
                "events_df": self.events_df
            }
            result = eval(template, {}, local_vars)
            return result
        except Exception as e:
            return f" Error executing template: {e}"

# === Example Usage ===
if __name__ == "__main__":
    agent = DataRetrievalAgent("/Users/vijay/Documents/PROJECTS/Spotify recommender systerm/AI-Meeting-Minutes-Generator-main/final /accountancy/bi agent/llm_module.json")
    
    # Example: Template from TemplateMappingAgent
    template = "device_category_df.sort_values('sessions', ascending=False)"

    
    output = agent.execute_template(template)
    print(output)
