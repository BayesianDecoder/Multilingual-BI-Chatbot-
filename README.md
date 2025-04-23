


# 🚀 Multilingual BI Chatbot using LangGraph, LLMs & Pandas

This project is a dynamic **Business Intelligence (BI) chatbot pipeline** that processes multilingual queries (English, Mandarin, Cantonese) to generate business insights from analytics data. It leverages **LangGraph**, **LLMs (Ollama + DeepSeek + Qwen)**, **Pandas**, and **Streamlit** for interactive visualization.

---

## 📊 Overview

The system intelligently understands business-related queries, fetches relevant data, and responds with clear insights and visualizations. It is modular, scalable, and designed for future enhancements like integrating real-time data from APIs such as **Google Analytics**.

---

## ⚡ Pipeline Architecture

The workflow is built using **LangGraph's StateGraph**, consisting of 4 key nodes:

```plaintext
User Query
   │
   ▼
[Node 1] IntentRecognizer
   │
   ▼
[Node 2] TemplateMapper
   │
   ▼
[Node 3] DataRetrievalAgent
   │
   ▼
[Node 4] ResponseGenerationAgent
   │
   ▼
   END ➔ Streamlit UI + Plotly Visualization

```
# 🔹 Node Descriptions

## 1️⃣ IntentRecognizer
**Purpose:**  
Classifies the user query into a business intent and detects the query language.

**Key Features:**
- Uses `langdetect` for language detection.
- Sends query to `deepseek-r1:1.5b` LLM for:
  - Intent classification (e.g., `fetch_top_pages`, `compare`, `trend`).
  - Parameter extraction (date ranges, metrics).

**Output:**  
Updates state with `intent`, `parameters`, and `language`.


## 2️⃣ TemplateMapper
**Purpose:**  
Generates or retrieves a dynamic Pandas query template to fetch relevant data.

**Key Features:**
- Uses **SQLite** to cache templates for efficiency.
- If intent is new, prompts LLM to generate the query.
- Provides fallback for invalid LLM responses.

**Output:**  
Adds the Pandas `template` to the state.


## 3️⃣ DataRetrievalAgent
**Purpose:**  
Executes the Pandas command to retrieve data from preloaded datasets.

**Key Features:**
- Loads data from `llm_module.json`.
- Dynamically executes the query using `eval()`.
- Handles different return types (`DataFrame`, `Series`, etc.).
- Robust error handling.

🚀 **Future Enhancement:**  
This node can be extended to connect directly with **Google Analytics API**, replacing static JSON data with live, real-time analytics data. This makes the chatbot suitable for production BI environments.

**Output:**  
Appends the retrieved `data` to the state.


## 4️⃣ ResponseGenerationAgent
**Purpose:**  
Generates human-readable insights and suggests follow-up questions.

**Key Features:**
- Uses `qwen:7b-chat` LLM for:
  - Business insight generation.
  - Follow-up question suggestions.
- Supports multilingual responses (**English**, **Mandarin**, **Cantonese**).
- Translates column names for localized responses.

# 🎯 Example Flow

## User Query:
**"Show top performing pages last week"**

| **Node**              | **Action**                                                           |
|-----------------------|----------------------------------------------------------------------|
| **IntentRecognizer**  | ➔ Detects: `fetch_top_pages`  <br> ➔ Language: `English`             |
| **TemplateMapper**    | ➔ Generates: `top_pages_df.sort_values('screenPageViews').head(5)`   |
| **DataRetrievalAgent**| ➔ Retrieves top 5 pages from dataset                                 |
| **ResponseGenerator** | ➔ Insight: `"Homepage leads with 520 views..."` <br> ➔ Suggests next question |


# 🚀 Installation Guide for Multilingual BI Chatbot

Follow these steps to set up and run the Multilingual BI Chatbot on your local machine.

---

## 1️⃣ Clone the Repository

```bash
git clone https://github.com/your-repo/bi-multilingual-chatbot.git
cd bi-multilingual-chatbot


```

## 2️⃣ Install Dependencies
Make sure you have Python 3.9+ and pip installed.

```bash
pip install -r requirements.txt

```

## 3️⃣ Start Ollama Server
This project uses Ollama for LLM inference. Ensure you have Ollama installed and running.
```bash
ollama serve

```

## 4️⃣ Run the Streamlit App

```bash
streamlit run app.py
```


# 📂 Project Structure

```plaintext
├── app.py                 # Streamlit UI - Frontend chatbot interface
├── main.py                # LangGraph pipeline - Core logic with 4 intelligent agents
├── llm_module.json        # Mock analytics data used for querying
├── template_memory.db     # SQLite cache to store and reuse generated templates
├── requirements.txt       # Python dependencies
└── README.md              # Project documentation
```

# 🎥 Additional Resources

- **📄 Code Explanation Walkthrough**:  
[Watch Here](https://drive.google.com/file/d/1hIsu1UFxhT2q4hddIiMYHfQKhCUM9Q1z/view?usp=drivesdk)

- **🚀 Sample Demo of the Chatbot in Action**:  
[View Demo](https://drive.google.com/file/d/1SZdfJMQWY-mjt3KuTanZhYu2YYd08sL5/view?usp=drivesdk)




## 🚀 Tech Stack Used

| **Technology**   | **Purpose**                                             |
|------------------|----------------------------------------------------------|
| **Python**       | Core programming language                                |
| **Streamlit**    | Interactive chatbot UI                                   |
| **Pandas**       | Data manipulation and analysis                           |
| **Plotly**       | Data visualization (charts & graphs)                     |
| **LangGraph**    | Manages multi-step AI workflows (StateGraph pipeline)    |
| **Ollama + LLMs**| Local LLM serving (`deepseek-r1:1.5b`, `qwen:7b-chat`)   |
| **LangDetect**   | Language detection (English, Mandarin, Cantonese)        |
| **SQLite**       | Template caching for dynamic query generation            |
| **JSON**         | Mock data storage                                        |



## 🚀 Future Improvements
- Google Analytics API Integration for live data.
- Add more intents (e.g., revenue analysis, user segmentation).
- Role-based access and user customization.
  
