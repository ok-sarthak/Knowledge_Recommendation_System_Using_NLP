# 🧠 Knowledge Recommendation Engine using Gemini

A smart knowledge management system that uses Google's Gemini 1.5 Flash to understand, categorize, and retrieve information. This application allows users to build a personal knowledge base and interact with it using natural language.

[![Hugging Face Spaces](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Spaces-blue)](https://huggingface.co/spaces/sharkthak/Knowledge_Recommendation)

---

## 📖 About The Project

This project addresses a common challenge in information management: finding relevant knowledge in a growing database can be difficult. Traditional systems often rely on simple keyword matching, which can fail if the user's query doesn't contain the exact words used in the stored documents. This application solves that problem by leveraging the power of Large Language Models (LLMs) to enable **semantic search**.

### The Workflow

Instead of just storing raw text, the application follows an intelligent processing pipeline for every piece of knowledge added:

1.  **Enrichment**: When a user submits a piece of text, it's sent to the Gemini 1.5 Flash model, which automatically:
    * **Summarizes** the content into a concise overview.
    * **Extracts** the most relevant keywords.
    * **Categorizes** the text into a predefined set (e.g., Technology, Health, Business).
    * **Analyzes** the overall sentiment (Positive, Negative, or Neutral).

2.  **Vectorization**: The original text is then converted into a numerical representation called a **vector embedding** using Google's `embedding-001` model. This vector captures the core semantic meaning of the text, allowing for comparisons based on concepts rather than just words.

3.  **Storage**: The original content, along with all its generated metadata and the vector embedding, is stored in a local SQLite database.

### The Retrieval Process

When a user asks a question, the system performs a sophisticated retrieval process:

1.  The user's query is also converted into a vector embedding.
2.  This query vector is compared against the vectors of all knowledge items in the database using **cosine similarity**. This mathematical operation determines which documents are semantically closest in meaning to the user's question.
3.  The most relevant documents (the "context") are retrieved.
4.  Finally, this context is passed to the Gemini model along with the original question. The model then synthesizes a comprehensive, conversational answer based on the information found in the knowledge base.

This approach ensures that users can find what they're looking for using natural, everyday language, making the system feel intuitive and highly efficient.

### ✨ Key Features

* **Add Knowledge**: Easily add new information to the system. The app automatically processes and enriches it.
* **Semantic Search**: Ask questions in plain English. The app finds the most relevant documents based on meaning, not just keywords.
* **AI-Powered Answers**: Get conversational answers to your questions, synthesized from the knowledge you've added.
* **Knowledge Statistics**: View a dashboard with statistics about your knowledge base, including total items and a breakdown by category and sentiment.
* **Simple UI**: A clean, multi-tab interface built with Gradio makes the system easy to use.

### 🛠️ Tech Stack

* **Backend**: Python
* **AI Model**: Google Gemini 1.5 Flash & `embedding-001`
* **Web Framework**: Gradio
* **Database**: SQLite
* **Core Libraries**: `google-generativeai`, `scikit-learn`, `pandas`

---

## 🚀 Setup and Deployment

This application is designed to be deployed for free on Hugging Face Spaces.

### Prerequisites

* A [Hugging Face](https://huggingface.co/) account (free).
* A Google AI API Key. You can get one from the [Google AI Studio](https://aistudio.google.com/app/apikey).

### Deployment Steps

1.  **Create a New Space:**
    * On Hugging Face, click your profile picture and select **New Space**.
    * Give your Space a name.
    * Select **Gradio** as the Space SDK.
    * Choose **Public** for visibility.
    * Click **Create Space**.

2.  **Create `requirements.txt`:**
    * Create a file named `requirements.txt` with the following content:
        ```text
        google-generativeai
        gradio
        scikit-learn
        pandas
        ```

3.  **Upload Files:**
    * In your new Space, go to the **Files** tab.
    * Upload your `app.py` file.
    * Upload the `requirements.txt` file you just created.

4.  **Add Your API Key as a Secret:**
    * In your Space, go to the **Settings** tab.
    * Scroll down to the **Repository secrets** section.
    * Click **New secret**.
    * **Name:** `GOOGLE_API_KEY`
    * **Value:** Paste your actual Google AI API key here.

The Space will automatically build and launch your application.

---

## 📂 Project Files

* **`app.py`**: The main Python script containing all the application logic. It defines the `KnowledgeInformationSystem` class, handles all interactions with the Gemini API, and builds the Gradio user interface.
* **`requirements.txt`**: Lists all the Python libraries required to run the application.
* **`knowledge_base_gemini.db`** (auto-generated): The SQLite database file where all the knowledge items are stored. This file will be created automatically the first time you add an item.
