# app.py for Hugging Face Spaces

import google.generativeai as genai
import os
import json
import sqlite3
import numpy as np
from datetime import datetime
from typing import List, Dict, Tuple
import logging
from dataclasses import dataclass
import gradio as gr
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd

# --- Configuration ---
# Hugging Face will set the GOOGLE_API_KEY as an environment variable from the secrets manager.
try:
    api_key = os.environ.get('GOOGLE_API_KEY')
    if not api_key:
        raise ValueError("API Key not found in environment variables. Please set it in your Hugging Face Space secrets.")
    genai.configure(api_key=api_key)
except Exception as e:
    print(f"Error configuring API key: {e}")

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# --- Data Classes and Core Logic ---
@dataclass
class KnowledgeItem:
    id: int
    content: str
    summary: str
    keywords: List[str]
    category: str
    sentiment: str
    embedding: List[float]
    created_at: str
    relevance_score: float = 0.0

class KnowledgeInformationSystem:
    # MODIFIED FOR HUGGING FACE: The db_path points to a local file in the Space.
    def __init__(self, api_key: str, db_path: str = "knowledge_base_gemini.db"):
        """Initialize the Knowledge Information System with Google GenAI API"""
        self.model = genai.GenerativeModel('gemini-1.5-flash-latest')
        self.chat = self.model.start_chat(history=[])
        self.db_path = db_path
        self._initialize_database()

    def _initialize_database(self):
        """Create database tables for knowledge storage"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS knowledge_items (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                content TEXT NOT NULL,
                summary TEXT,
                keywords TEXT,
                category TEXT,
                sentiment TEXT,
                embedding TEXT,
                created_at TEXT,
                relevance_score REAL DEFAULT 0.0
            )
        ''')
        conn.commit()
        conn.close()

    def generate_embedding(self, text: str) -> List[float]:
        """Generate embeddings using Google GenAI API"""
        try:
            result = genai.embed_content(
                model="models/embedding-001",
                content=text,
                task_type="RETRIEVAL_DOCUMENT"
            )
            return result['embedding']
        except Exception as e:
            logger.error(f"Error generating embedding: {e}")
            return []

    def extract_keywords(self, text: str) -> List[str]:
        """Extract keywords from text using Gemini"""
        try:
            response = self.model.generate_content(f"Extract 5-10 important keywords from the following text. Return only the keywords separated by commas:\n\n{text}")
            keywords = [kw.strip() for kw in response.text.split(',')]
            return keywords
        except Exception as e:
            logger.error(f"Error extracting keywords: {e}")
            return []

    def generate_summary(self, text: str) -> str:
        """Generate summary using Gemini"""
        try:
            response = self.model.generate_content(f"Provide a concise summary of the following text in 2-3 sentences:\n\n{text}")
            return response.text.strip()
        except Exception as e:
            logger.error(f"Error generating summary: {e}")
            return text[:200] + "..."

    def categorize_content(self, text: str) -> str:
        """Categorize content using Gemini"""
        try:
            response = self.model.generate_content(f"Categorize the following text into one of these categories: Technology, Business, Science, Education, Health, Finance, Legal, General. Return only the category name:\n\n{text}")
            return response.text.strip()
        except Exception as e:
            logger.error(f"Error categorizing content: {e}")
            return "General"

    def analyze_sentiment(self, text: str) -> str:
        """Analyze the sentiment of the content using Gemini"""
        try:
            response = self.model.generate_content(f"Analyze the sentiment of the following text. Respond with only one word: Positive, Negative, or Neutral.\n\n{text}")
            return response.text.strip()
        except Exception as e:
            logger.error(f"Error analyzing sentiment: {e}")
            return "Neutral"

    def add_knowledge(self, content: str) -> Dict[str, str]:
        """Add new knowledge to the system"""
        try:
            summary = self.generate_summary(content)
            keywords = self.extract_keywords(content)
            category = self.categorize_content(content)
            sentiment = self.analyze_sentiment(content)
            embedding = self.generate_embedding(content)

            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            cursor.execute('''
                INSERT INTO knowledge_items (content, summary, keywords, category, sentiment, embedding, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            ''', (content, summary, json.dumps(keywords), category, sentiment, json.dumps(embedding), datetime.now().isoformat()))
            conn.commit()
            item_id = cursor.lastrowid
            conn.close()

            return {
                "status": "success",
                "message": f"Knowledge item added successfully with ID: {item_id}",
                "id": str(item_id),
                "summary": summary,
                "category": category,
                "sentiment": sentiment,
                "keywords": ", ".join(keywords)
            }
        except Exception as e:
            logger.error(f"Error adding knowledge: {e}")
            return {"status": "error", "message": f"Failed to add knowledge: {str(e)}"}

    def search_knowledge(self, query: str, top_k: int = 5) -> List[KnowledgeItem]:
        """Search knowledge using semantic similarity"""
        try:
            query_embedding = genai.embed_content(model="models/embedding-001", content=query, task_type="RETRIEVAL_QUERY")['embedding']
            if not query_embedding:
                return []

            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            cursor.execute('SELECT * FROM knowledge_items')
            rows = cursor.fetchall()
            conn.close()

            knowledge_items = []
            for row in rows:
                item_embedding = json.loads(row[6]) if row[6] else []
                if item_embedding:
                    similarity = cosine_similarity([query_embedding], [item_embedding])[0][0]
                    item = KnowledgeItem(
                        id=row[0], content=row[1], summary=row[2],
                        keywords=json.loads(row[3]) if row[3] else [],
                        category=row[4], sentiment=row[5], embedding=item_embedding,
                        created_at=row[7], relevance_score=float(similarity)
                    )
                    knowledge_items.append(item)

            knowledge_items.sort(key=lambda x: x.relevance_score, reverse=True)
            return knowledge_items[:top_k]
        except Exception as e:
            logger.error(f"Error searching knowledge: {e}")
            return []

    def get_knowledge_stats(self) -> Dict[str, int]:
        """Get statistics about the knowledge base"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            cursor.execute('SELECT COUNT(*) FROM knowledge_items')
            total_items = cursor.fetchone()[0]
            cursor.execute('SELECT category, COUNT(*) FROM knowledge_items GROUP BY category')
            category_counts = dict(cursor.fetchall())
            cursor.execute('SELECT sentiment, COUNT(*) FROM knowledge_items GROUP BY sentiment')
            sentiment_counts = dict(cursor.fetchall())
            conn.close()
            return {"total_items": total_items, "categories": category_counts, "sentiments": sentiment_counts}
        except Exception as e:
            logger.error(f"Error getting stats: {e}")
            return {"total_items": 0, "categories": {}, "sentiments": {}}

    def generate_answer(self, query: str, context_items: List[KnowledgeItem]) -> str:
        """Generate answer using retrieved knowledge and chat history"""
        if not context_items:
            response = self.chat.send_message(f"Answer this question: {query}")
            return response.text

        context = "\n\n".join([f"Content: {item.content}\nSummary: {item.summary}" for item in context_items[:3]])
        try:
            response = self.chat.send_message(
                f"Based on the following context from the knowledge base, please provide a comprehensive answer to the user's question. If the context isn't sufficient, say so but still try to answer the question from your own knowledge.\n\nContext:\n{context}\n\nQuestion: {query}"
            )
            return response.text
        except Exception as e:
            logger.error(f"Error generating answer: {e}")
            return f"Error generating answer: {str(e)}"

# --- Gradio Interface Functions ---
knowledge_system = None

def initialize_system() -> str:
    """Initialize the knowledge system with API key"""
    global knowledge_system
    if not api_key:
        return "❌ API Key not found. Please set the GOOGLE_API_KEY environment variable in your Space secrets."
    try:
        knowledge_system = KnowledgeInformationSystem(api_key)
        return "✅ Knowledge system initialized successfully with Gemini 1.5 Flash!"
    except Exception as e:
        return f"❌ Error initializing system: {str(e)}"

def add_knowledge_item(content: str) -> Tuple[str, str, str, str, str]:
    """Add knowledge item through Gradio interface"""
    if not knowledge_system:
        return "❌ Please initialize the system first", "", "", "", ""
    if not content.strip():
        return "❌ Please enter some content", "", "", "", ""
    result = knowledge_system.add_knowledge(content)
    if result["status"] == "success":
        return (result["message"], result["summary"], result["category"], result["sentiment"], result["keywords"])
    else:
        return result["message"], "", "", "", ""

def search_and_answer(query: str, history: List[List[str]]) -> Tuple[str, List[List[str]]]:
    """Search knowledge base and generate answer"""
    if not knowledge_system:
        history.append([query, "❌ System not initialized. Please go to the 'Setup' tab and initialize the system."])
        return "", history
    if not query.strip():
        history.append([query, "❌ Please enter a search query."])
        return "", history
    results = knowledge_system.search_knowledge(query, top_k=3)
    answer = knowledge_system.generate_answer(query, results)
    history.append([query, answer])
    return "", history

def get_system_statistics() -> str:
    """Get knowledge base statistics"""
    if not knowledge_system:
        return "❌ Please initialize the system first"
    stats = knowledge_system.get_knowledge_stats()
    stats_text = f"📊 Knowledge Base Statistics\n\nTotal Items: {stats['total_items']}\n\n"
    if stats['categories']:
        stats_text += "Categories:\n" + "\n".join([f"- {category}: {count}" for category, count in stats['categories'].items()])
    if stats['sentiments']:
        stats_text += "\n\nSentiments:\n" + "\n".join([f"- {sentiment}: {count}" for sentiment, count in stats['sentiments'].items()])
    return stats_text

# --- Main Gradio Interface Block ---
def create_gradio_interface():
    """Create and configure Gradio interface"""
    with gr.Blocks(title="Knowledge Recommendation Engine", theme=gr.themes.Soft()) as demo:
        gr.Markdown("# 🧠 Knowledge Recommendation Engine (Gemini 1.5 Flash)")
        gr.Markdown("An intelligent knowledge management system powered by Google's Gemini and embeddings.")

        with gr.Tab("Setup"):
            gr.Markdown("## Initialize System")
            gr.Markdown("Your Google AI API key is loaded from your Space secrets. Click the button to start.")
            init_button = gr.Button("Initialize System", variant="primary")
            init_status = gr.Textbox(label="Status", interactive=False)
            init_button.click(initialize_system, inputs=[], outputs=[init_status])

        with gr.Tab("Add Knowledge"):
            gr.Markdown("## Add New Knowledge")
            content_input = gr.Textbox(label="Content", placeholder="Enter the knowledge content...", lines=8)
            add_button = gr.Button("Add Knowledge", variant="primary")
            with gr.Row():
                with gr.Column():
                    add_status = gr.Textbox(label="Status", interactive=False)
                    generated_summary = gr.Textbox(label="Generated Summary", interactive=False)
                with gr.Column():
                    generated_category = gr.Textbox(label="Category", interactive=False)
                    generated_sentiment = gr.Textbox(label="Sentiment", interactive=False)
            generated_keywords = gr.Textbox(label="Keywords", interactive=False)
            add_button.click(add_knowledge_item, inputs=[content_input], outputs=[add_status, generated_summary, generated_category, generated_sentiment, generated_keywords])

        with gr.Tab("Search & Query"):
            gr.Markdown("## Chat with your Knowledge Base")
            chatbot = gr.Chatbot(label="Conversation", height=500)
            msg = gr.Textbox(label="Your Question", placeholder="Ask a question...")
            clear = gr.Button("Clear Chat History")
            msg.submit(search_and_answer, [msg, chatbot], [msg, chatbot])
            clear.click(lambda: None, None, chatbot, queue=False)

        with gr.Tab("Statistics"):
            gr.Markdown("## Knowledge Base Statistics")
            stats_button = gr.Button("Refresh Statistics", variant="secondary")
            stats_output = gr.Textbox(label="Statistics", lines=12, interactive=False)
            stats_button.click(get_system_statistics, outputs=[stats_output])
    return demo

# MODIFIED FOR HUGGING FACE: The launch command is simplified.
if __name__ == "__main__":
    demo = create_gradio_interface()
    demo.launch()
