# 🍎 Lucfin Nutrition Assistant

> **An Adaptive Multimodal RAG System for Vietnamese Cuisine Nutrition Analysis.**

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue?logo=python)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.109-009688?logo=fastapi)](https://fastapi.tiangolo.com/)
[![RAG Architecture](https://img.shields.io/badge/RAG-Advanced-orange)]()
[![Mobile](https://img.shields.io/badge/Client-Android-green?logo=android)](https://www.android.com/)

**Lucfin** is not just a chatbot; it's an intelligent nutrition consultant powered by **Adaptive RAG (Retrieval-Augmented Generation)**. It combines Computer Vision (YOLO) for food recognition with a high-performance Vector Database to provide accurate, scientifically-backed nutritional insights for Vietnamese dishes.

---

## 🏗️ System Architecture

Lucfin operates on a **3-Layer Pipeline** architecture designed to optimize latency and accuracy:

1.  **Layer 1 (Perception):** Android Client captures food images via Camera.
2.  **Layer 2 (Orchestration):** A **Semantic Router** classifies user intent (Vision Follow-up vs. Deep Knowledge vs. Chitchat).
3.  **Layer 3 (Cognition):**
    * 🔥 **Fast Track:** Direct injection of YOLO detection results.
    * 💧 **Deep Track:** Hybrid Retrieval (Vector + Keyword) + Cross-Encoder Re-ranking.
    * 🛡️ **Safety Track:** Guardrails for hallucination prevention.

<p align="center">
  <img src="images/adative_architectureRAG.png" alt="Lucfin 3-Layer Architecture" width="100%">
  <br>
  <em>Figure 1: The Adaptive Multimodal RAG Pipeline of Lucfin</em>
</p>

---

## 🚀 Key Features & Demo

We don't just claim it works; here is the proof of our **Backend (Uvicorn Logs)** synchronizing with the **Frontend (Android App)**.

### 1. 👁️ Visual Intelligence (AI Camera)
**Capability:** Instantly recognizes Vietnamese dishes from images and injects context into the conversation without searching the database.

| Backend Execution (Uvicorn Log)
|:---:|
| <img src="images/CvDetect.png" width="400"> 
| *Log evidence: `YOLO detected: Cơm hến`* |

### 2. 🧠 Deep Nutrition Knowledge (RAG)
**Capability:** answers complex nutritional queries (Calories, Protein, Fat) by retrieving and re-ranking documents from the internal Knowledge Base.

| Backend Execution (Uvicorn Log)
|:---:|
| <img src="images/deepRAG.png" width="400"> 
| *Log evidence: `Retrieving nodes...`, `Re-ranker score: 0.95`* |

### 3. 💬 Smart Routing & Chitchat: case1: check_gaurdrail
**Capability:** The **Semantic Router** distinguishes between technical queries and casual conversation to provide natural responses.

| Backend Execution (Uvicorn Log)
|:---:|
| <img src="images/trungKL.png" width="400"> |
| *Log evidence: `Intent: CHITCHAT`, `Routing to: Safety Pipeline`* | *Natural, human-like interaction.* |

---
### 4. 💬 Smart Routing & Chitchat: case2: check_unreliable_content
**Capability:** The **Semantic Router** distinguishes between technical queries and casual conversation to provide natural responses.

| Backend Execution (Uvicorn Log)
|:---:|
| <img src="images/weather.png" width="400"> 
| *Log evidence: `Intent: CHITCHAT`, `Routing to: Safety Pipeline`* |

---
### 5. 💬 Smart Routing & Chitchat: case3: check_normal_speaking
**Capability:** The **Semantic Router** distinguishes between technical queries and casual conversation to provide natural responses.

| Backend Execution (Uvicorn Log)
|:---:|
| <img src="images/chaohoi.png" width="400"> 
| *Log evidence: `Intent: CHITCHAT`, `Routing to: Safety Pipeline`* |

---

## 🛠️ Technology Stack

* **Core AI Engine:**
    * **LLM:** Qwen-3-32B (via Groq API) for high-fidelity generation.
    * **Embeddings:** `AITeamVN/Vietnamese_Embedding` (State-of-the-art for Vietnamese).
    * **Vision:** YOLOv11 custom trained on Vietnamese Food Dataset.
* **RAG Infrastructure:**
    * **Vector Store:** ChromaDB (Local & Efficient).
    * **Orchestration:** LangChain & LlamaIndex.
    * **Re-ranking:** Cross-Encoder for Context Precision.
* **Backend:** FastAPI (Python), Uvicorn.
* **Client:** Native Android (Java/Kotlin).

---

## 📊 Performance Evaluation

The system was evaluated using the **Ragas Framework** (LLM-as-a-Judge) on a test set of 30 QA pairs:

* ✅ **Answer Relevancy:** **~95.7%** (High intent understanding).
* ✅ **Faithfulness:** **~94.5%** (Minimizes hallucination via Guardrails).
* ✅ **Context Precision:** **~91.2%** (Accurate retrieval).

---

## ⚙️ Installation & Setup

### Prerequisites
* Python 3.10+
* CUDA-enabled GPU (Optional, for faster Re-ranking)
* Groq API Key

### Steps

1.  **Clone the repository**
    ```bash
    git clone https://github.com/Phuc75nguyen/LucfinChatbot.git
    cd LucfinChatbot
    ```

2.  **Install dependencies**
    ```bash
    python -m venv .venv
    source .venv/bin/activate  # On Windows: .venv\Scripts\activate
    pip install -r requirements.txt
    ```

3.  **Setup Environment Variables**
    Create a `.env` file:
    ```env
    GROQ_API_KEY=gsk_your_key_here
    ```

4.  **Build Vector Index (First run only)**
    ```bash
    python build_index.py
    ```

5.  **Run the Server**
    ```bash
    python main.py
    ```
    *Server will start at `http://0.0.0.0:8000`*

---

## 👨‍💻 Author

**[Nguyen Tan Phuc]**
* Role: AI Engineer & System Architect
* Contact: phuc75nguyen@gmail.com
---
*Built with Team Phuc-Linh❤️ and lots of coffee.*