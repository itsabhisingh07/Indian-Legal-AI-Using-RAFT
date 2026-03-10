# Privacy-Preserving Legal AI for the Indian Context 🇮🇳

An offline, fully localized Retrieval-Augmented Generation (RAG) architecture powered by a Fine-Tuned Llama-3 (8B) model. This system provides highly accurate, jurisdictionally compliant legal assistance based on the **Bharatiya Nyaya Sanhita (BNS)**, the **Indian Penal Code (IPC)**, and the **Constitution of India**.

---

##  Overview
Commercial Large Language Models (LLMs) pose severe data privacy risks when handling confidential legal case files. Furthermore, generalized models suffer from "jurisdictional bias" (defaulting to US/UK laws) and are highly prone to hallucinating fictitious statutes. 

This project solves these vulnerabilities by deploying a **100% offline**, privacy-first RAG pipeline. The foundational model was instruction-tuned using Parameter-Efficient Fine-Tuning (PEFT) on a custom dataset of 1,100+ Indian legal queries to natively comprehend regional terminology.

##  Key Features
* **Absolute Data Privacy:** The entire inference loop (LLM + Vector DB) runs locally. No case facts are ever transmitted to third-party cloud APIs.
* **Mathematical Hallucination Shield:** Implements a strict `< 0.85` distance threshold during K-Nearest Neighbor (KNN) vector search. If a user asks an out-of-bounds or non-legal query, the system deterministically blocks the generation to prevent hallucinations.
* **Semantic Query Expansion:** Intercepts conversational prompts (e.g., *"What is Article 5?"*) and maps them to dense statutory context prior to vectorization, completely bridging the semantic gap.
* **Zero Western Bias:** Fine-tuned explicitly to recognize Indian jurisprudence, overcoming the standard biases found in base open-source models.

##  Technology Stack
* **Foundational Model:** Llama-3 (8B)
* **Fine-Tuning:** PEFT / LoRA (Trained on Google Colab GPUs)
* **Vector Database:** ChromaDB (Persistent Local Storage)
* **Dataset Generation:** Groq & Gemini Pro Inference Endpoints
* **Evaluation Framework:** LLM-as-a-Judge

##  Performance Metrics
The architecture was rigorously evaluated against a "Golden Dataset" of 20 complex jurisdictional collisions and out-of-bounds queries:
* **Retrieval Accuracy:** `95%` (Successfully anchored to exact BNS/Constitutional clauses).
* **Hallucination Blocking Rate:** `75%` (Successfully intercepted out-of-domain prompts like drone regulations and recipes).
* **Latency Trade-off:** Added merely ~1.0 second of compute overhead for strict mathematical safety.

##  Local Setup & Installation
*(Note: Model weights and the compiled ChromaDB database are not included in this repository due to size constraints.)*

1. Clone the repository:
   ```bash
   git clone [https://github.com/itsabhisingh07/Indian-Legal-AI-Using-RAFT.git](https://github.com/itsabhisingh07/Indian-Legal-AI-Using-RAFT.git)

   Install the required dependencies:

    pip install -r requirements.txt
    Place your raw legal PDFs (BNS, IPC, Constitution) into the data/ directory.

    Run the ingestion script to build your local vector database:

    python scripts/ingest_data.py
    Launch the local inference pipeline


## Academic Research
A formal research paper detailing this architecture, titled "Privacy-Preserving Legal Assistance: An Offline RAG-Based Large Language Model for the Indian Legal Context", is currently undergoing peer review for academic publication.
