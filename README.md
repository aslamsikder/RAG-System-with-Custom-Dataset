# RAG-System-with-Custom-Dataset

## RAG System for Answering the Question with related Explanation

### Project Overview
This project implements a modular question answering system using LangChain, OpenAI GPT models, and Chroma vector embeddings. Users can ask questions in Bangla or English, and the system retrieves relevant context from a CSV dataset, then generates step-by-step answers using an LLM.

Additionally, the project automatically detects the most relevant document for retrieval evaluation and computes the following metrics:

* Hit@1, Hit@3, Hit@5
* Mean Reciprocal Rank (MRR)

The app uses Streamlit for an interactive frontend.

### Features
* Bangla QA: Supports Bangla questions.
* Vector-based Retrieval: Uses Chroma vector store and OpenAI embeddings for context retrieval.
* Step-by-step Explanations: LLM generates detailed answers.
* Automatic Retrieval Metrics: Detects the most similar document as ground truth and calculates Hit@1, Hit@3, Hit@5, MRR.
* Fully Modular: Separate modules for configuration, data loading, embeddings, chain building, and metrics.
* Interactive Frontend: Users type questions and immediately see answers and metrics.

## ⚙️ How to Run

### 1. Create Environment
```bash
conda create -n RAGSystem python=3.11 -y
conda activate RAGSystem
```

### 2. Install Requirements
```bash
pip install -r requirements.txt
```
### 3. Set API Key (Keep your all API in .env file) - best approach
Create a `.env` file in the project root:
```
OPENAI_API_KEY = "your_key_here"
```

### 4. Run the App
```bash
streamlit run src/app.py

streamlit run streamlit_app.py
```

---

## Project Structure

```
RAG-System-with-Custom-Dataset/
├──data/
|   └── questions.csv       # CSV dataset
├── research/
│   └── experiment.ipynb    # First I explored the full project here before writing moduler structure code
├── src/                    # Start Modular Coding
│   ├─ __init__.pt.py
│   ├─ app.py               # Main Streamlit app with auto metrics
│   ├─ config.py            # Environment setup
│   ├─ data_loader.py       # Load CSV & split documents
│   ├─ embeddings.py        # Create vector store
│   ├─ chain_builder.py     # LLM + retrieval chain
│   ├─ metrics.py           # Hit@k and MRR evaluation         
└── LICENSE
└── README.md
└── requirements.txt        # All required library is saved here
└── setup.py
└── streamlit_app.py        # Complete project in one file just for test purpose, after doing research  
└── .env                    # All API Key is stored here
```

---

## 🧰 Tech Stack

- `Streamlit`
- `LangChain`
- `OpenAI` 
- `ChromaDB`
---

## 📌 Example Output

| Question                                            | Answer        | Explanation                                     |
|-----------------------------------------------------|---------------|-------------------------------------------------|
| নোবেল প্রাপ্ত কাব্যগ্রন্থ 'গীতাঞ্জলি' এর ইংরেজি অনুবাদক কে? | রবীন্দ্রনাথ ঠাকুর  | 'গীতাঞ্জলি' কাব্যগ্রন্থের ইংরেজি অনুবাদক হলেন রবীন্দ্রনাথ ঠাকুর। এই অনুবাদের জন্য তিনি ১৯১৩ সালে নোবেল পুরস্কার লাভ করেন। |

---

## ✍️ Author
Developed by **Aslam Sikder**, September 2025  
Email: [aslamsikder.edu@gmail.com](mailto:aslamsikder.edu@gmail.com)  
LinkedIn: [Aslam Sikder - Linkedin](https://www.linkedin.com/in/aslamsikder)  
Google Scholar: [Aslam Sikder - Google Scholar](https://scholar.google.com/citations?hl=en&user=Ip1qQi8AAAAJ)
