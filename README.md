# Document Query Chatbot (RAG-Based)

This repository contains a **RAG-based Document Query Chatbot** powered by **Llama 3.2** from **Ollama**. The chatbot can retrieve answers from uploaded documents using a **hybrid retriever (vector search + BM25)**, rerank results with a **cross-encoder**, maintain **conversational history**, support **function calling**, and **store chat history in SQLite3**.

## Features
- **Hybrid Retrieval**: Uses both **BM25** and **Vector Search** to retrieve relevant document passages.
- **Cross-Encoder Reranking**: Improves retrieved results before passing them to the LLM.
- **Conversational & History-Aware**: Maintains previous interactions for contextual responses.
- **Function Calling**: Supports external API calls and tool usage within responses.
- **Chat History Storage**: Saves interactions in an **SQLite3** database for future reference.
- **Gradio UI**: Provides an interactive chat interface.

## File Structure
```
📂 chatbot-repo
├── chatbot.ipynb  # Jupyter Notebook for chatbot implementation
├── chat.py        # Functions extracted from chatbot.ipynb
├── UI.py          # Gradio-based chatbot inference UI
├── README.md      # Project documentation
```

## Setup Instructions

### 1. Install Dependencies
Ensure you have **Python 3.8+** installed. Then, install the required dependencies:
```bash
pip install -r requirements.txt
```
*(You may need to create a `requirements.txt` file listing dependencies like `gradio`, `langchain`, `ollama`, `sqlite3`, etc.)*

### 2. Install and Run Ollama
Ollama is required to run **Llama 3.2** locally. Install it using:
```bash
curl -fsSL https://ollama.com/install.sh | sh
```
Then, download **Llama 3.2**:
```bash
ollama pull llama3
```

### 3. Run the Chatbot
Start the chatbot with:
```bash
python UI.py
```
This will launch the **Gradio UI**, allowing users to interact with the chatbot.

## Usage
1. Upload documents (PDFs, text files, etc.).
2. Ask questions related to the uploaded content.
3. The chatbot retrieves and reranks the best answers.
4. View past conversations stored in SQLite3.

## Future Improvements
- Support for **more LLMs** (e.g., GPT, Mistral, Claude).
- Improved **retrieval and ranking** strategies.
- Integration with **external APIs** for function calling.
- Deployment on **Hugging Face Spaces** or **Streamlit Cloud**.

## License
This project is licensed under the **MIT License**.

---

🔥 **Developed by Sardul Ojha** 🚀
