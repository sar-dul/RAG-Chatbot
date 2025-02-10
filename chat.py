from langchain_ollama import ChatOllama

from langchain.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma
from langchain_community.retrievers import BM25Retriever
from langchain.retrievers import EnsembleRetriever 

from langchain_core.prompts import MessagesPlaceholder
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import create_history_aware_retriever
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain

from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory

from langchain_core.tools import tool
import datetime

import warnings
import logging


# Suppress LangChain deprecation warnings
warnings.filterwarnings("ignore")

# Suppress httpx logging
logging.getLogger("httpx").setLevel(logging.WARNING)

# Initialize the Ollama LLM
model = ChatOllama(model="llama3.2")

loader = PyPDFLoader("leave.pdf")
docs = loader.load()
text_splitter = RecursiveCharacterTextSplitter(chunk_size=300, chunk_overlap=50)
all_splits = text_splitter.split_documents(docs)

embeddings = OllamaEmbeddings(model="llama3.2")
### Vector Search and bm25 : Hybrid Retreiver
vector_store = Chroma(embedding_function=embeddings)
_ = vector_store.add_documents(documents=all_splits)

vector_retriever = vector_store.as_retriever(type = "similarity", search_kwargs = {"k" : 5})
bm25_retriever = BM25Retriever.from_documents(all_splits)
bm25_retriever.k = 5
ensemble_retriever = EnsembleRetriever(retrievers=[bm25_retriever, vector_retriever],weights=[0.4, 0.6])


system_prompt = (
    "You are an assistant for question-answering tasks. "
    "Use the following pieces of retrieved context to answer "
    "the question. If you don't know the answer, say that you "
    "don't know. Use three sentences maximum and keep the "
    "answer concise."
    "\n\n"
    "{context}"
)
contextualize_q_system_prompt = (
    "Given a chat history and the latest user question "
    "which might reference context in the chat history, "
    "formulate a standalone question which can be understood "
    "without the chat history. Do NOT answer the question, "
    "just reformulate it if needed and otherwise return it as is."
)

contextualize_q_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", contextualize_q_system_prompt),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ]
)
history_aware_retriever = create_history_aware_retriever(
    model, ensemble_retriever, contextualize_q_prompt
)

qa_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system_prompt),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ]
)

question_answer_chain = create_stuff_documents_chain(model, qa_prompt)

rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)
store = {}


def get_session_history(session_id: str) -> BaseChatMessageHistory:
    if session_id not in store:
        store[session_id] = ChatMessageHistory()
    return store[session_id]


conversational_rag_chain = RunnableWithMessageHistory(
    rag_chain,
    get_session_history,
    input_messages_key="input",
    history_messages_key="chat_history",
    output_messages_key="answer",
)

# Assuming conversational_rag_chain is already defined and configured

def chatbot_conversation(session_id: str = None):
    if session_id is None:
        session_id = str(datetime.datetime.now())
    print("Chatbot: Hello! How can I assist you today?")
    
    while True:
        user_input = input("You: ")
        
        if user_input.lower() in ["exit", "quit", "bye"]:
            print("Chatbot: Goodbye! Have a great day!")
            break
        
        # Invoke the conversational RAG chain with the user's input
        response = conversational_rag_chain.invoke(
            {"input": user_input},
            config={"configurable": {"session_id": session_id}},
        )["answer"]
        
        # Display the user's question and the chatbot's response
        print(f"You: {user_input}")
        print(f"Chatbot: {response}\n")
