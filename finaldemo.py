import streamlit as st
import tempfile
import os
from typing import List, TypedDict
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import MarkdownTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langgraph.graph import StateGraph, END

import streamlit as st
MASTER_GROQ_API_KEY = st.secrets["GROQ_API_KEY"]


CLIENT_CONFIG = {
    "APP_TITLE": "MatrixAnalyst Pro",
    "SUBTITLE": "Enterprise Multi-Agent Financial Intelligence Platform powered by LangGraph.",
    "CORE_LLM_MODEL": "openai/gpt-oss-20b",
    "MATH_TRIGGERS": ["calculate", "ratio", "margin", "percentage change", "sum", "total revenue"],
    "VALID_DEPARTMENTS": {
        "DEPT-FINANCE-2026": "FinancePass123",
        "DEPT-AUDIT-2026": "AuditPass456",
        "DEPT-EXECUTIVE-HQ": "ExecBoard789"
    }
}

class GraphState(TypedDict):
    question: str
    retriever: object
    llm: object
    context_docs: List[str]
    generation: str

@st.cache_resource(show_spinner=False)
def process_pdf(uploaded_file):
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tf:
        tf.write(uploaded_file.getbuffer())
        file_path = tf.name
    
    loader = PyPDFLoader(file_path)
    docs = loader.load()
    
    text_splitter = MarkdownTextSplitter(chunk_size=1200, chunk_overlap=250)
    splits = text_splitter.split_documents(docs)
    
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vectorstore = FAISS.from_documents(splits, embeddings)
    os.remove(file_path)
    return vectorstore.as_retriever(search_kwargs={"k": 4})

def route_question(state: GraphState):
    question = state["question"].lower()
    has_explicit_numbers = any(char.isdigit() for char in question)
    triggers = CLIENT_CONFIG["MATH_TRIGGERS"]
    
    if has_explicit_numbers and any(trigger in question for trigger in triggers):
        return "math_processor"
    if any(trigger in question for trigger in triggers):
        return "math_processor"
    return "document_retriever"

def retrieve_docs(state: GraphState):
    question = state["question"]
    retriever = state["retriever"]
    fetched_docs = retriever.invoke(question)
    doc_texts = [doc.page_content for doc in fetched_docs]
    return {"context_docs": doc_texts}

def execute_math_logic(state: GraphState):
    print("---NODE: COMPUTE PROMPT LAYER---")
    question = state["question"]
    retriever = state["retriever"]
    llm = state["llm"]
    
    # Check if the user is passing numbers directly in the prompt string
    has_explicit_numbers = any(char.isdigit() for char in question)
    
    # UPGRADE: If user gives numbers, skip vector retrieval entirely to prevent context pollution!
    if has_explicit_numbers:
        doc_texts = ["No document context required. Execute calculations using the numeric variables provided directly by the user in their prompt query sentence."]
    else:
        # Standard flow: pull raw values out of the uploaded file
        fetched_docs = retriever.invoke(question)
        doc_texts = [doc.page_content for doc in fetched_docs]
    
    math_system_prompt = (
        "You are a forensic financial auditor. Calculate metrics based strictly on numeric values found in the context or provided directly by the user.\n"
        "1. Identify the exact numbers required.\n"
        "2. State the arithmetic formula explicitly using clean, standard plain text characters (e.g., '/' and '*').\n"
        "3. CRITICAL: Do NOT use LaTeX formulas or strings like '\\frac' or '\\text'. Write math naturally as clean text.\n"
        "4. Show calculations step-by-step with the raw numbers.\n\n"
        "Context:\n{context}"
    )
    
    math_prompt = ChatPromptTemplate.from_messages([
        ("system", math_system_prompt),
        ("human", "{input}")
    ])
    
    math_chain = math_prompt | llm | StrOutputParser()
    result = math_chain.invoke({"context": "\n\n".join(doc_texts), "input": question})
    return {"generation": result, "context_docs": doc_texts}


def generate_standard_answer(state: GraphState):
    if state.get("generation"):
        return state
    question = state["question"]
    context = "\n\n".join(state["context_docs"])
    llm = state["llm"]
    
    system_prompt = (
        "You are an expert corporate financial analyst. Answer the user's question using "
        "only the provided context. If you do not know, say it is not explicitly stated.\n\nContext:\n{context}"
    )
    standard_prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", "{input}"),
    ])
    standard_chain = standard_prompt | llm | StrOutputParser()
    ai_response = standard_chain.invoke({"context": context, "input": question})
    return {"generation": ai_response}

workflow = StateGraph(GraphState)
workflow.add_node("document_retriever", retrieve_docs)
workflow.add_node("math_processor", execute_math_logic)
workflow.add_node("answer_generator", generate_standard_answer)
workflow.set_conditional_entry_point(route_question, {"math_processor": "math_processor", "document_retriever": "document_retriever"})
workflow.add_edge("document_retriever", "answer_generator")
workflow.add_edge("math_processor", "answer_generator")
workflow.add_edge("answer_generator", END)
compiled_rag_graph = workflow.compile()

if "authenticated" not in st.session_state:
    st.session_state.authenticated = False
if "auth_dept" not in st.session_state:
    st.session_state.auth_dept = ""

if not st.session_state.authenticated:
    st.set_page_config(page_title="Secure Workspace Access", layout="centered")
    st.markdown("<style>.stApp { background-color: #0B0F19 !important; color: #E2E8F0 !important; } .stButton>button { background-color: #8644FF !important; color: white !important; border-radius: 8px !important; }</style>", unsafe_allow_html=True)
    st.markdown("<h2 style='text-align: center; color: #00E5FF;'>🏢 Department Workspace Access</h2>", unsafe_allow_html=True)
    st.markdown("<p style='text-align: center; color: #94A3B8;'>Provide authorized corporate Department ID parameters to unlock processing nodes.</p>", unsafe_allow_html=True)
    
    with st.form("auth_gate"):
        dept_input = st.text_input("Department ID", placeholder="e.g., DEPT-FINANCE-2026")
        password_input = st.text_input("Security Access Password", type="password", placeholder="••••••••")
        submit_button = st.form_submit_button("Authenticate Secure Session", use_container_width=True)
        
        if submit_button:
            dept_db = CLIENT_CONFIG["VALID_DEPARTMENTS"]
            if dept_input in dept_db and dept_db[dept_input] == password_input:
                st.session_state.authenticated = True
                st.session_state.auth_dept = dept_input
                st.success("Authentication confirmed! Access Granted.")
                st.rerun()
            else:
                st.error("Invalid Department Credentials. Access Denied.")
else:
    st.set_page_config(page_title=CLIENT_CONFIG["APP_TITLE"], layout="wide")
    st.markdown("<style>.stApp { background-color: #0B0F19 !important; color: #E2E8F0 !important; } [data-testid='stSidebar'] { background-color: #161F30 !important; border-right: 1px solid #1E293B; } .stChatInputContainer { border-radius: 12px !important; border: 1px solid #00E5FF !important; background-color: #161F30 !important; } .main-title { background: linear-gradient(90deg, #00E5FF, #8644FF); -webkit-background-clip: text; -webkit-text-fill-color: transparent; font-weight: 800; font-size: 2.8rem; margin-bottom: 0rem; } .sub-title { color: #94A3B8; font-size: 1.1rem; font-style: italic; margin-top: -0.5rem; margin-bottom: 2rem; } .profile-box { background-color: #1E293B; padding: 12px; border-radius: 8px; border-left: 4px solid #00E5FF; margin-bottom: 15px; font-family: monospace; }</style>", unsafe_allow_html=True)

    with st.sidebar:
        st.markdown(f"<div class='profile-box'>🏢 <b>Active Corporate Area:</b><br>{st.session_state.auth_dept}</div>", unsafe_allow_html=True)
        st.header("📁 Corporate Data Source")
        uploaded_file = st.file_uploader("Upload Company PDF (10-K, Annual Report)", type=["pdf"])
        
        retriever_obj = None
        if uploaded_file:
            st.success(f"Loaded: {uploaded_file.name}")
            with st.spinner("Processing Data Target Document..."):
                retriever_obj = process_pdf(uploaded_file)
        
        st.markdown("---")
        if st.button("🚪 Close Secure Session", use_container_width=True):
            st.session_state.authenticated = False
            st.session_state.auth_dept = ""
            st.rerun()

    st.markdown(f'<p class="main-title">⚡ {CLIENT_CONFIG["APP_TITLE"]}</p>', unsafe_allow_html=True)
    st.markdown(f'<p class="sub-title">{CLIENT_CONFIG["SUBTITLE"]}</p>', unsafe_allow_html=True)

    if "messages" not in st.session_state:
        st.session_state.messages = []

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    if prompt := st.chat_input("Ask a targeted query context question..."):
        with st.chat_message("user"):
            st.markdown(prompt)
        st.session_state.messages.append({"role": "user", "content": prompt})

        if not uploaded_file or retriever_obj is None:
            st.warning("A source document must be mounted inside the configuration pipeline first.")
        else:
            with st.chat_message("assistant"):
                with st.spinner("Executing Graph Pipeline Nodes..."):
                    try:
                        llm_instance = ChatGroq(api_key=MASTER_GROQ_API_KEY, model_name=CLIENT_CONFIG["CORE_LLM_MODEL"])
                        initial_state = {"question": prompt, "retriever": retriever_obj, "llm": llm_instance, "context_docs": [], "generation": ""}
                        final_output_state = compiled_rag_graph.invoke(initial_state)
                        ai_response = final_output_state["generation"]
                        st.markdown(ai_response)
                        st.session_state.messages.append({"role": "assistant", "content": ai_response})
                    except Exception as e:
                        st.error(f"Execution Error: {str(e)}")