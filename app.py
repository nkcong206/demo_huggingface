import streamlit as st
import os
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.document_loaders import TextLoader
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.prompts import PromptTemplate

from langchain_core.output_parsers import StrOutputParser

from langchain_core.runnables import RunnablePassthrough
from langchain_chroma import Chroma
import Raptor

page = st.title("Chat with AskUSTH")

if "gemini_api" not in st.session_state:
    st.session_state.gemini_api = None

if "rag" not in st.session_state:
    st.session_state.rag = None
    
if "llm" not in st.session_state:
    st.session_state.llm = None

@st.cache_resource
def get_chat_google_model(api_key):
    os.environ["GOOGLE_API_KEY"] = api_key
    return ChatGoogleGenerativeAI(
        model="gemini-1.5-flash",
        temperature=0,
        max_tokens=None,
        timeout=None,
        max_retries=2,
    )

@st.cache_resource
def get_embedding_model():
    model_name = "bkai-foundation-models/vietnamese-bi-encoder"
    model_kwargs = {'device': 'cpu'}
    encode_kwargs = {'normalize_embeddings': False}
    
    model = HuggingFaceEmbeddings(
        model_name=model_name,
        model_kwargs=model_kwargs,
        encode_kwargs=encode_kwargs
    )   
    return model

if "embd" not in st.session_state:
    st.session_state.embd = get_embedding_model()

if "model" not in st.session_state:
    st.session_state.model = None

if "save_dir" not in st.session_state:
    st.session_state.save_dir = None 

if "uploaded_files" not in st.session_state:
    st.session_state.uploaded_files = set()
    
@st.dialog("Setup Gemini")
def vote():
    st.markdown(
        """
        Để sử dụng Google Gemini, bạn cần cung cấp API key. Tạo key của bạn [tại đây](https://ai.google.dev/gemini-api/docs/get-started/tutorial?lang=python&hl=vi) và dán vào bên dưới.
        """
    )
    key = st.text_input("Key:", "")
    if st.button("Save") and key != "":
        st.session_state.gemini_api = key
        st.rerun()  

if st.session_state.gemini_api is None:
    vote()

if st.session_state.gemini_api and st.session_state.model is None:
    st.session_state.model = get_chat_google_model(st.session_state.gemini_api)

if st.session_state.save_dir is None:
    save_dir = "./Documents"
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    st.session_state.save_dir = save_dir
    
def load_txt(file_path):
    loader_sv = TextLoader(file_path=file_path, encoding="utf-8")
    doc = loader_sv.load()
    return doc

with st.sidebar:
    uploaded_files = st.file_uploader("Chọn file txt", accept_multiple_files=True, type=["txt"])   
    if st.session_state.gemini_api:
        if uploaded_files:
            documents = []
            uploaded_file_names = set()
            new_docs = False
            for uploaded_file in uploaded_files:
                uploaded_file_names.add(uploaded_file.name)
                if uploaded_file.name not in st.session_state.uploaded_files:
                    file_path = os.path.join(st.session_state.save_dir, uploaded_file.name)
                    with open(file_path, mode='wb') as w:
                        w.write(uploaded_file.getvalue())
                else:
                    continue

                new_docs = True
                
                doc = load_txt(file_path)
                
                documents.extend([*doc])
                
            if new_docs:  
                st.session_state.uploaded_files = uploaded_file_names
                st.session_state.rag = None
        else:
            st.session_state.uploaded_files = set()
            st.session_state.rag = None

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

@st.cache_resource
def compute_rag_chain(_model, _embd, docs_texts):
    results = Raptor.recursive_embed_cluster_summarize(_model, _embd, docs_texts, level=1, n_levels=3)
    all_texts = docs_texts.copy()
    i = 0
    for level in sorted(results.keys()):
        summaries = results[level][1]["summaries"].tolist()
        all_texts.extend(summaries)
        print(f"summary {i} -------------------------------------------------")
        print(summaries)
        i += 1
    print("all_texts ______________________________________")
    print(all_texts)
    vectorstore = Chroma.from_texts(texts=all_texts, embedding=_embd)
    retriever = vectorstore.as_retriever()
    template = """
        Bạn là một trợ lí AI hỗ trợ tuyển sinh và sinh viên. \n
        Hãy trả lời câu hỏi chính xác, tập trung vào thông tin liên quan đến câu hỏi. \n
        Nếu bạn không biết câu trả lời, hãy nói không biết, đừng cố tạo ra câu trả lời.\n
        Dưới đây là thông tin liên quan mà bạn cần sử dụng tới:\n
        {context}\n
        hãy trả lời:\n
        {question}
        """
    prompt = PromptTemplate(template=template, input_variables=["context", "question"])
    rag_chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | _model
        | StrOutputParser()
    )
    return rag_chain

@st.dialog("Setup RAG")
def load_rag():
    docs_texts = [d.page_content for d in documents]
    st.session_state.rag = compute_rag_chain(st.session_state.model, st.session_state.embd, docs_texts)
    st.rerun()  

if st.session_state.uploaded_files and st.session_state.model is not None:
    if st.session_state.rag is None:
        load_rag()

if st.session_state.model is not None:
    if st.session_state.llm is None:
        mess = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    "Bản là một trợ lí AI hỗ trợ tuyển sinh và sinh viên",
                ),
                ("human", "{input}"),
            ]
        )
        chain = mess | st.session_state.model
        st.session_state.llm = chain

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

for message in st.session_state.chat_history:
    with st.chat_message(message["role"]):
        st.write(message["content"])

prompt = st.chat_input("Bạn muốn hỏi gì?")
if st.session_state.model is not None:
    if prompt:
        st.session_state.chat_history.append({"role": "user", "content": prompt})
            
        with st.chat_message("user"):
            st.write(prompt)
            
        with st.chat_message("assistant"):
            if st.session_state.rag is not None:
                respone = st.session_state.rag.invoke(prompt)
                st.write(respone)
            else: 
                ans = st.session_state.llm.invoke(prompt)
                respone = ans.content
                st.write(respone)
                
        st.session_state.chat_history.append({"role": "assistant", "content": respone})

