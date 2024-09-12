import streamlit as st
import os
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.prompts import PromptTemplate

from langchain_core.output_parsers import StrOutputParser

from langchain_core.runnables import RunnablePassthrough
from langchain_qdrant import QdrantVectorStore
import Raptor
from io import StringIO
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams

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

@st.cache_resource
def load_chromadb(collection_name):
    client = QdrantClient(
    url="https://da9fadd2-dc5a-4481-ac79-4e2677a2354b.europe-west3-0.gcp.cloud.qdrant.io", 
    api_key="X_-IVToBM07Mot4Mmzg5xNjYzc1DlIgl0VQDUNmGhI_Z-WA5FJ2ETA" 
)

    client.recreate_collection(
        collection_name=collection_name, 
        vectors_config=VectorParams(size=768, distance=Distance.COSINE) 
    )
    db = QdrantVectorStore(
        client=client,
        collection_name=collection_name,
        embedding=st.session_state.embd,
    )
    return db

@st.cache_resource
def update_chromadb(collection_name):
    client = QdrantClient(
    url="https://da9fadd2-dc5a-4481-ac79-4e2677a2354b.europe-west3-0.gcp.cloud.qdrant.io", 
    api_key="X_-IVToBM07Mot4Mmzg5xNjYzc1DlIgl0VQDUNmGhI_Z-WA5FJ2ETA" 
    )
    
    try:
        client.delete_collection(collection_name=collection_name)
    except Exception as e:
        print(f"Warning: {e}")

    client.recreate_collection(
        collection_name=collection_name, 
        vectors_config=VectorParams(size=768, distance=Distance.COSINE) 
    )
    db = QdrantVectorStore(
        client=client,
        collection_name=collection_name,
        embedding=st.session_state.embd,
    )
    return db

if "vector_store" not in st.session_state:
    st.session_state.vector_store = load_chromadb(f"data{st.session_state.num}")

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

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

@st.cache_resource
def rag_chain(_model, _vectorstore):
    retriever = _vectorstore.as_retriever()
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
    rag = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | _model
        | StrOutputParser()
    )
    return rag

if st.session_state.model is not None and st.session_state.vector_store is not None:
    st.session_state.rag = rag_chain(st.session_state.model, st.session_state.vector_store)

if "new_docs" not in st.session_state:
    st.session_state.new_docs = False

with st.sidebar:
    uploaded_files = st.file_uploader("Chọn file txt", accept_multiple_files=True, type=["txt"])   
    if st.session_state.model:
        documents = []
        uploaded_file_names = set()
        if uploaded_files:
            for uploaded_file in uploaded_files:
                uploaded_file_names.add(uploaded_file.name)      
        if uploaded_file_names != st.session_state.uploaded_files and not st.session_state.new_docs:
            st.session_state.uploaded_files = uploaded_file_names
            st.session_state.new_docs = True
            if uploaded_files:
                for uploaded_file in uploaded_files: 
                    stringio=StringIO(uploaded_file.getvalue().decode('utf-8'))
                    read_data=str(stringio.read())
                    documents.append(read_data)
                
def update_rag_chain(_model, _embd, _vectorstore, docs_texts):
    results = Raptor.recursive_embed_cluster_summarize(_model, _embd, docs_texts, level=1, n_levels=3)
    all_texts = docs_texts.copy()
    for level in sorted(results.keys()):
        summaries = results[level][1]["summaries"].tolist()
        all_texts.extend(summaries)
    _vectorstore.add_texts(texts=all_texts)
    rag = rag_chain(_model, _vectorstore)
    return rag

def reset_rag_chain(_model, _vectorstore):
    rag = rag_chain(_model, _vectorstore)
    return rag

if "query_router" not in st.session_state:
    st.session_state.query_router = None  

@st.cache_resource
def query_router(_model):
    mess = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                """Bạn là một chatbot hỗ trợ giải đáp về đại học, nhiệm vụ của bạn là phân loại câu hỏi. 
                Nếu câu hỏi về đại học thì trả về 'university', nếu không liên quan tới tuyển sinh và sinh viên thì trả về 'other'. 
                Bắt buộc Kết quả chỉ trả về với một trong hai lựa chọn trên. 
                Không được trả lời thêm bất kỳ thông tin nào khác.""",
            ),
            ("human", "{input}"),
        ]
    )
    chain = mess | _model
    return chain

if st.session_state.model is not None:
    st.session_state.query_router = query_router(st.session_state.model)

@st.dialog("Update DB")
def update_vectorstore(_model, _embd, _vectorstore, docs):
    docs_texts = [d for d in docs]
    st.session_state.rag = update_rag_chain(_model, _embd, _vectorstore, docs_texts)   
    st.rerun()  
    
@st.dialog("Reset DB")
def reset_vectorstore(_model, _vectorstore):
    st.session_state.rag = reset_rag_chain(_model, _vectorstore) 
    st.rerun()

if st.session_state.new_docs:
    st.session_state.new_docs = False
    st.session_state.vector_store = update_chromadb("data")
    if st.session_state.uploaded_files:
        update_vectorstore(st.session_state.model, st.session_state.embd, st.session_state.vector_store, documents)
    else:
        reset_vectorstore(st.session_state.model, st.session_state.vector_store)
    
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
            router = st.session_state.query_router.invoke(prompt)
            switch = router.content
            if "university" in switch:
                respone = st.session_state.rag.invoke(prompt)
                f_response = f"RAG: {respone}"
                st.write(f_response)
            else: 
                respone = st.session_state.llm.invoke(prompt)
                f_response = f"other: {respone.content}"
                st.write(f_response)
                
        st.session_state.chat_history.append({"role": "assistant", "content": f_response})

