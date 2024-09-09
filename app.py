import streamlit as st
import os
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.document_loaders import TextLoader
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.prompts import PromptTemplate

from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import umap
from langchain.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from sklearn.mixture import GaussianMixture

from langchain_core.runnables import RunnablePassthrough
from langchain_chroma import Chroma




def global_cluster_embeddings(
    embeddings: np.ndarray,
    dim: int,
    n_neighbors: Optional[int] = None,
    metric: str = "cosine",
) -> np.ndarray:
    if n_neighbors is None:
        n_neighbors = int((len(embeddings) - 1) ** 0.5)
    return umap.UMAP(
        n_neighbors=n_neighbors, n_components=dim, metric=metric
    ).fit_transform(embeddings)


def local_cluster_embeddings(
    embeddings: np.ndarray, dim: int, num_neighbors: int = 10, metric: str = "cosine"
) -> np.ndarray:
    return umap.UMAP(
        n_neighbors=num_neighbors, n_components=dim, metric=metric
    ).fit_transform(embeddings)


def get_optimal_clusters(
    embeddings: np.ndarray, max_clusters: int = 50, random_state: int = 200
) -> int:
    max_clusters = min(max_clusters, len(embeddings))
    n_clusters = np.arange(1, max_clusters)
    bics = []
    for n in n_clusters:
        gm = GaussianMixture(n_components=n, random_state=random_state)
        gm.fit(embeddings)
        bics.append(gm.bic(embeddings))
    return n_clusters[np.argmin(bics)]


def GMM_cluster(embeddings: np.ndarray, threshold: float, random_state: int = 0):
    n_clusters = get_optimal_clusters(embeddings, random_state = 200)
    gm = GaussianMixture(n_components=n_clusters, random_state=random_state)
    gm.fit(embeddings)
    probs = gm.predict_proba(embeddings)
    labels = [np.where(prob > threshold)[0] for prob in probs]
    return labels, n_clusters


def perform_clustering(
    embeddings: np.ndarray,
    dim: int,
    threshold: float,
) -> List[np.ndarray]:
    if len(embeddings) <= dim + 1:
        # Avoid clustering when there's insufficient data
        return [np.array([0]) for _ in range(len(embeddings))]

    # Global dimensionality reduction
    reduced_embeddings_global = global_cluster_embeddings(embeddings, dim)
    # Global clustering
    global_clusters, n_global_clusters = GMM_cluster(
        reduced_embeddings_global, threshold
    )

    all_local_clusters = [np.array([]) for _ in range(len(embeddings))]
    total_clusters = 0

    # Iterate through each global cluster to perform local clustering
    for i in range(n_global_clusters):
        # Extract embeddings belonging to the current global cluster
        global_cluster_embeddings_ = embeddings[
            np.array([i in gc for gc in global_clusters])
        ]

        if len(global_cluster_embeddings_) == 0:
            continue
        if len(global_cluster_embeddings_) <= dim + 1:
            # Handle small clusters with direct assignment
            local_clusters = [np.array([0]) for _ in global_cluster_embeddings_]
            n_local_clusters = 1
        else:
            # Local dimensionality reduction and clustering
            reduced_embeddings_local = local_cluster_embeddings(
                global_cluster_embeddings_, dim
            )
            local_clusters, n_local_clusters = GMM_cluster(
                reduced_embeddings_local, threshold
            )

        # Assign local cluster IDs, adjusting for total clusters already processed
        for j in range(n_local_clusters):
            local_cluster_embeddings_ = global_cluster_embeddings_[
                np.array([j in lc for lc in local_clusters])
            ]
            indices = np.where(
                (embeddings == local_cluster_embeddings_[:, None]).all(-1)
            )[1]
            for idx in indices:
                all_local_clusters[idx] = np.append(
                    all_local_clusters[idx], j + total_clusters
                )

        total_clusters += n_local_clusters

    return all_local_clusters

def embed(embd,texts):
    text_embeddings = embd.embed_documents(texts)
    text_embeddings_np = np.array(text_embeddings)
    return text_embeddings_np

def embed_cluster_texts(embd,texts):
    text_embeddings_np = embed(embd,texts)  # Generate embeddings
    cluster_labels = perform_clustering(
        text_embeddings_np, 10, 0.1
    )  # Perform clustering on the embeddings
    df = pd.DataFrame()  # Initialize a DataFrame to store the results
    df["text"] = texts  # Store original texts
    df["embd"] = list(text_embeddings_np)  # Store embeddings as a list in the DataFrame
    df["cluster"] = cluster_labels  # Store cluster labels
    return df

def fmt_txt(df: pd.DataFrame) -> str:
    unique_txt = df["text"].tolist()
    return "--- --- \n --- --- ".join(unique_txt)


def embed_cluster_summarize_texts(model,embd,
    texts: List[str], level: int
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    df_clusters = embed_cluster_texts(embd,texts)

    # Prepare to expand the DataFrame for easier manipulation of clusters
    expanded_list = []

    # Expand DataFrame entries to document-cluster pairings for straightforward processing
    for index, row in df_clusters.iterrows():
        for cluster in row["cluster"]:
            expanded_list.append(
                {"text": row["text"], "embd": row["embd"], "cluster": cluster}
            )

    # Create a new DataFrame from the expanded list
    expanded_df = pd.DataFrame(expanded_list)

    # Retrieve unique cluster identifiers for processing
    all_clusters = expanded_df["cluster"].unique()
    # Summarization
    template = """Bạn là một chatbot hỗ trợ tuyển sinh và sinh viên đại học, hãy tóm tắt chi tiết tài liệu quy chế dưới đây.
    Đảm bảo rằng nội dung tóm tắt giúp người dùng hiểu rõ các quy định và quy trình liên quan đến tuyển sinh hoặc đào tạo tại đại học.
    Tài liệu:
    {context}
    """
    prompt = ChatPromptTemplate.from_template(template)
    chain = prompt | model | StrOutputParser()

    summaries = []
    for i in all_clusters:
        df_cluster = expanded_df[expanded_df["cluster"] == i]
        formatted_txt = fmt_txt(df_cluster)
        summaries.append(chain.invoke({"context": formatted_txt}))
    df_summary = pd.DataFrame(
        {
            "summaries": summaries,
            "level": [level] * len(summaries),
            "cluster": list(all_clusters),
        }
    )
    return df_clusters, df_summary

def recursive_embed_cluster_summarize(model,embd,
    texts: List[str], level: int = 1, n_levels: int = 3
) -> Dict[int, Tuple[pd.DataFrame, pd.DataFrame]]:
    results = {}  
    df_clusters, df_summary = embed_cluster_summarize_texts(model,embd,texts, level)

    results[level] = (df_clusters, df_summary)

    unique_clusters = df_summary["cluster"].nunique()
    if level < n_levels and unique_clusters > 1:
        new_texts = df_summary["summaries"].tolist()
        next_level_results = recursive_embed_cluster_summarize(model,embd,
            new_texts, level + 1, n_levels
        )
        results.update(next_level_results)

    return results



page = st.title("Chat with AskUSTH")

if "gemini_api" not in st.session_state:
    st.session_state.gemini_api = None

if "rag" not in st.session_state:
    st.session_state.rag = None
    
if "llm" not in st.session_state:
    st.session_state.llm = None
    
if "model" not in st.session_state:
    st.session_state.model = None

if "embd" not in st.session_state:
    st.session_state.embd = None

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
    if st.button("Save"):
        st.session_state.gemini_api = key
        st.rerun()  

if st.session_state.gemini_api is None:
    vote()
else:
    os.environ["GOOGLE_API_KEY"] = st.session_state.gemini_api 
    
    st.session_state.model = ChatGoogleGenerativeAI(
        model="gemini-1.5-flash",
        temperature=0,
        max_tokens=None,
        timeout=None,
        max_retries=2,
    )
    
    st.write(f"Key is set to: {st.session_state.gemini_api}")
    model_name="bkai-foundation-models/vietnamese-bi-encoder"
    model_kwargs = {'device': 'cpu'}
    encode_kwargs = {'normalize_embeddings': False}
    
    st.session_state.embd = HuggingFaceEmbeddings(
        model_name=model_name,
        model_kwargs=model_kwargs,
        encode_kwargs=encode_kwargs
    )   
     
    st.write(f"loaded vietnamese-bi-encoder")       

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
    uploaded_files = st.file_uploader("Chọn file CSV", accept_multiple_files=True, type=["txt"])   
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
                
if st.session_state.uploaded_files:
    if st.session_state.gemini_api is not None:
        if st.session_state.rag is None:
            docs_texts  = [d.page_content for d in documents]

            results = recursive_embed_cluster_summarize(st.session_state.model, st.session_state.embd, docs_texts, level=1, n_levels=3)

            all_texts = docs_texts.copy()

            for level in sorted(results.keys()):
                summaries = results[level][1]["summaries"].tolist()
                all_texts.extend(summaries)

            vectorstore = Chroma.from_texts(texts=all_texts, embedding=st.session_state.embd)
            
            retriever = vectorstore.as_retriever()

            def format_docs(docs):
                return "\n\n".join(doc.page_content for doc in docs)

            template = """<|im_start|>system\nBản là một trợ lí AI hỗ trợ tuyển sinh và sinh viên. Hãy trả lời câu hỏi chính xác, tập trung vào thông tin liên quan đến câu hỏi. Nếu bạn không biết câu trả lời, hãy nói không biết, đừng cố tạo ra câu trả lời.
                    \n

                        Dưới đây là thông tin liên quan mà bạn có thể sử dụng:\n{context}<|im_end|>\n<|im_start|>hãy trả lời: \n{question}<|im_end|>\n<|im_start|>assistant"""
            prompt = PromptTemplate(template = template, input_variables=["context", "question"])
            rag_chain = (
                {"context": retriever | format_docs, "question": RunnablePassthrough()}
                | prompt
                | st.session_state.model
                | StrOutputParser()
            ) 
            st.session_state.rag = rag_chain
            
if st.session_state.gemini_api is not None:
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
if st.session_state.gemini_api:
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
