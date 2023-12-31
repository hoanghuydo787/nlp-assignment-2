import json
import os
from operator import itemgetter

import faiss
import numpy as np
from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.chains import LLMChain, RetrievalQA
from langchain.chains.question_answering import load_qa_chain
from langchain.document_loaders import TextLoader
from langchain.embeddings import GPT4AllEmbeddings, HuggingFaceEmbeddings
from langchain.llms import HuggingFaceHub, LlamaCpp
from langchain.memory import ConversationBufferMemory
from langchain.prompts import ChatPromptTemplate, PromptTemplate
from langchain.prompts.prompt import PromptTemplate
from langchain.retrievers import (BM25Retriever, EnsembleRetriever,
                                  ParentDocumentRetriever)
from langchain.schema import Document, format_document
from langchain.storage import InMemoryStore
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_core.messages import AIMessage, HumanMessage, get_buffer_string
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import (RunnableLambda, RunnableParallel,
                                      RunnablePassthrough)


# prepare embedding
text_splitter = RecursiveCharacterTextSplitter(chunk_size = 1000, chunk_overlap = 200)
# DOCUMENT_DIR = "../data/subsections/subsections"
DOCUMENT_DIR = "../data/translated_subsections/translated_subsections"

docs = []
for filename in sorted(os.listdir(DOCUMENT_DIR)):
    filepath = os.path.join(DOCUMENT_DIR, filename)
    with open(filepath, "r", encoding="utf-8") as f:
        file_data = json.load(f)
        subs = text_splitter.split_text(file_data['subsection_content'])
        subs = [file_data['subsection_title'] + '\n' + text for text in subs]
        for i, sub in enumerate(subs):
            docs.append(Document(
                page_content=sub,
                metadata={
                    "filename": filename,
                    "filepath": filepath,
                    "document_name": file_data["document_name"],
                    "document_name_accent": file_data["document_name_accent"],
                    "document_title": file_data["document_title"],
                    "document_category": file_data["document_category"],
                    "subsection_name": file_data["subsection_name"],
                    "subsection_title": file_data["subsection_title"],
                    "chunk_id": i
                }
            ))

# model_name = "VoVanPhuc/sup-SimCSE-VietNamese-phobert-base"
model_name = "BAAI/bge-large-en-v1.5"

model_kwargs={'device': 'cpu'}
encode_kwargs = {'normalize_embeddings': True} # set True to compute cosine similarity

vib_embeddings = HuggingFaceEmbeddings(
    model_name=model_name,
    model_kwargs=model_kwargs,
    encode_kwargs=encode_kwargs
)

# vectorstore = FAISS.from_documents(
#     documents=docs,
#     embedding=vib_embeddings
# )
# vector_retriever = vectorstore.as_retriever(search_kwargs=dict(k=2))
# vectorstore.save_local('../en_index/1000_chunks')

vectorstore = FAISS.load_local(
    '../en_index/1000_chunks',
    vib_embeddings
)
vector_retriever = vectorstore.as_retriever(search_kwargs=dict(k=1))

bm25_retriever = BM25Retriever.from_documents(docs)
bm25_retriever.k = 1

retriever = EnsembleRetriever(retrievers=[bm25_retriever, vector_retriever],
                             weights=[0.5, 0.5]) 

# # snippet = """U não là tình trạng các khối u hình thành trong sọ não, đe dọa tính mạng người bệnh. U não thường có 2 loại là lành tính và ác tính. Đâu là điểm chung của chúng?
# # A. Đều là các bệnh nguy hiểm
# # B. Đều là ung thư
# # C. Nguyên nhân chính xác không thể xác định
# # D. Xảy ra nhiều nhất ở người già"""
snippet = """Brain tumors are masses formed within the skull, posing a threat to the patient's life. Generally, there are two types of brain tumors: benign and malignant. What is a common characteristic between them?
A. Both are dangerous illnesses.
B. Both are cancers.
C. The exact primary cause cannot be determined.
D. Occur most frequently in older people."""
# kags = bm25_retriever.get_relevant_documents(snippet)
# kags = vector_retriever.get_relevant_documents(snippet)
kags = retriever.get_relevant_documents(snippet)
# print(kags)

n_gpu_layers = 1
n_batch = 512
callback_manager = CallbackManager([StreamingStdOutCallbackHandler()])

llm = LlamaCpp(
    model_path="C:\\Users\\hoang\\VisualStudioCodeProjects\\PythonProjects\\nlp-assignment-2\\models\\llama-2-7b-chat.Q6_K.gguf",
    n_gpu_layers=n_gpu_layers,
    n_batch=n_batch,
    n_ctx=2048,
    f16_kv=True,  # MUST set to True, otherwise you will run into problem after a couple of calls
    callback_manager=callback_manager,
    verbose=False,
)

model = llm

_template = """Given the following conversation and a follow up question, rephrase the follow up question to be a standalone question.

Chat History:
{chat_history}
Follow Up Input: {question}
Standalone question:"""
CONDENSE_QUESTION_PROMPT = PromptTemplate.from_template(_template)

template = """Answer the question based only on the following context:
{context}

Question: {question}
"""
ANSWER_PROMPT = ChatPromptTemplate.from_template(template)

DEFAULT_DOCUMENT_PROMPT = PromptTemplate.from_template(template="{page_content}")

def _combine_documents(
    docs, document_prompt=DEFAULT_DOCUMENT_PROMPT, document_separator="\n\n"
):
    doc_strings = [format_document(doc, document_prompt) for doc in docs]
    return document_separator.join(doc_strings)

memory = ConversationBufferMemory(
    return_messages=True, output_key="answer", input_key="question"
)
loaded_memory = RunnablePassthrough.assign(
    chat_history=RunnableLambda(memory.load_memory_variables) | itemgetter("history"),
)
standalone_question = {
    "standalone_question": {
        "question": lambda x: x["question"],
        "chat_history": lambda x: get_buffer_string(x["chat_history"]),
    }
    | CONDENSE_QUESTION_PROMPT
    | model
    | StrOutputParser(),
}
retrieved_documents = {
    "docs": itemgetter("standalone_question") | retriever,
    "question": lambda x: x["standalone_question"],
}
final_inputs = {
    "context": lambda x: _combine_documents(x["docs"]),
    "question": itemgetter("question"),
}
answer = {
    "answer": final_inputs | ANSWER_PROMPT | model,
    "docs": itemgetter("docs"),
}
final_chain = loaded_memory | standalone_question | retrieved_documents | answer

import streamlit as st

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("What is up?"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)
    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        inputs = {"question": prompt}
        result = final_chain.invoke(inputs)
        full_response = result['answer']
        memory.save_context(inputs, {"answer": result["answer"]})
        memory.load_memory_variables({})
        message_placeholder.markdown(full_response)
    st.session_state.messages.append({"role": "assistant", "content": full_response})
