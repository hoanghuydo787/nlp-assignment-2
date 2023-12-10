# 1. Set up RAG

## 1.1 Import

# !pip install chromadb
!pip install langchain -q
!pip install sentence_transformers -q
!pip install accelerate -q
!pip install bitsandbytes -q
!pip install gdown -q
# !pip install openai

import os
from getpass import getpass
import time
# langchain packages
from langchain.chains import RetrievalQA
from langchain.chat_models import ChatOpenAI
from langchain.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Chroma, FAISS
from langchain import HuggingFaceHub
from langchain.llms import OpenAI
from langchain.chains.question_answering import load_qa_chain
from langchain.retrievers import ParentDocumentRetriever
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.storage import InMemoryStore

import os
import collections
import matplotlib.pyplot as plt

!pip install chardet -q
!pip install rank_bm25 -q
# !pip install faiss-gpu

## 2.3 Vector Store

import json
from langchain.retrievers import BM25Retriever, EnsembleRetriever
from langchain.schema import Document

DOCUMENT_DIR = "/kaggle/input/vn-medical-qa-dataset/subsections/subsections"

docs = []
for filename in sorted(os.listdir(DOCUMENT_DIR)):
    filepath = os.path.join(DOCUMENT_DIR, filename)
    with open(filepath, "r", encoding="utf-8") as f:
        file_data = json.load(f)
        docs.append(Document(
            page_content=file_data["subsection_data"],
            metadata={
                "filename": filename,
                "filepath": filepath,
                "document_name": file_data["document_name"],
                "document_name_accent": file_data["document_name_accent"],
                "document_title": file_data["document_title"],
                "document_category": file_data["document_category"],
                "subsection_name": file_data["subsection_name"],
                "subsection_title": file_data["subsection_title"],
            }
        ))

bm25_retriever = BM25Retriever.from_documents(docs)
bm25_retriever.k = 5

# retriever = EnsembleRetriever(retrievers=[bm25_retriever, vector_retriever],
#                              weights=[0.55, 0.45]) 

### Sanity check

snippet = """U não là tình trạng các khối u hình thành trong sọ não, đe dọa tính mạng người bệnh. U não thường có 2 loại là lành tính và ác tính. Đâu là điểm chung của chúng?
A. Đều là các bệnh nguy hiểm
B. Đều là ung thư
C. Nguyên nhân chính xác không thể xác định
D. Xảy ra nhiều nhất ở người già"""

docs = bm25_retriever.get_relevant_documents(snippet)

docs[:2]

## 2.4 Reader

from langchain.chains import RetrievalQA
from langchain.llms import OpenAI
from langchain.prompts import PromptTemplate
from langchain.llms import HuggingFacePipeline
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
import torch

# 1st VERSION with quantization
# model_path = "vilm/vietcuna-3b-v2"
# model_path = "vilm/vietcuna-7b-v3"
model_path = "vlsp-2023-vllm/hoa-7b"
# model_path = "infCapital/llama2-7b-chatvi"

model = AutoModelForCausalLM.from_pretrained(
            model_path,
            device_map="auto",  # automatically store parameters on gpu, cpu or disk
            low_cpu_mem_usage=True,  # try to limit RAM
            load_in_8bit=True,  # load model in low precision to save memory
            torch_dtype=torch.float16,  # load model in low precision to save memory
            offload_state_dict=True,  # offload onto disk if needed
            offload_folder="offload",  # offload model to `offload/`
        )
tokenizer = AutoTokenizer.from_pretrained(model_path)

pipel = pipeline(
            "text-generation",
            model=model,
            tokenizer=tokenizer,
        )

llm = HuggingFacePipeline(pipeline=pipel)

# template = """Sử dụng các trích đoạn sau đây để trả lời câu hỏi trắc nghiệm. Nếu không đề cập trong trích đoạn, chọn không biết.

# {context}

# Câu hỏi: {question}
# Trả lời:"""

# template = """Sử dụng các trích đoạn sau đây để trả lời câu hỏi trắc nghiệm.

# {context}

# Câu hỏi: {question}
# Trả lời:"""

template = """Sử dụng các trích đoạn sau đây để trả lời câu hỏi. Chọn MỘT hoặc NHIỀU đáp án đúng. 

{context}

Câu hỏi: {question}
Trả lời:"""

qa = RetrievalQA.from_chain_type(
                    llm=llm, #OpenAI()
                    chain_type="stuff",
                    retriever=bm25_retriever,
                    return_source_documents=True,
                    chain_type_kwargs={
                                "prompt": PromptTemplate(
                                    template=template,
                                    input_variables=["question", "context"]
                                ),
                          }
                )


# 2nd VERSION with custom template
# template = """Use the following pieces of context to answer the question at the end by choosing the correct option(s).

# {context}

# Question: {question}
# Option: {option}
# Helpful Answer:"""
# qa = RetrievalQA.from_chain_type(
#             llm=model,#OpenAI()
#             chain_type="stuff",
#             retriever=retriever,
#             return_source_documents=True,
#             chain_type_kwargs={
#                 "prompt": PromptTemplate(
#                     template=template,
#                     input_variables=["option", "question", "context"],
#                 ),
#           },)

print(qa.combine_documents_chain.llm_chain.prompt.template)

def process_llm_response(llm_response):
    print("Answer: ", llm_response['result'])
    print('\n\nSources:')
    for source in llm_response['source_documents']:
        print(source.metadata['source'])
        print(source.page_content)
        print('@'*1000)

import pandas as pd

testset = pd.read_csv("MEDICAL/public_test.csv")

import math
sample = testset.loc[87]
newline = '\n'
options = [str(p) for p in sample.loc[['option_' + str(i) for i in range(1,7)]] if str(p) != 'nan']
labels = {0:'A', 1:'B', 2:'C', 3:'D', 4:'E', 5:'F'}
options = [labels[i] + '. ' + p if not p.startswith(f'{labels[i]}.') else p for i, p in enumerate(options)]

sample_query = f"""{sample['question']}
{newline.join(options)}
"""

# sample_query = """Ông Biền năm nay 73 tuổi. Trong một bữa cơm gia đình, ông đột nhiên bị méo, lệch một bên miệng và mặt. 3 tháng sau ông qua đời. Ông Biền có thể đã mắc những căn bệnh nào dưới đây?
# A. Không biết
# B. Tai biến mạch máu não

# """
print(sample_query)

# response = qa(sample_query)
# process_llm_response(response)

response = qa(sample_query)
process_llm_response(response)


