from operator import itemgetter

from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.chains import LLMChain
from langchain.embeddings import GPT4AllEmbeddings
from langchain.llms import LlamaCpp
from langchain.prompts import ChatPromptTemplate, PromptTemplate
from langchain.prompts.prompt import PromptTemplate
from langchain.schema import format_document
from langchain.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_core.messages import AIMessage, HumanMessage, get_buffer_string
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import (RunnableLambda, RunnableParallel,
                                      RunnablePassthrough)

vectorstore = FAISS.from_texts(
    ["harrison worked at kensho"], embedding=GPT4AllEmbeddings()
)
retriever = vectorstore.as_retriever()

template = """Answer the question based only on the following context:
{context}

Question: {question}
"""
prompt = ChatPromptTemplate.from_template(template)


n_gpu_layers = 1  # Metal set to 1 is enough.
n_batch = 512  # Should be between 1 and n_ctx, consider the amount of RAM of your Apple Silicon Chip.
callback_manager = CallbackManager([StreamingStdOutCallbackHandler()])

# Make sure the model path is correct for your system!
llm = LlamaCpp(
    model_path="C:\\Users\\hoang\\VisualStudioCodeProjects\\PythonProjects\\nlp-assignment-2\\models\\llama-2-7b-chat.Q6_K.gguf",
    n_gpu_layers=n_gpu_layers,
    n_batch=n_batch,
    n_ctx=2048,
    f16_kv=True,  # MUST set to True, otherwise you will run into problem after a couple of calls
    callback_manager=callback_manager,
    verbose=True,
)

model = llm

chain = (
    {"context": retriever, "question": RunnablePassthrough()}
    | prompt
    | model
    | StrOutputParser()
)

chain.invoke("where did harrison work?")

# 'Harrison worked at Kensho.'

template = """Answer the question based only on the following context:
{context}

Question: {question}

Answer in the following language: {language}
"""
prompt = ChatPromptTemplate.from_template(template)

chain = (
    {
        "context": itemgetter("question") | retriever,
        "question": itemgetter("question"),
        "language": itemgetter("language"),
    }
    | prompt
    | model
    | StrOutputParser()
)

chain.invoke({"question": "where did harrison work", "language": "italian"})

# 'Harrison ha lavorato a Kensho.'



_template = """Given the following conversation and a follow up question, rephrase the follow up question to be a standalone question, in its original language.

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


_inputs = RunnableParallel(
    standalone_question=RunnablePassthrough.assign(
        chat_history=lambda x: get_buffer_string(x["chat_history"])
    )
    | CONDENSE_QUESTION_PROMPT
    | model
    | StrOutputParser(),
)
_context = {
    "context": itemgetter("standalone_question") | retriever | _combine_documents,
    "question": lambda x: x["standalone_question"],
}
conversational_qa_chain = _inputs | _context | ANSWER_PROMPT | model


conversational_qa_chain.invoke(
    {
        "question": "where did harrison work?",
        "chat_history": [],
    }
)

# AIMessage(content='Harrison was employed at Kensho.')

conversational_qa_chain.invoke(
    {
        "question": "where did he work?",
        "chat_history": [
            HumanMessage(content="Who wrote this notebook?"),
            AIMessage(content="Harrison"),
        ],
    }
)

# AIMessage(content='Harrison worked at Kensho.')



from operator import itemgetter

from langchain.memory import ConversationBufferMemory

memory = ConversationBufferMemory(
    return_messages=True, output_key="answer", input_key="question"
)

# First we add a step to load memory
# This adds a "memory" key to the input object
loaded_memory = RunnablePassthrough.assign(
    chat_history=RunnableLambda(memory.load_memory_variables) | itemgetter("history"),
)
# Now we calculate the standalone question
standalone_question = {
    "standalone_question": {
        "question": lambda x: x["question"],
        "chat_history": lambda x: get_buffer_string(x["chat_history"]),
    }
    | CONDENSE_QUESTION_PROMPT
    | model
    | StrOutputParser(),
}
# Now we retrieve the documents
retrieved_documents = {
    "docs": itemgetter("standalone_question") | retriever,
    "question": lambda x: x["standalone_question"],
}
# Now we construct the inputs for the final prompt
final_inputs = {
    "context": lambda x: _combine_documents(x["docs"]),
    "question": itemgetter("question"),
}
# And finally, we do the part that returns the answers
answer = {
    "answer": final_inputs | ANSWER_PROMPT | model,
    "docs": itemgetter("docs"),
}
# And now we put it all together!
final_chain = loaded_memory | standalone_question | retrieved_documents | answer


inputs = {"question": "where did harrison work?"}
result = final_chain.invoke(inputs)
result

{'answer': AIMessage(content='Harrison was employed at Kensho.'),
 'docs': [Document(page_content='harrison worked at kensho')]}

# Note that the memory does not save automatically
# This will be improved in the future
# For now you need to save it yourself
memory.save_context(inputs, {"answer": result["answer"]})

memory.load_memory_variables({})

# {'history': [HumanMessage(content='where did harrison work?'),
#   AIMessage(content='Harrison was employed at Kensho.')]}

inputs = {"question": "but where did he really work?"}
result = final_chain.invoke(inputs)
result

# {'answer': AIMessage(content='Harrison actually worked at Kensho.'),
#  'docs': [Document(page_content='harrison worked at kensho')]}

import collections
import json

import matplotlib.pyplot as plt
from langchain import HuggingFaceHub
from langchain.chains import RetrievalQA
from langchain.chains.question_answering import load_qa_chain
from langchain.document_loaders import TextLoader
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.retrievers import (BM25Retriever, EnsembleRetriever,
                                  ParentDocumentRetriever)
from langchain.schema import Document
from langchain.storage import InMemoryStore
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS, Chroma

text_splitter = RecursiveCharacterTextSplitter(chunk_size = 1000, chunk_overlap = 200)
DOCUMENT_DIR = "../data/subsections/subsections"

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

model_name = "VoVanPhuc/sup-SimCSE-VietNamese-phobert-base"
model_kwargs={'device': 'cuda'}
encode_kwargs = {'normalize_embeddings': True} # set True to compute cosine similarity

vib_embeddings = HuggingFaceEmbeddings(
    model_name=model_name,
    model_kwargs=model_kwargs,
    encode_kwargs=encode_kwargs
)

embeddings = vib_embeddings.embed_documents([doc.page_content for doc in docs])

import faiss
import numpy as np

embeddings = np.array(embeddings, dtype=np.float32)
index = faiss.IndexFlatL2(embeddings.shape[1])
index.add(embeddings)

print(index)
faiss.write_index(index, '../embeddings.index')

vectorstore = FAISS.from_documents(
    documents=docs,
    embedding=vib_embeddings
)
vector_retriever = vectorstore.as_retriever(search_kwargs=dict(k=2))
vectorstore.save_local('../')

bm25_retriever = BM25Retriever.from_documents(docs)
bm25_retriever.k = 2

retriever = EnsembleRetriever(retrievers=[bm25_retriever, vector_retriever],
                             weights=[0.5, 0.5]) 



snippet = """U não là tình trạng các khối u hình thành trong sọ não, đe dọa tính mạng người bệnh. U não thường có 2 loại là lành tính và ác tính. Đâu là điểm chung của chúng?
A. Đều là các bệnh nguy hiểm
B. Đều là ung thư
C. Nguyên nhân chính xác không thể xác định
D. Xảy ra nhiều nhất ở người già"""

kags = bm25_retriever.get_relevant_documents(snippet)
