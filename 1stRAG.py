# -*- coding: utf-8 -*-
"""RAG.ipynb


"""

# Commented out IPython magic to ensure Python compatibility.
import sys

if 'google.colab' in sys.modules:
#   %pip install --upgrade pip
#   %pip install langchain-openai langchain-chroma langchain-huggingface langchain-community tiktoken python-dotenv scikit-learn plotly

import os
import glob
import tiktoken
import numpy as np
from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sklearn.manifold import TSNE
import plotly.graph_objects as go

load_dotenv(override =True);
db_name="vector_db"
# api_key = os.getenv("OPENAI_API_KEY");

# if not api_key:
#    print("Check envirotnment file..key not found");
# else:
#    print(f"everythign looks fine and api key is {api_key}");

# ollama_url = "http://127.0.0.1:11434/v1"
# ollama_model = "qwen3:1.7b";

knowledge_base_path = "drive/MyDrive/LLM/llmCourse/week5data/**/*.md";
files = glob.glob(knowledge_base_path,recursive=True);
print(f"Found {len(files)} files in the knowledge folder")

entire_knowledge_base = "";
for file_path in files:
   with open(file_path,'r',encoding='utf-8') as f:
      entire_knowledge_base+=f.read();
      entire_knowledge_base+= "\n\n"

print(f"Total character in entire knowledge base : {len(entire_knowledge_base)}")

encoding = tiktoken.encoding_for_model(model_name="gpt-4.1-nano");
tokens = encoding.encode(entire_knowledge_base);
token_count = len(tokens);
print(token_count)

print(tokens)

folders = glob.glob("drive/MyDrive/LLM/llmCourse/week5data/*");
documents = [];
for folder in folders:
   doc_type = os.path.basename(folder);
   print(doc_type)
   loader = DirectoryLoader(folder,glob="**/*.md",loader_cls=TextLoader,loader_kwargs={'encoding':'utf-8'});
   folder_docs = loader.load();
   print(f"------loader-----{(folder_docs)}")
   for doc in folder_docs:
      doc.metadata["doc_type"]=doc_type;
      documents.append(doc);

text_splitter = RecursiveCharacterTextSplitter(chunk_size=900, chunk_overlap=200)
chunks = text_splitter.split_documents(documents)

print(f"Divided into {len(chunks)} chunks")
print(f"First chunk:\n\n{chunks[0]}")

# Pick an embedding model

embeddings = HuggingFaceEmbeddings(model_name="BAAI/bge-m3")
#embeddings = OpenAIEmbeddings(model="text-embedding-3-large")

if os.path.exists(db_name):
    Chroma(persist_directory=db_name, embedding_function=embeddings).delete_collection()

vectorstore = Chroma.from_documents(documents=chunks, embedding=embeddings, persist_directory=db_name)
print(f"Vectorstore created with {vectorstore._collection.count()} documents")

# Let's investigate the vectors

collection = vectorstore._collection
count = collection.count()

sample_embedding = collection.get(limit=1, include=["embeddings"])["embeddings"][0]
dimensions = len(sample_embedding)
print(f"There are {count:,} vectors with {dimensions:,} dimensions in the vector store")

# Prework

result = collection.get(include=['embeddings', 'documents', 'metadatas'])
vectors = np.array(result['embeddings'])
documents = result['documents']
metadatas = result['metadatas']
doc_types = [metadata['doc_type'] for metadata in metadatas]
colors = [['blue', 'green', 'red', 'orange'][['products', 'employees', 'contracts', 'company'].index(t)] for t in doc_types]

# We humans find it easier to visalize things in 2D!
# Reduce the dimensionality of the vectors to 2D using t-SNE
# (t-distributed stochastic neighbor embedding)

tsne = TSNE(n_components=2, random_state=42)
reduced_vectors = tsne.fit_transform(vectors)

# Create the 2D scatter plot
fig = go.Figure(data=[go.Scatter(
    x=reduced_vectors[:, 0],
    y=reduced_vectors[:, 1],
    mode='markers',
    marker=dict(size=5, color=colors, opacity=0.8),
    text=[f"Type: {t}<br>Text: {d[:100]}..." for t, d in zip(doc_types, documents)],
    hoverinfo='text'
)])

fig.update_layout(title='2D Chroma Vector Store Visualization',
    scene=dict(xaxis_title='x',yaxis_title='y'),
    width=800,
    height=600,
    margin=dict(r=20, b=10, l=10, t=40)
)

fig.show()

# Let's try 3D!

tsne = TSNE(n_components=3, random_state=42)
reduced_vectors = tsne.fit_transform(vectors)

# Create the 3D scatter plot
fig = go.Figure(data=[go.Scatter3d(
    x=reduced_vectors[:, 0],
    y=reduced_vectors[:, 1],
    z=reduced_vectors[:, 2],
    mode='markers',
    marker=dict(size=5, color=colors, opacity=0.8),
    text=[f"Type: {t}<br>Text: {d[:100]}..." for t, d in zip(doc_types, documents)],
    hoverinfo='text'
)])

fig.update_layout(
    title='3D Chroma Vector Store Visualization',
    scene=dict(xaxis_title='x', yaxis_title='y', zaxis_title='z'),
    width=900,
    height=700,
    margin=dict(r=10, b=10, l=10, t=40)
)

fig.show()

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

from langchain_chroma import Chroma
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_huggingface import HuggingFaceEmbeddings
import gradio as gr

DB_NAME = "vector_db"
load_dotenv(override=True)

ollama_url = "http://127.0.0.1:11434/v1"
ollama_model = "qwen3:1.7b";

embeddings = HuggingFaceEmbeddings(model_name="BAAI/bge-m3")
vectorstore = Chroma(persist_directory=DB_NAME, embedding_function=embeddings)

!pip install -qU langchain langchain-ollama
from langchain_ollama import ChatOllama
retriever = vectorstore.as_retriever()
llm = ChatOllama(
    model="qwen3:1.7b",
    base_url="http://127.0.0.1:11434",
    temperature=0,
)

# Commented out IPython magic to ensure Python compatibility.
!curl -fsSL https://ollama.com/install.sh | sh

# %pip install -qU langchain-ollama requests
!which ollama
!ollama --version

print(retriever.invoke("Who is Avery?"))
def systemprompt(context):
  return f"""
You are a knowledgeable, friendly assistant representing the company Insurellm.
You are chatting with a user about Insurellm.
If relevant, use the given context to answer any question.
If you don't know the answer, say so.
Context:
{context}
"""

from openai import OpenAI
client = OpenAI(
    api_key=os.getenv("GEMINI_API_KEY"),
    base_url="https://generativelanguage.googleapis.com/v1beta/openai/"
)
def answer_question(question: str, history):
    docs = retriever.invoke(question)
    context = "\n\n".join(doc.page_content for doc in docs)
    system_prompt = systemprompt(context)
    print(system_prompt)
    #response = llm.invoke([SystemMessage(content=system_prompt), HumanMessage(content=question)])
    response = client.chat.completions.create(
        model="gemini-2.5-flash",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": question}
        ]);
    return response.choices[0].message.content

answer_question("Who is Averi Lancaster?", [])

gr.ChatInterface(answer_question).launch()

