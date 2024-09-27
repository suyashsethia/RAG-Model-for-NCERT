# main.py
import asyncio
import faiss
import numpy as np
import json
import pandas as pd
# import pdfplumber
import fitz  
import re
import textwrap
import os
import numpy as np
import torch
import logging
import google.generativeai as genai
from sentence_transformers import SentenceTransformer
from typing import AsyncIterable, Annotated
from decouple import config
from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from fastui import prebuilt_html, FastUI, AnyComponent
from fastui import components as c
from fastui.components.display import DisplayLookup, DisplayMode
from fastui.events import PageEvent, GoToEvent
from pydantic import BaseModel, Field
from starlette.responses import StreamingResponse
# from fastui.components.display import Link  # Import Link from FastUI
# Or use `os.getenv('API_KEY')` to fetch an environment variable.
API_KEY="AIzaSyCzcopOrcDHgZIdjrFuOrYeBasjG0qiwec" #fine grained token

genai.configure(api_key=API_KEY)
model = genai.GenerativeModel('gemini-1.5-flash-001')

# Create the app object
app = FastAPI()
# Message history


# Root endpoint
@app.get('/api/', response_model=FastUI, response_model_exclude_none=True)
def api_index(chat: str | None = None) -> list[AnyComponent]:
    return [
        c.PageTitle(text='FastUI Chatbot'),
        c.Page(
            components=[
                # Header
                #center the header
                c.Heading(text='RAG-bot' , class_name='text-center'),
                c.Paragraph(text='This is a RAG ased chatbot built with FastUI and FastAPI'),
                # Chat form
                #small heading level 3 write your query here
                c.Heading(text='Write your query here', level=4),
                c.ModelForm(model=ChatForm, submit_url=".", method='GOTO'),
                # Chatbot response
                c.Div(
                    components=[
                        c.ServerLoad(
                            path=f"/sse/{chat}", # Server side events endpoint
                            sse=True,
                            load_trigger=PageEvent(name='load'),
                            components=[],
                        )
                    ],
                    class_name='my-2 p-2 border rounded'),
            ],
        ),
        # Footer
        c.Footer(
            extra_text='Made for Sarvam AI',
            links=[]

        )
    ]


# Load the model for generating embeddings
encoding_model = SentenceTransformer('all-MiniLM-L6-v2')

# Chunking function using sliding window strategy
def chunk_documents(documents, window_size=256, overlap=16):
    chunks = []
    for doc in documents:
        words = doc.split()
        for i in range(0, len(words), window_size - overlap):
            chunk = ' '.join(words[i:i + window_size])
            chunks.append(chunk)
    return chunks

# Function to create and store FAISS index
def create_faiss_index(chunks, index_save_path='index.faiss', json_save_path='memory.json'):
    # Convert chunks to embeddings
    chunk_embeddings = encoding_model.encode(chunks)
    chunk_embeddings = np.array(chunk_embeddings).astype('float32')
    
    # Initialize FAISS index
    dimension = chunk_embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)  # Using L2 distance metric

    # Add embeddings to the index
    index.add(chunk_embeddings)
    
    # Save FAISS index in binary format
    faiss.write_index(index, index_save_path)

    # Save chunks in memory.json
    memory = {
        "chunks": chunks
    }
    
    with open(json_save_path, 'w') as f:
        json.dump(memory, f)

# Function to load the FAISS index and chunks from disk
def load_faiss_index(index_save_path='index.faiss', json_save_path='memory.json'):
    # Load FAISS index
    index = faiss.read_index(index_save_path)

    # Load chunks
    with open(json_save_path, 'r') as f:
        memory = json.load(f)
    chunks = memory["chunks"]
    
    return index, chunks

# Function to retrieve the top k documents for a query
def retrieve_top_k_documents(query, k=3, index_save_path='index.faiss', json_save_path='memory.json'):
    # Load the FAISS index and chunks from disk
    index, chunks = load_faiss_index(index_save_path, json_save_path)

    # Generate embedding for the query
    query_embedding = encoding_model.encode([query]).astype('float32')

    # Search the index for top k documents
    distances, indices = index.search(query_embedding, k)

    # Return the top k chunks and their scores
    top_chunks = [(chunks[i], distances[0][j]) for j, i in enumerate(indices[0])]
    return top_chunks


# documents= documents_method1

# # Chunk the documents using the sliding window strategy
# chunks = chunk_documents(documents)
# # Create and store the FAISS index and chunks
# create_faiss_index(chunks)
# Example query
# the above function is only needed to be run once to create the index and save it to disk
def make_prompt(query):
    results = retrieve_top_k_documents(query, k=3)
    
    if results[0][1]<1.5:

        prompt = ("""SYSTEM:

    You are a friendly, knowledgeable, and enthusiastic high school teacher. Your goal is to help students with their academic queries across various subjects. As a teacher, you should explain concepts clearly and patiently, always offering encouragement and support. Use examples and analogies whenever possible to help students understand complex topics. If the student seems confused, clarify your explanation, but avoid overwhelming them with too much information at once.

    Make sure to maintain a positive and motivating tone, encouraging curiosity and a love for learning. If you're unsure of the answer, reassure the student that it's okay not to know everything right away and guide them on how to explore the topic further.

    You should base your response on the provided context and documents retrieved by the system, answering the student’s question with accurate and relevant information.

    You are a teacher who should provide a helpful and educational response to the student's query. Ensure your answer is concise, informative, and tailored to the student's academic level.

   USER:                 

    Use the following context to answer the student's query:

    QUESTION: '{query}'

    PASSAGES: 
    1. {relevant_passage_1}
    2. {relevant_passage_2}
    3. {relevant_passage_3}

    ASSISTANT:

    ANSWER:

    
    """).format(query=query, relevant_passage_1=results[0][0], relevant_passage_2=results[1][0], relevant_passage_3=results[2][0])
    else:
        prompt = textwrap.dedent("""SYSTEM:

You are a friendly, knowledgeable, and enthusiastic high school teacher. Your goal is to help students with their academic queries across various subjects, even when no specific documents are provided. As a teacher, you should explain concepts clearly and patiently, always offering encouragement and support. Use examples and analogies to simplify complex topics, and provide guidance on how students can further explore or understand the material.

If the student asks a question that doesn't match the available information, reassure them that it's okay to ask about different topics and answer based on your own expertise. Be sure to maintain a positive and motivating tone, and encourage the student to stay curious and engaged.

Your response should be concise, informative, and tailored to the student's academic level.

USER:                    

QUESTION: '{query}'

ASSISTATNT:

ANSWER:


""").format(query=query)
                  
    return prompt

async def ai_response_generator(prompt: str) -> AsyncIterable[str]:
    # Create the formatted prompt
    generated_prompt = make_prompt(prompt)
    # print("This is an info log message get_answers")
    # Generate the content using Gemini API (not streaming, so we chunk manually)
    output = ""
    response = model.generate_content(generated_prompt ,stream=True)
    for chunk in response:
        output += chunk.text
        m = FastUI(root=[c.Markdown(text=output)])
        msg = f'data: {m.model_dump_json(by_alias=True, exclude_none=True)}\n\n'
        yield msg
        await asyncio.sleep(0.05)  

    while True:
        yield msg
        await asyncio.sleep(1)

# Server side events endpoint
@app.get('/api/sse/{prompt}')
async def sse_ai_response(prompt: str) -> StreamingResponse:
    
    if prompt:
        return StreamingResponse(ai_response_generator(prompt), media_type='text/event-stream')
    else:
        return StreamingResponse(empty_response(), media_type='text/event-stream')

# Empty response generator
async def empty_response() -> AsyncIterable[str]:
    # Send the message
    m = FastUI(root=[c.Markdown(text='')])
    msg = f'data: {m.model_dump_json(by_alias=True, exclude_none=True)}\n\n'
    yield msg
    # Avoid the browser reconnecting
    while True:
        yield msg
        await asyncio.sleep(1)

# Pre-built HTML
@app.get('/{path:path}')
async def html_landing() -> HTMLResponse:
    """Simple HTML page which serves the React app, comes last as it matches all paths."""
    return HTMLResponse(prebuilt_html(title='FastUI Demo'))

# Chat form
class ChatForm(BaseModel):
    chat: str = Field(title=' ', max_length=1000)