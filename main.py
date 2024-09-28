# main.py
import asyncio
import faiss
import numpy as np
import json
import pandas as pd
from fastui import events as e # Import events
# import pdfplumber
from IPython.display import Audio
import fitz  
from fastapi.staticfiles import StaticFiles
from IPython.display import Audio
# Serve the directory where audio files are saved
from gtts import gTTS
from io import BytesIO
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
from transformers import RobertaTokenizer, RobertaForSequenceClassification
import torch
import os
from fastapi.responses import FileResponse
from fastapi import FastAPI, Request
from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse
templates = Jinja2Templates(directory="templates")

import io
import wave
import pyaudio
import keyboard
from google.cloud import speech_v1p1beta1 as speech  # Correct import for Speech-to-Text
from google.cloud import texttospeech  # Text-to-Speech
from typing import Any, List, Union
# Set the environment variable to the path of your service account key
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "animated-spirit-433009-p5-773b7d3034a1.json"
label_dict = {'Biology': 0, 'Physics': 1, 'Chemistry': 2}

# Load the trained model and tokenizer from the 'saved_model' directory
model_for_confidence = './saved_model'
tokenizer = RobertaTokenizer.from_pretrained(model_for_confidence)
model_for_confidence = RobertaForSequenceClassification.from_pretrained(model_for_confidence)

# Move the model to GPU (if available)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model_for_confidence.to(device)

API_KEY="AIzaSyCzcopOrcDHgZIdjrFuOrYeBasjG0qiwec" #fine grained token
genai.configure(api_key=API_KEY)
model = genai.GenerativeModel('gemini-1.5-flash-001')

# Create the app object
app = FastAPI()


@app.get('/api/', response_model=FastUI, response_model_exclude_none=True)
def api_index(chat: str | None = None) -> list[AnyComponent]:
    return [
        c.PageTitle(text='RAG bot'),
        c.Page(
            components=[
                # Header
                #center the header
                c.Heading(text='RAG-bot' , class_name='text-center'),
                c.Paragraph(text='This is a RAG based chatbot built with FastUI and FastAPI . it can answer your queries by specifically tailoring the prompt based on user query'),
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
                        ),
                    ],
                    class_name='my-2 p-2 border rounded'
                ),
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

# Inference function for physics, chemistry, and biology
def predict(text):
    model_for_confidence.eval()  # Set the model to evaluation mode
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=128).to(device)  # Move inputs to the correct device
    
    with torch.no_grad():
        outputs = model_for_confidence(**inputs)  # Forward pass
        probs = torch.nn.functional.softmax(outputs.logits, dim=-1)  # Softmax to get probabilities
        # confidences = probs[0].tolist()  # Convert to list of confidence scores
        predicted_class = torch.argmax(probs).item()  # Get predicted class
        confidence_score = probs[0][predicted_class].item()
    # return the predicted class and confidence of the prediction


    return {
        'predicted_class': list(label_dict.keys())[predicted_class],
        'confidence_scores': confidence_score
    }

def make_prompt(query):

    results = retrieve_top_k_documents(query, k=3)
    prediction = predict(query)
    predicted_class = prediction['predicted_class']
    confidence_score = prediction['confidence_scores']
    
    prompt_confidence_relevant = (""""SYSTEM:

    
    You are a friendly, knowledgeable, and enthusiastic high school {predicted_class} teacher. Your goal is to help students with their academic queries in {predicted_class}. As a teacher, you should explain concepts clearly and patiently, always offering encouragement and support. Use examples and analogies whenever possible to help students understand complex topics of {predicted_class}. If the student seems confused, clarify your explanation, but avoid overwhelming them with too much information at once.

    Make sure to maintain a positive and motivating tone, encouraging curiosity and a love for learning. If you're unsure of the answer, reassure the student that it's okay not to know everything right away and guide them on how to explore the topic further in {predicted_class}.

    You should base your response on the provided context and documents retrieved by the system, answering the student’s question with accurate and relevant information.

    You are a {predicted_class} teacher who should provide a helpful and educational response to the student's query. Ensure your answer is concise, informative, and tailored to the student's academic level.

   USER:                 

    Use the following context to answer the student's query:

    QUESTION: '{query}'

    PASSAGES: 
    1. {relevant_passage_1}
    2. {relevant_passage_2}
    3. {relevant_passage_3}

    ASSISTANT:

    ANSWER:
                                  start the answer with the following:
                                    "as a {predicted_class} Teacher RAG Agent , the answer to your question is as follows:"

    
    """).format(query=query, relevant_passage_1=results[0][0], relevant_passage_2=results[1][0], relevant_passage_3=results[2][0], predicted_class=predicted_class, confidence = confidence_score)

    prompt_confidence_not_relevant = ("""SYSTEM:

You are a friendly, knowledgeable, and enthusiastic high school {predicted_class} teacher. Your goal is to help students with their academic queries across {predicted_class}, even when no specific documents are provided. As a teacher, you should explain {predicted_class} concepts clearly and patiently, always offering encouragement and support. Use examples and analogies to simplify complex topics, and provide guidance on how students can further explore or understand the material.

If the student asks a question that doesn't match the available information, reassure them that it's okay to ask about different topics and answer based on your own {predicted_class} expertise. Be sure to maintain a positive and motivating tone, and encourage the student to stay curious and engaged.

Your response should be concise, informative, and tailored to the student's academic level.

USER:                    

QUESTION: '{query}'

ASSISTATNT:

ANSWER:
start the answer with the following:
                                    "as a {predicted_class} Teacher Agent , the answer to your question is as follows:"

""").format(query=query,predicted_class=predicted_class, confidence = confidence_score) 
    
    prompt_non_confidence_relevant  = ("""SYSTEM:

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
         start the answer with the following:
                                    "as a RAG Agent , the answer to your question is as follows:"

    
    """).format(query=query, relevant_passage_1=results[0][0], relevant_passage_2=results[1][0], relevant_passage_3=results[2][0])
    
    prompt_non_confidence_non_relevant = ("""SYSTEM:

You are a friendly, knowledgeable, and enthusiastic high school teacher. Your goal is to help students with their academic queries across various subjects, even when no specific documents are provided. As a teacher, you should explain concepts clearly and patiently, always offering encouragement and support. Use examples and analogies to simplify complex topics, and provide guidance on how students can further explore or understand the material.

If the student asks a question that doesn't match the available information, reassure them that it's okay to ask about different topics and answer based on your own expertise. Be sure to maintain a positive and motivating tone, and encourage the student to stay curious and engaged.

Your response should be concise, informative, and tailored to the student's academic level.

USER:                    

QUESTION: '{query}'

ASSISTATNT:

ANSWER:
    start the answer with the following:
                                    "the answer to your question is as follows:"

""").format(query=query)

    if confidence_score > 0.9:
        if results[0][1]<1.5:
            return prompt_confidence_relevant
        else:
            return prompt_confidence_not_relevant
    else:
        if results[0][1]<1.5:
            return prompt_non_confidence_relevant
        else:
            return prompt_non_confidence_non_relevant

def text_to_speech(text, output_audio_file):
    """Convert text to speech using Google Text-to-Speech API and save to an audio file."""
    
    # Initialize the client with service account credentials
    client = texttospeech.TextToSpeechClient()

    # Define the request payload
    synthesis_input = texttospeech.SynthesisInput(text=text)
    voice = texttospeech.VoiceSelectionParams(
        language_code="en-IN", name="en-US-Journey-F"
    )
    audio_config = texttospeech.AudioConfig(
        audio_encoding=texttospeech.AudioEncoding.LINEAR16,
        effects_profile_id=["small-bluetooth-speaker-class-device"],
    )

    # Send the request to the Google Text-to-Speech API
    response = client.synthesize_speech(
        input=synthesis_input, voice=voice, audio_config=audio_config
    )

    # Write audio content to file
    with open(output_audio_file, "wb") as out:
        out.write(response.audio_content)
    print(f"Audio content written to file: {output_audio_file}")

# latest_ai_response = ""
async def ai_response_generator(prompt: str) -> AsyncIterable[str]:
    # Create the formatted prompt
    generated_prompt = make_prompt(prompt)
    # print("This is an info log message get_answers")
    # Generate the content using Gemini API (not streaming, so we chunk manually)
    output = ""
    response = model.generate_content(generated_prompt ,stream=True)
    for chunk in response:
        output += chunk.text
        # latest_ai_response = output  # Update the latest response
        m = FastUI(root=[c.Markdown(text=output)])
        msg = f'data: {m.model_dump_json(by_alias=True, exclude_none=True)}\n\n'
        yield msg
        await asyncio.sleep(0.2)  
    # Send the message
    # while True:
    #     yield msg
    #     await asyncio.sleep(0.3)
    
    output = re.sub(r'[^a-zA-Z0-9.,!?-]+', ' ', output)
    text_to_speech(output, 'output_audio.wav')


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

@app.get('/{path:path}')
async def html_landing() -> HTMLResponse:
    """Simple HTML page which serves the React app, comes last as it matches all paths."""
    return HTMLResponse(prebuilt_html(title='FastUI Demo'))

# Chat form
class ChatForm(BaseModel):
    chat: str = Field(title=' ', max_length=1000)
