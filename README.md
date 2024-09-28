# NCERT RAG Agent with Subject Classification

This project implements a Retrieval Augmented Generation (RAG) system using NCERT textbook data, enhanced with subject classification and basic agent capabilities.  It allows users to query the NCERT dataset and receive contextually relevant answers, leveraging a vector database for retrieval and a large language model (LLM) for generation.  The system also classifies the query into different subjects (Physics, Chemistry, Biology) to tailor the LLM prompt for more accurate responses.  Finally, it includes text-to-speech functionality for audio output.

## Live Demo video 

[Watch the video](https://www.youtube.com/watch?v=your_video_id)


## Features

* **RAG System:** Uses FAISS for efficient vector search and retrieval of relevant information from the NCERT dataset.
* **LLM Integration:** Employs Google's Gemini API with the `gemini-1.5-flash-001` model for generating comprehensive answers.
* **Smart Retrieval:**  Determines relevance of retrieved documents based on distance score between the query and retrieved documents. Only uses retrieved context if the distance is below a threshold (1.5).
* **Subject Classification:** Fine-tuned Roberta model classifies queries into Physics, Chemistry, and Biology.  Uses confidence scores to tailor the LLM prompt for subject-specific responses.
* **Custom Prompts:**  Dynamically generates prompts based on both retrieval relevance and subject classification.The start of the prompt is specially tailored to show if the model is using the retrived documents and Subject classification or not . 
* **Text-to-Speech:** Uses Google's Text-to-Speech API to generate audio responses (Sarvam AI API integration attempted but encountered issues).
* **FastAPI Backend:** Exposes endpoints for query processing and communication with the frontend.
* **FastUI Frontend:** Simple and modern user interface for interacting with the RAG system.

## Installation

1. Clone the repository: bash`git clone https://github.com/suyashsethia/RAG-Model-for-NCERT`
2. Install dependencies: `pip install -r requirements.txt`


## Usage

1. Run the FastAPI server: `uvicorn main:app --reload`
2. Access the frontend in your browser at `http://127.0.0.1:8000`.
3. Enter your query in the provided text box and submit.
4. The generated response will be displayed, along with a downloadable audio file (`output_audio.wav`).

## Code Structure

* **`main.py`:**  FastAPI application, defines API endpoints and handles communication with the frontend.
* **`code.ipynb`:** Jupyter notebook containing all the preprocessing steps, FAISS implementation, Roberta model training, Sarvam AI and Google TTS API code, and other core logic.
* **`memory.json`:**  FAISS index containing vector embeddings of the NCERT data.
* **`requirements.txt`:**  List of project dependencies.
* **`roberta_classification_model.pt`:** Saved Roberta model for subject classification.


## Approach and Implementation Details

* **Data Preprocessing:**  NCERT PDF data is extracted, cleaned, and tokenized into sentences. Sentences are grouped into documents of comparable length.
* **Vector Database:**  `all-MiniLM-L6-v2` is used to generate sentence embeddings, which are stored in FAISS for efficient similarity search.
* **Roberta Fine-tuning:**  A Roberta model is fine-tuned on a dataset (linked in the project description) to classify queries into Physics, Chemistry, and Biology.
* **Prompt Engineering:**  LLM prompts are dynamically generated based on the following logic:
    * If the distance score between the query and retrieved documents is below a threshold, the retrieved context is included in the prompt.
    * If the Roberta model classifies the query with high confidence (> 0.9), a subject-specific prompt is used.
* **Text-to-Speech:**  Due to issues with the Sarvam AI API, Google's Text-to-Speech API is used for generating audio responses.


## Future Improvements

* **Enhanced Agent Capabilities:** Implement more complex agent actions and tools.
* **Improved User Interface:** Enhance the FastUI frontend with features like audio playback directly in the browser.
* **Explore Alternative LLMs:**  Experiment with other LLMs and compare performance.
* **Robust Error Handling:** Implement better error handling and user feedback mechanisms.
* **Sarvam AI API Integration:**  Revisit Sarvam AI API integration once the server issues are resolved.



## Dataset

NCERT PDF Data: [link provided in original assignment]
Subject Classification Dataset: [https://www.kaggle.com/datasets/vivmankar/physics-vs-chemistry-vs-biology](https://www.kaggle.com/datasets/vivmankar/physics-vs-chemistry-vs-biology)


## Acknowledgements

This project utilizes Google's Gemini API and FAISS library.  Thanks to the providers of the NCERT dataset and the subject classification dataset.Thanks to SarvamAI to provide me with the opportunity to do thsi worderfull project .  
