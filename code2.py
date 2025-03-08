import streamlit as st
import os
import requests
from bs4 import BeautifulSoup
import numpy as np
from mistralai import Mistral
import faiss

# Set up your Mistral API key
api_key = os.getenv("MISTRAL_API_KEY")  # Fetch from environment
if not api_key:
    st.error("MISTRAL_API_KEY is not set! Please provide a valid API key.")
    st.stop()

# Function to fetch and parse policy data
def fetch_policy_data(url):
    try:
        response = requests.get(url)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, "html.parser")
        tag = soup.find("div")
        return tag.text.strip() if tag else "No content found."
    except Exception as e:
        st.error(f"Failed to fetch policy data: {e}")
        return ""

# Chunk text into smaller parts for embeddings
def chunk_text(text, chunk_size=512):
    return [text[i:i+chunk_size] for i in range(0, len(text), chunk_size)]

# Generate embeddings using Mistral
def get_text_embedding(text_list):
    if not text_list:
        return []
    
    client = Mistral(api_key=api_key)
    try:
        embeddings_batch = client.embeddings.create(model="mistral-embed", inputs=text_list)
        return [e.embedding for e in embeddings_batch.data]
    except Exception as e:
        st.error(f"Error fetching embeddings: {e}")
        return []

# Create FAISS index for fast retrieval
def create_faiss_index(embeddings):
    if not embeddings:
        return None

    embedding_vectors = np.array(embeddings, dtype=np.float32)
    d = embedding_vectors.shape[1]
    
    index = faiss.IndexFlatL2(d)
    faiss_index = faiss.IndexIDMap(index)
    faiss_index.add_with_ids(embedding_vectors, np.array(range(len(embedding_vectors))))
    
    return faiss_index

# Search for the most relevant chunks
def search_relevant_chunks(faiss_index, query_embedding, k=2):
    if faiss_index is None or query_embedding is None:
        return []
    
    query_embedding = np.array([query_embedding], dtype=np.float32)
    D, I = faiss_index.search(query_embedding, k)
    return I[0]

# Generate an answer using Mistral
def mistral_answer(query, context):
    prompt = f"""
    Context information is below.
    ---------------------
    {context}
    ---------------------
    Given the context information and not prior knowledge, answer the query.
    Query: {query}
    Answer:
    """
    client = Mistral(api_key=api_key)
    try:
        messages = [{"role": "user", "content": prompt}]
        chat_response = client.chat.complete(model="mistral-large-latest", messages=messages)
        return chat_response.choices[0].message.content
    except Exception as e:
        return f"Error generating response: {e}"

# Streamlit Interface
def streamlit_app():
    st.title("UDST Policies Q&A")

    # Policy URLs
    policies = {
        "Sport and Wellness": "https://www.udst.edu.qa/about-udst/institutional-excellence-ie/policies-and-procedures/sport-and-wellness/",
        "Student Conduct": "https://www.udst.edu.qa/about-udst/institutional-excellence-ie/policies-and-procedures/student-conduct/",
        "Academic Integrity": "https://www.udst.edu.qa/about-udst/institutional-excellence-ie/policies-and-procedures/academic-integrity/",
        "Academic Freedom": "https://www.udst.edu.qa/about-udst/institutional-excellence-ie/policies-and-procedures/academic-freedom-policy",
        "Retention Policy": "https://www.udst.edu.qa/about-udst/institutional-excellence-ie/policies-and-procedures/academic-members%E2%80%99-retention-policy",
        "Professional Development": "https://www.udst.edu.qa/about-udst/institutional-excellence-ie/policies-and-procedures/academic-professional-development",
        "Credit Hour Policy": "https://www.udst.edu.qa/about-udst/institutional-excellence-ie/policies-and-procedures/credit-hour-policy",
        "Examination Policy": "https://www.udst.edu.qa/about-udst/institutional-excellence-ie/policies-and-procedures/examination-policy",
        "Program Accreditation": "https://www.udst.edu.qa/about-udst/institutional-excellence-ie/policies-and-procedures/program-accreditation-policy",
        "Intellectual Property": "https://www.udst.edu.qa/about-udst/institutional-excellence-ie/policies-and-procedures/intellectual-property-policy",
    }

    # Fetch and preprocess policy data
    st.subheader("Fetching Policy Data...")
    policy_texts = {name: fetch_policy_data(url) for name, url in policies.items()}
    chunked_policies = {name: chunk_text(text) for name, text in policy_texts.items()}

    # Generate policy embeddings
    st.subheader("Generating Policy Embeddings...")
    policy_embeddings = {name: get_text_embedding(chunks) for name, chunks in chunked_policies.items()}

    # Create FAISS index per policy
    policy_indexes = {name: create_faiss_index(embeds) for name, embeds in policy_embeddings.items()}

    # User query input
    query = st.text_input("Enter your Query:")

    if query:
        # Generate query embedding
        query_embedding = get_text_embedding([query])[0] if get_text_embedding([query]) else None
        if not query_embedding:
            st.error("Failed to generate query embedding.")
            return

        # Classify intent (Find the most relevant policy)
        st.subheader("Classifying Intent...")
        best_policy = None
        best_score = float("inf")

        for policy_name, faiss_index in policy_indexes.items():
            if faiss_index:
                score = search_relevant_chunks(faiss_index, query_embedding, k=1)
                if len(score) > 0 and score[0] < best_score:
                    best_score = score[0]
                    best_policy = policy_name

        if not best_policy:
            st.error("No relevant policy found.")
            return

        st.success(f"Most Relevant Policy: {best_policy}")

        # Retrieve relevant chunks from the best policy
        faiss_index = policy_indexes[best_policy]
        chunks = chunked_policies[best_policy]
        I = search_relevant_chunks(faiss_index, query_embedding, k=2)
        retrieved_chunks = [chunks[i] for i in I.tolist() if i < len(chunks)]
        context = " ".join(retrieved_chunks)

        # Generate and display the answer
        answer = mistral_answer(query, context)
        st.text_area("Answer:", answer, height=200)

# Run the Streamlit app
if __name__ == '__main__':
    streamlit_app()
