import streamlit as st
import os
import requests
from bs4 import BeautifulSoup
import numpy as np
from mistralai import Mistral
import faiss

# Set up Mistral API key
api_key = os.getenv("MISTRAL_API_KEY")

# List of policy URLs
policies = {
    "Sport and Wellness": "https://www.udst.edu.qa/about-udst/institutional-excellence-ie/policies-and-procedures/sport-and-wellness/",
    "Student Conduct": "https://www.udst.edu.qa/about-udst/institutional-excellence-ie/policies-and-procedures/student-conduct/",
    "Academic Integrity": "https://www.udst.edu.qa/about-udst/institutional-excellence-ie/policies-and-procedures/academic-integrity/",
    "Academic Freedom": "https://www.udst.edu.qa/about-udst/institutional-excellence-ie/policies-and-procedures/academic-freedom-policy",
    "Academic Retention": "https://www.udst.edu.qa/about-udst/institutional-excellence-ie/policies-and-procedures/academic-members%E2%80%99-retention-policy",
    "Professional Development": "https://www.udst.edu.qa/about-udst/institutional-excellence-ie/policies-and-procedures/academic-professional-development",
    "Credit Hour": "https://www.udst.edu.qa/about-udst/institutional-excellence-ie/policies-and-procedures/credit-hour-policy",
    "Examination": "https://www.udst.edu.qa/about-udst/institutional-excellence-ie/policies-and-procedures/examination-policy",
    "Program Accreditation": "https://www.udst.edu.qa/about-udst/institutional-excellence-ie/policies-and-procedures/program-accreditation-policy",
    "Intellectual Property": "https://www.udst.edu.qa/about-udst/institutional-excellence-ie/policies-and-procedures/intellectual-property-policy"
}

# Function to fetch and clean policy data
def fetch_policy_data(url):
    response = requests.get(url)
    soup = BeautifulSoup(response.text, "html.parser")
    text = " ".join([p.text.strip() for p in soup.find_all("p")])
    return text

# Chunking text for embedding processing
def chunk_text(text, chunk_size=512):
    return [text[i:i+chunk_size] for i in range(0, len(text), chunk_size)]

# Get text embeddings using Mistral
def get_text_embedding(text_list):
    client = Mistral(api_key=api_key)
    embeddings_batch = client.embeddings.create(model="mistral-embed", inputs=text_list)
    return [e.embedding for e in embeddings_batch.data]

# Build FAISS index for a policy
def create_faiss_index(embeddings):
    d = len(embeddings[0])
    index = faiss.IndexFlatL2(d)
    index.add(np.array(embeddings, dtype=np.float32))
    return index

# Find the closest matching policy
def classify_intent(query_embedding, policy_embeddings):
    query_vector = np.array([query_embedding], dtype=np.float32)
    distances = {policy: np.linalg.norm(np.array(policy_embeddings[policy]) - query_vector) for policy in policies}
    return min(distances, key=distances.get)

# Retrieve relevant chunks from FAISS index
def search_relevant_chunks(faiss_index, query_embedding, chunks, k=2):
    D, I = faiss_index.search(np.array([query_embedding], dtype=np.float32), k)
    return " ".join([chunks[i] for i in I[0]])

# Generate answer using Mistral
def mistral_answer(query, context):
    prompt = f"""
    Context information:
    {context}
    ---------------------
    Given the context, answer the query:
    Query: {query}
    Answer:
    """
    client = Mistral(api_key=api_key)
    response = client.chat.complete(model="mistral-large-latest", messages=[{"role": "user", "content": prompt}])
    return response.choices[0].message.content

# Streamlit UI
def streamlit_app():
    st.title("UDST Policies Q&A Chatbot")
    
    # Precompute policy embeddings for classification
    st.session_state.policy_embeddings = {
        policy: get_text_embedding([policy]) for policy in policies
    }
    
    # User query input
    query = st.text_input("Enter your query about UDST policies:")
    
    if query:
        query_embedding = get_text_embedding([query])[0]
        relevant_policy = classify_intent(query_embedding, st.session_state.policy_embeddings)
        
        # Fetch and process policy data
        policy_text = fetch_policy_data(policies[relevant_policy])
        chunks = chunk_text(policy_text)
        chunk_embeddings = get_text_embedding(chunks)
        faiss_index = create_faiss_index(chunk_embeddings)
        
        # Retrieve relevant text
        context = search_relevant_chunks(faiss_index, query_embedding, chunks, k=2)
        answer = mistral_answer(query, context)
        
        # Display results
        st.subheader(f"Relevant Policy: {relevant_policy}")
        st.text_area("Answer:", answer, height=200)

if __name__ == "__main__":
    streamlit_app()
