import streamlit as st
import os
import requests
from bs4 import BeautifulSoup
import numpy as np
import time
from mistralai import Mistral
import faiss

# Set up Mistral API Key
api_key = "liU22SqhcuY6W5ckIPxOzcm4yro1CJLX"
os.environ["MISTRAL_API_KEY"] = api_key

# Fetch and parse policy data
@st.cache_data
def fetch_policy_data(url):
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, "html.parser")
        tag = soup.find("div")
        return tag.text.strip() if tag else "No valid policy text found."
    except requests.RequestException as e:
        return f"Error fetching policy data: {str(e)}"

# Chunk text into smaller parts
def chunk_text(text, chunk_size=512):
    return [text[i:i+chunk_size] for i in range(0, len(text), chunk_size)]

# Get embeddings with retry logic
@st.cache_data
def get_text_embedding(list_txt_chunks, retries=3):
    for attempt in range(retries):
        try:
            client = Mistral(api_key=api_key)
            response = client.embeddings.create(model="mistral-embed", inputs=list_txt_chunks)
            return [e.embedding for e in response.data]
        except Exception as e:
            if attempt < retries - 1:
                time.sleep(2 ** attempt)  # Exponential backoff
            else:
                st.error(f"Error generating embeddings: {str(e)}")
                return []

# Create FAISS index
@st.cache_resource
def create_faiss_index(embeddings):
    if not embeddings:
        return None
    try:
        embedding_vectors = np.array(embeddings).astype('float32')
        d = embedding_vectors.shape[1]
        index = faiss.IndexFlatL2(d)
        faiss_index = faiss.IndexIDMap(index)
        faiss_index.add_with_ids(embedding_vectors, np.arange(len(embedding_vectors)))
        return faiss_index
    except Exception as e:
        st.error(f"Error creating FAISS index: {str(e)}")
        return None

# Search for relevant chunks
def search_relevant_chunks(faiss_index, query_embedding, k=2):
    if faiss_index is None:
        return []
    try:
        query_embedding = np.array(query_embedding).astype('float32')
        D, I = faiss_index.search(query_embedding, k)
        return I[0]
    except Exception as e:
        st.error(f"Error in FAISS search: {str(e)}")
        return []

# Generate AI-based answer
def mistral_answer(query, context):
    if not context.strip():
        return "Sorry, I couldn't find relevant information in the selected policy."
    
    try:
        prompt = f"""
        Context information is below:
        ---------------------
        {context}
        ---------------------
        Given the context information only, answer the query:
        Query: {query}
        Answer:
        """
        client = Mistral(api_key=api_key)
        messages = [{"role": "user", "content": prompt}]
        chat_response = client.chat.complete(model="mistral-large-latest", messages=messages)
        return chat_response.choices[0].message.content.strip()
    except Exception as e:
        return f"Error generating response: {str(e)}"

# Streamlit UI
def streamlit_app():
    st.title("UDST Policies Q&A")

    policies = {
        "Academic Annual Leave Policy": "https://www.udst.edu.qa/about-udst/institutional-excellence-ie/policies-and-procedures/academic-annual-leave-policy",
        "Academic Appraisal Policy": "https://www.udst.edu.qa/about-udst/institutional-excellence-ie/policies-and-procedures/academic-appraisal-policy",
        "Academic Appraisal Procedure": "https://www.udst.edu.qa/about-udst/institutional-excellence-ie/policies-and-procedures/academic-appraisal-procedure",
        "Academic Credentials Policy": "https://www.udst.edu.qa/about-udst/institutional-excellence-ie/policies-and-procedures/academic-credentials-policy",
        "Academic Freedom Policy": "https://www.udst.edu.qa/about-udst/institutional-excellence-ie/policies-and-procedures/academic-freedom-policy",
        "Academic Members' Retention Policy": "https://www.udst.edu.qa/about-udst/institutional-excellence-ie/policies-and-procedures/academic-members%E2%80%99-retention-policy",
        "Academic Qualifications Policy": "https://www.udst.edu.qa/about-udst/institutional-excellence-ie/policies-and-procedures/academic-qualifications-policy",
        "Credit Hour Policy": "https://www.udst.edu.qa/about-udst/institutional-excellence-ie/policies-and-procedures/credit-hour-policy",
        "Intellectual Property Policy": "https://www.udst.edu.qa/about-udst/institutional-excellence-ie/policies-and-procedures/intellectual-property-policy",
        "Joint Appointment Policy": "https://www.udst.edu.qa/about-udst/institutional-excellence-ie/policies-and-procedures/joint-appointment-policy"
    }

    selected_policy_name = st.selectbox("Select a Policy", list(policies.keys()))
    selected_policy_url = policies[selected_policy_name]

    # Prevent auto-execution by using a button
    if st.button("Load Policy"):
        st.session_state["policy_text"] = fetch_policy_data(selected_policy_url)
        if "Error" in st.session_state["policy_text"]:
            st.error(st.session_state["policy_text"])
        else:
            st.session_state["chunks"] = chunk_text(st.session_state["policy_text"])
            st.session_state["embeddings"] = get_text_embedding(st.session_state["chunks"])
            st.session_state["faiss_index"] = create_faiss_index(st.session_state["embeddings"])
            st.success(f"Loaded {selected_policy_name} successfully!")

    # Show query input only if a policy is loaded
    if "faiss_index" in st.session_state and st.session_state["faiss_index"]:
        query = st.text_input("Enter your Query:")
        if st.button("Get Answer"):
            query_embedding = get_text_embedding([query])
            if query_embedding:
                relevant_indexes = search_relevant_chunks(st.session_state["faiss_index"], [query_embedding[0]], k=2)
                retrieved_chunks = [st.session_state["chunks"][i] for i in relevant_indexes if i < len(st.session_state["chunks"])]
                context = " ".join(retrieved_chunks)
                answer = mistral_answer(query, context)
            else:
                answer = "Failed to process your query."

            st.text_area("Answer:", answer, height=200)

# Run Streamlit app
if __name__ == "__main__":
    streamlit_app()
