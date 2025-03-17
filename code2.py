import streamlit as st
import os
import requests
from bs4 import BeautifulSoup
import numpy as np
import time
from mistralai import Mistral
import faiss
from sklearn.metrics.pairwise import cosine_similarity

# Set up Mistral API Key
api_key = "liU22SqhcuY6W5ckIPxOzcm4yro1CJLX"
os.environ["MISTRAL_API_KEY"] = api_key

# Function to retry API calls with backoff
def call_mistral_api_with_retry(api_function, *args, retries=3, delay=2):
    for attempt in range(retries):
        try:
            return api_function(*args)
        except Exception as e:
            if "rate limit exceeded" in str(e).lower():
                time.sleep(delay * (2 ** attempt))  # Exponential backoff
            else:
                st.error(f"API Error: {e}")
                return None
    return None

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

# Get embeddings with caching and rate limit handling
@st.cache_data
def get_text_embedding(list_txt_chunks):
    client = Mistral(api_key=api_key)

    response = call_mistral_api_with_retry(client.embeddings.create, model="mistral-embed", inputs=list_txt_chunks)

    if response is None or not hasattr(response, 'data'):
        st.error("Error: Could not retrieve embeddings from Mistral API. Please try again later.")
        return []

    return [e.embedding for e in response.data]

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

# Generate AI-based answer with retry logic
def mistral_answer(query, context):
    if not context.strip():
        return "Sorry, I couldn't find relevant information in the selected policy."

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
    response = call_mistral_api_with_retry(client.chat.complete, model="mistral-large-latest", messages=messages)

    if response:
        return response.choices[0].message.content.strip()
    else:
        return "Error generating response due to API rate limit."

# Intent classification: Match question to relevant policy
def classify_policy(query):
    """Classifies the query and selects the most relevant policy."""
    policy_descriptions = [p["description"] for p in policies.values()]

    query_embedding = get_text_embedding([query])
    if not query_embedding:  
        st.error("Error: Could not generate query embedding. Try again later.")
        return None, None

    policy_embeddings = get_text_embedding(policy_descriptions)
    if not policy_embeddings:
        st.error("Error: Could not generate policy embeddings. Try again later.")
        return None, None

    similarities = cosine_similarity([query_embedding[0]], policy_embeddings)[0]
    best_match_index = np.argmax(similarities)

    best_policy = list(policies.keys())[best_match_index]
    return best_policy, policies[best_policy]["url"]

# Policies with short descriptions for classification
policies = {
    "Academic Annual Leave Policy": {
        "url": "https://www.udst.edu.qa/about-udst/institutional-excellence-ie/policies-and-procedures/academic-annual-leave-policy",
        "description": "Rules about annual leave, vacation days, and time-off for academic staff."
    },
    "Academic Appraisal Policy": {
        "url": "https://www.udst.edu.qa/about-udst/institutional-excellence-ie/policies-and-procedures/academic-appraisal-policy",
        "description": "Performance reviews, evaluations, and appraisals for faculty members."
    },
    "Academic Credentials Policy": {
        "url": "https://www.udst.edu.qa/about-udst/institutional-excellence-ie/policies-and-procedures/academic-credentials-policy",
        "description": "Regulations regarding academic qualifications, degrees, and credential recognition."
    },
    "Intellectual Property Policy": {
        "url": "https://www.udst.edu.qa/about-udst/institutional-excellence-ie/policies-and-procedures/intellectual-property-policy",
        "description": "Ownership of research, patents, and intellectual property rights."
    },
    "Credit Hour Policy": {
        "url": "https://www.udst.edu.qa/about-udst/institutional-excellence-ie/policies-and-procedures/credit-hour-policy",
        "description": "How credit hours are assigned and calculated for courses."
    }
}

# Streamlit UI
def streamlit_app():
    st.title("Agentic RAG - UDST Policies Q&A")

    query = st.text_input("Enter your Query:")

    if st.button("Find Answer"):
        if query:
            selected_policy, selected_policy_url = classify_policy(query)

            if not selected_policy:
                st.error("Could not classify policy due to an API issue. Please retry later.")
            else:
                st.success(f"Automatically selected policy: **{selected_policy}**")

                policy_text = fetch_policy_data(selected_policy_url)
                chunks = chunk_text(policy_text)

                embeddings = get_text_embedding(chunks)
                if not embeddings:
                    st.error("Could not generate embeddings for the selected policy. Try again later.")
                else:
                    faiss_index = create_faiss_index(embeddings)

                    query_embedding = get_text_embedding([query])
                    if not query_embedding:
                        st.error("Failed to process query embedding.")
                    else:
                        relevant_indexes = search_relevant_chunks(faiss_index, [query_embedding[0]], k=2)
                        retrieved_chunks = [chunks[i] for i in relevant_indexes if i < len(chunks)]
                        context = " ".join(retrieved_chunks)

                        answer = mistral_answer(query, context)
                        st.text_area("Answer:", answer, height=200)

# Run Streamlit app
if __name__ == "__main__":
    streamlit_app()
