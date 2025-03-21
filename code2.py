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

# Policies and their descriptions for intent classification
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

# Intent classification: Match question to relevant policy
def classify_policy(query):
    policy_descriptions = [p["description"] for p in policies.values()]
    query_embedding = get_text_embedding([query])[0]
    policy_embeddings = get_text_embedding(policy_descriptions)

    similarities = cosine_similarity([query_embedding], policy_embeddings)[0]
    best_match_index = np.argmax(similarities)
    best_similarity_score = similarities[best_match_index]

    # Add a threshold to ensure the match is strong enough
    similarity_threshold = 0.7  # Adjust this value as needed
    if best_similarity_score < similarity_threshold:
        return None, None  # No good match found

    best_policy = list(policies.keys())[best_match_index]
    return best_policy, policies[best_policy]["url"]

# Streamlit UI
def streamlit_app():
    st.title("Agentic RAG - UDST Policies Q&A")

    query = st.text_input("Enter your Query:")
    
    if st.button("Find Answer"):
        if query:
            # Classify intent
            selected_policy, selected_policy_url = classify_policy(query)

            if selected_policy is None:
                st.warning("No relevant policy found for your query. Please try rephrasing or ask a different question.")
            else:
                st.success(f"Automatically selected policy: **{selected_policy}**")

                # Fetch and process policy
                policy_text = fetch_policy_data(selected_policy_url)
                chunks = chunk_text(policy_text)
                embeddings = get_text_embedding(chunks)
                faiss_index = create_faiss_index(embeddings)

                # Get query embedding and retrieve relevant context
                query_embedding = get_text_embedding([query])
                relevant_indexes = search_relevant_chunks(faiss_index, [query_embedding[0]], k=2)
                retrieved_chunks = [chunks[i] for i in relevant_indexes if i < len(chunks)]
                context = " ".join(retrieved_chunks)

                # Generate answer
                answer = mistral_answer(query, context)
                st.text_area("Answer:", answer, height=200)

# Run Streamlit app
if __name__ == "__main__":
    streamlit_app()
