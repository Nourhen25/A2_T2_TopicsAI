import streamlit as st
import os
import requests
from bs4 import BeautifulSoup
import numpy as np
import time
import faiss
from sklearn.metrics.pairwise import cosine_similarity
from mistralai import Mistral
import nltk
from tenacity import retry, stop_after_attempt, wait_exponential

# Download nltk tokenizer
nltk.download("punkt")
from nltk.tokenize import sent_tokenize

# 🔹 Secure API Key Handling
api_key = st.secrets["MISTRAL_API_KEY"]
os.environ["MISTRAL_API_KEY"] = api_key

# 🔹 Policies Data
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

# 🔹 Fetch and parse policy data
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

# 🔹 Smarter chunking using NLP
def chunk_text(text, chunk_size=512):
    sentences = sent_tokenize(text)
    chunks, current_chunk = [], ""

    for sentence in sentences:
        if len(current_chunk) + len(sentence) <= chunk_size:
            current_chunk += " " + sentence
        else:
            chunks.append(current_chunk.strip())
            current_chunk = sentence

    if current_chunk:
        chunks.append(current_chunk.strip())

    return chunks

# 🔹 Robust API call for text embeddings
@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=2))
@st.cache_data
def get_text_embedding(list_txt_chunks):
    client = Mistral(api_key=api_key)
    response = client.embeddings.create(model="mistral-embed", inputs=list_txt_chunks)
    return [e.embedding for e in response.data]

# 🔹 Create FAISS index safely
@st.cache_resource
def create_faiss_index(embeddings):
    if not embeddings or len(embeddings) == 0:
        st.error("No valid embeddings found. FAISS index not created.")
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

# 🔹 Retrieve relevant policy chunks
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

# 🔹 AI-based response generation
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

        if chat_response and chat_response.choices and chat_response.choices[0].message:
            return chat_response.choices[0].message.content.strip()
        else:
            return "Error: No valid response from Mistral API."
    except Exception as e:
        return f"Error generating response: {str(e)}"

# 🔹 Intent classification to find the best policy
def classify_policy(query):
    policy_descriptions = [p["description"] for p in policies.values()]
    query_embedding = get_text_embedding([query])[0]
    policy_embeddings = get_text_embedding(policy_descriptions)

    similarities = cosine_similarity([query_embedding], policy_embeddings)[0]
    best_match_index = np.argmax(similarities)
    best_policy = list(policies.keys())[best_match_index]

    return best_policy, policies[best_policy]["url"]

# 🔹 Streamlit UI
def streamlit_app():
    st.title("🔍 Agentic RAG - UDST Policies Q&A")

    query = st.text_input("Enter your question about UDST policies:")

    if st.button("Find Answer"):
        if query:
            # Step 1: Classify intent and select the best policy
            selected_policy, selected_policy_url = classify_policy(query)
            st.success(f"🔹 Automatically selected policy: **{selected_policy}**")

            # Step 2: Fetch and process policy content
            policy_text = fetch_policy_data(selected_policy_url)
            chunks = chunk_text(policy_text)
            embeddings = get_text_embedding(chunks)
            faiss_index = create_faiss_index(embeddings)

            # Step 3: Get query embedding and retrieve relevant content
            query_embedding = get_text_embedding([query])
            relevant_indexes = search_relevant_chunks(faiss_index, [query_embedding[0]], k=2)
            retrieved_chunks = [chunks[i] for i in relevant_indexes if i < len(chunks)]
            context = " ".join(retrieved_chunks)

            # Step 4: Generate the AI answer
            answer = mistral_answer(query, context)
            st.text_area("💡 Answer:", answer, height=200)

# Run Streamlit app
if __name__ == "__main__":
    streamlit_app()
