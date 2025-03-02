import os
import faiss
import streamlit as st
import requests
from bs4 import BeautifulSoup
from sentence_transformers import SentenceTransformer
from mistralai.client import MistralClient
from mistralai.models.chat_completion import ChatMessage

# ---- SET UP API KEY ----
os.environ["MISTRAL_API_KEY"] = "liU22SqhcuY6W5ckIPxOzcm4yro1CJLX"  # Replace with your actual key
API_KEY = os.getenv("MISTRAL_API_KEY")

# ---- INITIALIZE AI MODELS ----
embedder = SentenceTransformer("all-MiniLM-L6-v2")  # Converts text to embeddings
client = MistralClient(api_key=API_KEY)  # Mistral AI Client

# ---- FUNCTION: SCRAPE UDST POLICIES ----
def scrape_policies():
    url = "https://www.udst.edu.qa/about-udst/institutional-excellence-ie/udst-policies-and-procedures"
    response = requests.get(url)
    soup = BeautifulSoup(response.text, "html.parser")
    
    # Extract text (modify as needed to get relevant policies)
    policies = [p.text.strip() for p in soup.find_all("p") if len(p.text.strip()) > 50]
    return policies[:10]  # Select at least 10 policies

# ---- FUNCTION: BUILD FAISS DATABASE ----
def build_faiss_index(policies):
    embeddings = embedder.encode(policies)
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)  # Create FAISS index
    index.add(embeddings)  # Add policy embeddings
    return index, policies

# ---- FUNCTION: FIND BEST MATCHING POLICY ----
def retrieve_policy(query, index, policies):
    query_embedding = embedder.encode([query])
    _, idx = index.search(query_embedding, 1)  # Find closest match
    return policies[idx[0][0]]

# ---- FUNCTION: GET ANSWER FROM MISTRAL AI ----
def get_answer(query, context):
    messages = [
        ChatMessage(role="system", content="You are an AI assistant answering questions based on UDST policies."),
        ChatMessage(role="user", content=f"Context: {context}\n\nQuestion: {query}")
    ]
    response = client.chat(messages=messages, model="mistral-medium")  # Choose model
    return response.choices[0].message.content

# ---- STREAMLIT UI ----
st.title("UDST Policy Chatbot 🤖")

# Load policies and build FAISS index
policies = scrape_policies()
index, policies = build_faiss_index(policies)

# Select policy from list
selected_policy = st.selectbox("Select a policy:", policies)

# User input query
query = st.text_input("Ask a question about the policy:")

if st.button("Get Answer"):
    if query:
        relevant_policy = retrieve_policy(query, index, policies)  # Retrieve best-matching policy
        answer = get_answer(query, relevant_policy)  # Get AI-generated response
        st.text_area("Answer:", answer, height=200)
    else:
        st.warning("Please enter a question!")
