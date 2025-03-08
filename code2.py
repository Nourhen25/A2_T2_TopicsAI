import streamlit as st
import os
import requests
from bs4 import BeautifulSoup
import numpy as np
from mistralai import Mistral
import faiss
from sklearn.metrics.pairwise import cosine_similarity

# Set up your Mistral API key
api_key = "liU22SqhcuY6W5ckIPxOzcm4yro1CJLX"
os.environ["MISTRAL_API_KEY"] = api_key

# Fetching and parsing policy data
def fetch_policy_data(url):
    response = requests.get(url)
    html_doc = response.text
    soup = BeautifulSoup(html_doc, "html.parser")
    tag = soup.find("div")
    text = tag.text.strip()
    return text

# Chunking function to break text into smaller parts
def chunk_text(text, chunk_size=512):
    return [text[i:i+chunk_size] for i in range(0, len(text), chunk_size)]

# Get embeddings for text chunks
def get_text_embedding(list_txt_chunks):
    client = Mistral(api_key=api_key)
    embeddings_batch_response = client.embeddings.create(model="mistral-embed", inputs=list_txt_chunks)
    return embeddings_batch_response.data

# Initialize FAISS index
def create_faiss_index(embeddings):
    embedding_vectors = np.array([embedding.embedding for embedding in embeddings])
    d = embedding_vectors.shape[1]  # embedding size (dimensionality)
    
    index = faiss.IndexFlatL2(d)
    faiss_index = faiss.IndexIDMap(index)
    
    faiss_index.add_with_ids(embedding_vectors, np.array(range(len(embedding_vectors))))
    
    return faiss_index

# Search for the most relevant chunks based on query embedding
def search_relevant_chunks(faiss_index, query_embedding, k=2):
    D, I = faiss_index.search(query_embedding, k)
    return I

# Mistral model to generate answers based on context
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
    messages = [{"role": "user", "content": prompt}]
    chat_response = client.chat.complete(model="mistral-large-latest", messages=messages)
    return chat_response.choices[0].message.content

# Intent classification (basic keyword matching or similarity-based)
def classify_intent(query, policies):
    policy_titles = [policy.split('/')[-1] for policy in policies]  # Get last part of URL (policy name)
    
    # Generate embeddings for policy titles and the query
    policy_embeddings = get_text_embedding(policy_titles)
    query_embedding = get_text_embedding([query])[0].embedding.reshape(1, -1)
    
    # Calculate cosine similarity between the query and each policy title
    similarities = [cosine_similarity(query_embedding, np.array([embedding.embedding]).reshape(1, -1))[0][0] for embedding in policy_embeddings]
    
    # Get the index of the most similar policy
    best_match_index = np.argmax(similarities)
    return policies[best_match_index], policy_titles[best_match_index]

# Streamlit Interface
def streamlit_app():
    st.title('UDST Policies Q&A')

    # Select a policy from a list of 10 policies (example URLs)
    policies = [
        "https://www.udst.edu.qa/about-udst/institutional-excellence-ie/policies-and-procedures/academic-annual-leave-policy",
        "https://www.udst.edu.qa/about-udst/institutional-excellence-ie/policies-and-procedures/academic-appraisal-policy",
        "https://www.udst.edu.qa/about-udst/institutional-excellence-ie/policies-and-procedures/academic-appraisal-procedure",
        "https://www.udst.edu.qa/about-udst/institutional-excellence-ie/policies-and-procedures/academic-credentials-policy",
        "https://www.udst.edu.qa/about-udst/institutional-excellence-ie/policies-and-procedures/academic-freedom-policy",
        "https://www.udst.edu.qa/about-udst/institutional-excellence-ie/policies-and-procedures/academic-members%E2%80%99-retention-policy",
        "https://www.udst.edu.qa/about-udst/institutional-excellence-ie/policies-and-procedures/academic-qualifications-policy",
        "https://www.udst.edu.qa/about-udst/institutional-excellence-ie/policies-and-procedures/credit-hour-policy",
        "https://www.udst.edu.qa/about-udst/institutional-excellence-ie/policies-and-procedures/intellectual-property-policy",
        "https://www.udst.edu.qa/about-udst/institutional-excellence-ie/policies-and-procedures/joint-appointment-policy"
    ]
    
    # Input box for query
    query = st.text_input("Enter your Query:")

    if query:
        # Classify the intent (which policy is related to the query)
        selected_policy_url, selected_policy_title = classify_intent(query, policies)
        st.write(f"Classified intent to the policy: {selected_policy_title}")
        
        # Fetch policy data and chunk it
        policy_text = fetch_policy_data(selected_policy_url)
        chunks = chunk_text(policy_text)
        
        # Generate embeddings for the chunks and create a FAISS index
        embeddings = get_text_embedding(chunks)
        faiss_index = create_faiss_index(embeddings)
        
        # Embed the user query and search for relevant chunks
        query_embedding = np.array([get_text_embedding([query])[0].embedding])
        I = search_relevant_chunks(faiss_index, query_embedding, k=2)
        retrieved_chunks = [chunks[i] for i in I.tolist()[0]]
        context = " ".join(retrieved_chunks)
        
        # Generate answer from the model
        answer = mistral_answer(query, context)
        
        # Display the answer
        st.text_area("Answer:", answer, height=200)

# Run Streamlit app
if __name__ == '__main__':
    streamlit_app()
