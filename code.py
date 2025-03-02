import streamlit as st
import os
import requests
from bs4 import BeautifulSoup
import numpy as np
from mistralai import Mistral
import faiss

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
    # Convert the embeddings to a 2D NumPy array
    embedding_vectors = np.array([embedding.embedding for embedding in embeddings])
    
    # Ensure the shape is (num_embeddings, embedding_size)
    d = embedding_vectors.shape[1]  # embedding size (dimensionality)
    
    # Create the FAISS index and add the embeddings
    index = faiss.IndexFlatL2(d)
    faiss_index = faiss.IndexIDMap(index)
    
    # Add the embeddings with an id for each (FAISS requires ids)
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

# Streamlit Interface
def streamlit_app():
    st.title('UDST Policies Q&A')

    # Select a policy from a list of 10 policies (example URLs)
    policies = [
        "https://www.udst.edu.qa/about-udst/institutional-excellence-ie/policies-and-procedures/sport-and-wellness/",
        "https://www.udst.edu.qa/about-udst/institutional-excellence-ie/policies-and-procedures/student-conduct/",
        "https://www.udst.edu.qa/about-udst/institutional-excellence-ie/policies-and-procedures/academic-integrity/",
        "https://www.udst.edu.qa/about-udst/institutional-excellence-ie/policies-and-procedures/academic-freedom-policy",
        "https://www.udst.edu.qa/about-udst/institutional-excellence-ie/policies-and-procedures/academic-members%E2%80%99-retention-policy",
        "https://www.udst.edu.qa/about-udst/institutional-excellence-ie/policies-and-procedures/academic-professional-development",
        "https://www.udst.edu.qa/about-udst/institutional-excellence-ie/policies-and-procedures/credit-hour-policy",
        "https://www.udst.edu.qa/about-udst/institutional-excellence-ie/policies-and-procedures/examination-policy",
        "https://www.udst.edu.qa/about-udst/institutional-excellence-ie/policies-and-procedures/program-accreditation-policy",
        "https://www.udst.edu.qa/about-udst/institutional-excellence-ie/policies-and-procedures/intellectual-property-policy"
                # Add more URLs as needed
    ]
    selected_policy_url = st.selectbox('Select a Policy', policies)
    
    # Fetch policy data and chunk it
    policy_text = fetch_policy_data(selected_policy_url)
    chunks = chunk_text(policy_text)
    
    # Generate embeddings for the chunks and create a FAISS index
    embeddings = get_text_embedding(chunks)
    faiss_index = create_faiss_index(embeddings)
    
    # Input box for query
    query = st.text_input("Enter your Query:")
    
    if query:
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
