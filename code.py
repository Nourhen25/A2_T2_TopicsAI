import streamlit as st
import os
import requests
from bs4 import BeautifulSoup
import numpy as np
from mistralai import Mistral
import faiss

# Set up Mistral API Key
api_key = "liU22SqhcuY6W5ckIPxOzcm4yro1CJLX"
os.environ["MISTRAL_API_KEY"] = api_key

# Fetching and parsing policy data
def fetch_policy_data(url):
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()  # Raise error for bad responses
        soup = BeautifulSoup(response.text, "html.parser")
        tag = soup.find("div")  # Modify this if the content structure is different
        if tag:
            return tag.text.strip()
        else:
            return "No valid policy text found."
    except Exception as e:
        return f"Error fetching policy data: {str(e)}"

# Chunking function to break text into smaller parts
def chunk_text(text, chunk_size=512):
    return [text[i:i+chunk_size] for i in range(0, len(text), chunk_size)]

# Get embeddings for text chunks
def get_text_embedding(list_txt_chunks):
    try:
        client = Mistral(api_key=api_key)
        response = client.embeddings.create(model="mistral-embed", inputs=list_txt_chunks)
        return [e.embedding for e in response.data]
    except Exception as e:
        st.error(f"Error generating embeddings: {str(e)}")
        return []

# Initialize FAISS index
def create_faiss_index(embeddings):
    try:
        if not embeddings:
            return None
        embedding_vectors = np.array(embeddings).astype('float32')
        d = embedding_vectors.shape[1]  # Dimension of embeddings
        index = faiss.IndexFlatL2(d)
        faiss_index = faiss.IndexIDMap(index)
        faiss_index.add_with_ids(embedding_vectors, np.arange(len(embedding_vectors)))
        return faiss_index
    except Exception as e:
        st.error(f"Error creating FAISS index: {str(e)}")
        return None

# Search for relevant chunks
def search_relevant_chunks(faiss_index, query_embedding, k=2):
    try:
        if faiss_index is None:
            return []
        D, I = faiss_index.search(query_embedding, k)
        return I[0]  # Return top-k indexes
    except Exception as e:
        st.error(f"Error in FAISS search: {str(e)}")
        return []

# Mistral model to generate answers
def mistral_answer(query, context):
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

# Streamlit Interface
def streamlit_app():
    st.title("UDST Policies Q&A")

    # List of policies
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

    # Select policy from list
    selected_policy_name = st.selectbox("Select a Policy", list(policies.keys()))
    selected_policy_url = policies[selected_policy_name]

    # Fetch policy data
    st.write(f"Fetching policy: {selected_policy_name}")
    policy_text = fetch_policy_data(selected_policy_url)

    if "Error" in policy_text:
        st.error(policy_text)
        return

    # Chunk policy text
    chunks = chunk_text(policy_text)

    # Generate embeddings and create FAISS index
    embeddings = get_text_embedding(chunks)
    faiss_index = create_faiss_index(embeddings)

    # Input box for user query
    query = st.text_input("Enter your Query:")

    if query and faiss_index:
        # Embed the query
        query_embedding = get_text_embedding([query])
        if query_embedding:
            query_embedding = np.array([query_embedding[0]]).astype('float32')

            # Search for relevant chunks
            relevant_indexes = search_relevant_chunks(faiss_index, query_embedding, k=2)
            retrieved_chunks = [chunks[i] for i in relevant_indexes if i < len(chunks)]
            context = " ".join(retrieved_chunks)

            # Generate answer
            if context:
                answer = mistral_answer(query, context)
            else:
                answer = "Sorry, I couldn't find relevant information in the selected policy."

            # Display the answer
            st.text_area("Answer:", answer, height=200)
        else:
            st.error("Failed to generate query embedding.")

# Run Streamlit app
if __name__ == "__main__":
    streamlit_app()
