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
    try:
        client = Mistral(api_key=api_key)
        messages = [{"role": "user", "content": prompt}]
        chat_response = client.chat.complete(model="mistral-large-latest", messages=messages)
        return chat_response.choices[0].message.content
    except Exception as e:
        st.error(f"Error occurred while generating answer: {e}")
        return "Sorry, I couldn't generate an answer at the moment."

# Classify the user's query to map to a policy
def classify_intent(query):
    policies_keywords = {
        "Academic Annual Leave Policy": ["leave", "vacation", "academic annual leave"],
        "Academic Appraisal Policy": ["appraisal", "academic evaluation", "performance review"],
        "Academic Credentials Policy": ["credentials", "academic qualifications", "degree"],
        "Academic Freedom Policy": ["freedom", "expression", "academic freedom"],
        "Academic Members’ Retention Policy": ["retention", "staff retention", "faculty retention"],
        "Academic Qualifications Policy": ["qualifications", "academic qualifications", "degree"],
        "Credit Hour Policy": ["credit hour", "credits", "credit system"],
        "Intellectual Property Policy": ["intellectual property", "patent", "IP"],
        "Joint Appointment Policy": ["joint appointment", "dual position", "collaboration"]
    }

    for policy, keywords in policies_keywords.items():
        for keyword in keywords:
            if keyword.lower() in query.lower():
                return policy
    return "Unknown Policy"

# Streamlit Interface
def streamlit_app():
    st.title('UDST Policies Q&A')

    # Input box for query
    query = st.text_input("Enter your Query:")
    
    if query:
        # Step 1: Classify the intent of the query (which policy is related to the question)
        selected_policy = classify_intent(query)
        st.write(f"Query is related to: {selected_policy}")

        # Step 2: Fetch the corresponding policy data and chunk it
        if selected_policy != "Unknown Policy":
            # Define policy URLs
            policies_urls = {
                "Academic Annual Leave Policy": "https://www.udst.edu.qa/about-udst/institutional-excellence-ie/policies-and-procedures/academic-annual-leave-policy",
                "Academic Appraisal Policy": "https://www.udst.edu.qa/about-udst/institutional-excellence-ie/policies-and-procedures/academic-appraisal-policy",
                "Academic Credentials Policy": "https://www.udst.edu.qa/about-udst/institutional-excellence-ie/policies-and-procedures/academic-credentials-policy",
                "Academic Freedom Policy": "https://www.udst.edu.qa/about-udst/institutional-excellence-ie/policies-and-procedures/academic-freedom-policy",
                "Academic Members’ Retention Policy": "https://www.udst.edu.qa/about-udst/institutional-excellence-ie/policies-and-procedures/academic-members%E2%80%99-retention-policy",
                "Academic Qualifications Policy": "https://www.udst.edu.qa/about-udst/institutional-excellence-ie/policies-and-procedures/academic-qualifications-policy",
                "Credit Hour Policy": "https://www.udst.edu.qa/about-udst/institutional-excellence-ie/policies-and-procedures/credit-hour-policy",
                "Intellectual Property Policy": "https://www.udst.edu.qa/about-udst/institutional-excellence-ie/policies-and-procedures/intellectual-property-policy",
                "Joint Appointment Policy": "https://www.udst.edu.qa/about-udst/institutional-excellence-ie/policies-and-procedures/joint-appointment-policy"
            }

            selected_policy_url = policies_urls[selected_policy]
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
            
            # Step 3: Generate an answer from the model
            answer = mistral_answer(query, context)
            
            # Display the answer
            st.text_area("Answer:", answer, height=200)
        else:
            st.error("Could not classify the query to any known policy.")

# Run Streamlit app
if __name__ == '__main__':
    streamlit_app()
