import streamlit as st
import os
import requests
from bs4 import BeautifulSoup
import numpy as np
from mistralai import Mistral
import faiss

# Set up your Mistral API key
api_key = "ajZS5VlpB8GFUXsx4Ugw2C7CdkR4wbmK"

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

# Keyword-based classifier for policy classification (intent detection)
def classify_intent(query):
    # Define keywords for each policy (simplified classification)
    policies_keywords = {
        "Academic Annual Leave Policy": ["annual leave", "leave", "vacation"],
        "Academic Appraisal Policy": ["appraisal", "performance review", "evaluation"],
        "Academic Appraisal Procedure": ["procedure", "appraisal procedure"],
        "Academic Credentials Policy": ["credentials", "qualifications", "degree"],
        "Academic Freedom Policy": ["academic freedom", "freedom of expression"],
        "Academic Members' Retention Policy": ["retention", "staff retention", "members retention"],
        "Academic Qualifications Policy": ["qualifications", "academic qualifications"],
        "Credit Hour Policy": ["credit hour", "credit", "hours"],
        "Intellectual Property Policy": ["intellectual property", "IP", "patents"],
        "Joint Appointment Policy": ["joint appointment", "dual role", "joint faculty"]
    }
    
    # Convert the query to lowercase for case-insensitive matching
    query = query.lower()
    
    # Match the query to the policy with the highest number of keyword matches
    best_match = None
    max_matches = 0
    
    for policy, keywords in policies_keywords.items():
        matches = sum(keyword in query for keyword in keywords)
        if matches > max_matches:
            best_match = policy
            max_matches = matches
    
    return best_match

# Streamlit Interface
# Updated to fix the error when accessing the index of selected_policy
def streamlit_app():
    st.title('UDST Policies Q&A')

    # Define policy URLs
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
        selected_policy = classify_intent(query)
        
        if selected_policy:
            st.write(f"Query is related to: {selected_policy}")
            
            # Convert the keys of the dictionary to a list to use index()
            policy_keys = list(policies_keywords.keys())
            
            # Find the index of the selected policy
            selected_policy_index = policy_keys.index(selected_policy)
            
            # Fetch policy data and chunk it
            selected_policy_url = policies[selected_policy_index]
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
        else:
            st.write("Sorry, I couldn't classify the query to a specific policy.")

# Run Streamlit app
if __name__ == '__main__':
    streamlit_app()

          
