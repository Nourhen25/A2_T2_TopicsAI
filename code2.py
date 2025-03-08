import streamlit as st
import os
import requests
from bs4 import BeautifulSoup
import numpy as np
from mistralai import Mistral
import faiss
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB

# Set up your Mistral API key
api_key = "ajZS5VlpB8GFUXsx4Ugw2C7CdkR4wbmK"
os.environ["MISTRAL_API_KEY"] = api_key

# Define a dictionary for policies and their related keywords
policies_keywords = {
    "Academic Annual Leave Policy": ["annual leave", "leave", "vacation", "academic leave"],
    "Academic Appraisal Policy": ["appraisal", "evaluation", "assessment", "performance review"],
    "Academic Appraisal Procedure": ["appraisal procedure", "evaluation procedure", "assessment procedure"],
    "Academic Credentials Policy": ["academic credentials", "qualifications", "degree", "certifications"],
    "Academic Freedom Policy": ["freedom of expression", "academic freedom", "speech", "academic integrity"],
    "Academic Members’ Retention Policy": ["retention", "faculty retention", "academic retention"],
    "Academic Qualifications Policy": ["academic qualifications", "degree requirements", "degree qualifications"],
    "Credit Hour Policy": ["credit hour", "credits", "semester hours"],
    "Intellectual Property Policy": ["intellectual property", "IP", "patents", "copyright", "trademarks"],
    "Joint Appointment Policy": ["joint appointment", "dual role", "cross appointment"]
}

# Streamlit Interface
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

# Define a function to classify the intent of the query
def classify_intent(query):
    vectorizer = CountVectorizer(stop_words='english')
    classifier = MultinomialNB()

    # Prepare the training data
    train_texts = [
        "annual leave policy", "academic leave", "vacation time", "performance review", "evaluation of faculty",
        "qualification verification", "academic integrity", "freedom of speech", "faculty retention", 
        "degree requirements", "credit hour rules", "intellectual property policy", "joint appointment"
    ]
    
    # Map the labels to each policy
    train_labels = [
        "Academic Annual Leave Policy", "Academic Annual Leave Policy", "Academic Annual Leave Policy", 
        "Academic Appraisal Policy", "Academic Appraisal Policy", "Academic Credentials Policy", 
        "Academic Freedom Policy", "Academic Freedom Policy", "Academic Members’ Retention Policy", 
        "Academic Qualifications Policy", "Credit Hour Policy", "Intellectual Property Policy", 
        "Joint Appointment Policy"
    ]

    # Ensure the training texts and labels match in length
    assert len(train_texts) == len(train_labels), "Mismatch between training texts and labels lengths."
    
    # Vectorize the training data
    X_train = vectorizer.fit_transform(train_texts)
    y_train = train_labels

    # Train the classifier
    classifier.fit(X_train, y_train)

    # Vectorize the query and classify
    query_vec = vectorizer.transform([query])
    predicted_policy = classifier.predict(query_vec)[0]

    return predicted_policy

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

# Run Streamlit app
if __name__ == '__main__':
    streamlit_app()
