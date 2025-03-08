import streamlit as st
import os
import requests
from bs4 import BeautifulSoup
import numpy as np
from mistralai import Mistral
import faiss
from sklearn.preprocessing import LabelEncoder
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import TfidfVectorizer

# Set up your Mistral API key
api_key = "liU22SqhcuY6W5ckIPxOzcm4yro1CJLX"
os.environ["MISTRAL_API_KEY"] = api_key

# Sample intent classification data (can be expanded for better accuracy)
policies = [
    "academic-annual-leave-policy",
    "academic-appraisal-policy",
    "academic-appraisal-procedure",
    "academic-credentials-policy",
    "academic-freedom-policy",
    "academic-members-retention-policy",
    "academic-qualifications-policy",
    "credit-hour-policy",
    "intellectual-property-policy",
    "joint-appointment-policy"
]

policy_urls = {
    "academic-annual-leave-policy": "https://www.udst.edu.qa/about-udst/institutional-excellence-ie/policies-and-procedures/academic-annual-leave-policy",
    "academic-appraisal-policy": "https://www.udst.edu.qa/about-udst/institutional-excellence-ie/policies-and-procedures/academic-appraisal-policy",
    "academic-appraisal-procedure": "https://www.udst.edu.qa/about-udst/institutional-excellence-ie/policies-and-procedures/academic-appraisal-procedure",
    "academic-credentials-policy": "https://www.udst.edu.qa/about-udst/institutional-excellence-ie/policies-and-procedures/academic-credentials-policy",
    "academic-freedom-policy": "https://www.udst.edu.qa/about-udst/institutional-excellence-ie/policies-and-procedures/academic-freedom-policy",
    "academic-members-retention-policy": "https://www.udst.edu.qa/about-udst/institutional-excellence-ie/policies-and-procedures/academic-members%E2%80%99-retention-policy",
    "academic-qualifications-policy": "https://www.udst.edu.qa/about-udst/institutional-excellence-ie/policies-and-procedures/academic-qualifications-policy",
    "credit-hour-policy": "https://www.udst.edu.qa/about-udst/institutional-excellence-ie/policies-and-procedures/credit-hour-policy",
    "intellectual-property-policy": "https://www.udst.edu.qa/about-udst/institutional-excellence-ie/policies-and-procedures/intellectual-property-policy",
    "joint-appointment-policy": "https://www.udst.edu.qa/about-udst/institutional-excellence-ie/policies-and-procedures/joint-appointment-policy"
}

# Intent classifier (simple text classifier) trained once
vectorizer = TfidfVectorizer(stop_words='english')
le = LabelEncoder()

# Sample training data (can be expanded)
train_texts = [
    "What is the leave policy?",
    "How does the academic appraisal process work?",
    "What is the procedure for academic credentials?",
    "What does academic freedom entail?",
    "How do they retain academic staff?",
    "What are the qualifications needed for academic positions?",
    "How do credit hours work?",
    "What is the intellectual property policy?",
    "Explain the joint appointment policy."
]
train_labels = [
    "academic-annual-leave-policy",
    "academic-appraisal-policy",
    "academic-appraisal-procedure",
    "academic-credentials-policy",
    "academic-freedom-policy",
    "academic-members-retention-policy",
    "academic-qualifications-policy",
    "credit-hour-policy",
    "intellectual-property-policy",
    "joint-appointment-policy"
]

# Check lengths of training data
print(f"Length of train_texts: {len(train_texts)}")
print(f"Length of train_labels: {len(train_labels)}")

# Ensure they match
assert len(train_texts) == len(train_labels), "Mismatch between training texts and labels lengths."

# Vectorize training data
X_train = vectorizer.fit_transform(train_texts)
y_train = le.fit_transform(train_labels)

# Train the classifier once
classifier = MultinomialNB()
classifier.fit(X_train, y_train)

# Classify intent (which policy is related to the query)
def classify_intent(query):
    X_query = vectorizer.transform([query])
    predicted_label = classifier.predict(X_query)
    predicted_policy = le.inverse_transform(predicted_label)[0]
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

# Streamlit Interface
def streamlit_app():
    st.title('UDST Policies Q&A')

    # Input box for query
    query = st.text_input("Enter your Query:")

    if query:
        # Classify the intent of the query (i.e., which policy is it related to?)
        selected_policy = classify_intent(query)
        selected_policy_url = policy_urls[selected_policy]
        
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
