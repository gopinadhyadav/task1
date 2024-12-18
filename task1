import pdfplumber
import openai
import faiss
import numpy as np
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS

# Initialize OpenAI API
openai.api_key = "your-openai-api-key"

# Function to extract text from PDF
def extract_text_from_pdf(file:///C:/Users/DELL/Pictures/gopi%20pdfs/browaer%20insert.pdf):
    with pdfplumber.open(file:///C:/Users/DELL/Pictures/gopi%20pdfs/browaer%20insert.pdf) as pdf:
        text = ""
        for page in pdf.pages:
            text += page.extract_text()
    return text

# Function to encode text using OpenAI's embeddings model
def get_embeddings(texts):
    embeddings = OpenAIEmbeddings()
    return embeddings.embed_documents(texts)

# Function to retrieve relevant documents using FAISS
def retrieve_relevant_docs(query, faiss_index, top_k=3):
    # Generate the query embedding
    query_embedding = get_embeddings([query])[0]
    # Perform similarity search
    distances, indices = faiss_index.search(np.array([query_embedding], dtype=np.float32), top_k)
    return indices, distances

# Function to generate a response from OpenAI's GPT model
def generate_response(context, query):
    prompt = f"Context:\n{context}\n\nQuestion: {query}\nAnswer:"
    response = openai.Completion.create(
        model="gpt-4", 
        prompt=prompt, 
        max_tokens=150
    )
    return response.choices[0].text.strip()

def main(pdf_path, query):
    # Step 1: Extract text from PDF
    text = extract_text_from_pdf(file:///C:/Users/DELL/Pictures/gopi%20pdfs/browaer%20insert.pdf)
    
    # Step 2: Split the text into chunks (e.g., paragraphs)
    chunks = text.split("\n\n")  # Split based on double newlines (paragraphs)

    # Step 3: Encode chunks using OpenAI embeddings
    embeddings = get_embeddings(chunks)

    # Step 4: Build FAISS index
    dimension = len(embeddings[0])  # Embedding size
    faiss_index = faiss.IndexFlatL2(dimension)  # Flat L2 distance index
    faiss_index.add(np.array(embeddings, dtype=np.float32))

    # Step 5: Retrieve relevant chunks based on query
    indices, _ = retrieve_relevant_docs(query, faiss_index, top_k=3)

    # Step 6: Combine retrieved chunks to form context
    context = "\n\n".join([chunks[i] for i in indices[0]])

    # Step 7: Generate response using OpenAI GPT
    response = generate_response(context, query)
    return response

if __name__ == "__main__":
    # Corrected file path
    pdf_path = "C:/Users/DELL/Pictures/gopi pdfs/browaer insert.pdf"  # Correct the path here
    user_query = "What is the main topic of the document?"
    
    # Chat with PDF
    answer = main(file:///C:/Users/DELL/Pictures/gopi%20pdfs/browaer%20insert.pdf, user_query)
    print("Answer:", answer)
