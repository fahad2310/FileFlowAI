import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
 
# Function to find similar documents based on user input
def find_similar_documents(user_input, data):
    # Drop rows with NaN values in 'Text_Content' column
    data = data.dropna(subset=['Text_Content'])
 
    if data.empty:
        print("Error: 'Text_Content' column contains only NaN values.")
        return None
 
    # TF-IDF vectorization
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(data['Text_Content'])
 
    # Transform user input to the same vector space
    user_tfidf = vectorizer.transform([user_input])
 
    # Compute cosine similarities
    similarities = cosine_similarity(user_tfidf, tfidf_matrix)
 
    # Find top N similar documents (adjust N as needed)
    N = 5
    top_indices = similarities.argsort()[0][-N:][::-1]
 
    # Extract similar PDF names
    similar_pdf_names = [data.iloc[i]['PDF_Name'] for i in top_indices]
 
    return similar_pdf_names
 
 
def main():
    st.title("Document Similarity using TF-IDF and Cosine Similarity")
 
    # Load data from CSV
    csv_file_path = r"D:\Projects\Synergy\final1 1.csv"  # Replace with the path to your CSV file
    data = pd.read_csv(csv_file_path)
 
    # Display user input text area
    user_input = st.text_area("Enter text for document similarity")
 
    if st.button("Find Similar Documents"):
        similar_pdf_names = find_similar_documents(user_input, data)
 
        if similar_pdf_names:
            st.header("Top Similar Documents:")
            for name in similar_pdf_names:
                st.text(f"Document: {name}")
        else:
            st.text("No similar documents found.")

    st.markdown(
    """<style>
        div.stButton > button {
            display: block;
            margin: 0 auto;
        }
    </style>""",
    unsafe_allow_html=True
)
 
if __name__ == "__main__":
    main()