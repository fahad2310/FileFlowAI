import streamlit as st
from PyPDF2 import PdfReader
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import streamlit as st
import os
import google.generativeai as genai
from dotenv import load_dotenv

load_dotenv()
st.markdown('<link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/4.7.0/css/font-awesome.min.css">', unsafe_allow_html=True)

# Configure the API key securely (ensure you have a valid API key)
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

# Create the generative model
model = genai.GenerativeModel("gemini-pro")


def replace_last_part(original_string, replacement_text):
    # Find the last occurrence of '/'
    last_slash_index = original_string.rfind('\\')

    if last_slash_index != -1:
        # Extract the part after the last '/'
        remaining_text = original_string[last_slash_index + 1:]

        # Replace the target part with the replacement text
        modified_text = original_string.replace(remaining_text,replacement_text)

        return modified_text
    else:
        # If '/' is not found, return the original string
        return original_string


def get_generated_text(prompt, text1):
    try:
        # Combine prompt, text1, and text2 with separators
        combined_input = f"{prompt}\n{text1}\n"
        response = model.generate_content(combined_input)
        return response.text
    except Exception as e:
        print(f"Error generating text: {e}")
        return "Error occurred. Please try again."


def extract_text_with_pyPDF(pdf_file):
    pdf_reader = PdfReader(pdf_file)
    raw_text = ''

    for i, page in enumerate(pdf_reader.pages):
        text = page.extract_text()
        if text:
            raw_text += text

    return raw_text


def find_similar_documents(user_input, csv_file_path):
    try:
        # Read data from CSV (with error handling)
        data = pd.read_csv(csv_file_path)

        # Try multiple options to handle missing or different column names
        try:
            # Extract text and PDF names (assuming column names are in the header)
            documents = data['Text_Content'].tolist()
            pdf_names = data['PDF_Name'].tolist()

        except KeyError:  # Handle missing column name 'Text_Content'
            print("Error: 'Text_Content' column not found in the CSV file.")
            return None

        else:
            # Handle cases where the column name might be different:
            # Check if 'Text_Content' exists, otherwise use the first column
            if 'Text_Content' in data.columns:
                documents = data['Text_Content'].tolist()
            else:
                documents = data.iloc[:, 0].tolist()

            # Check if 'PDF_Name' exists, otherwise use the second column
            if 'PDF_Name' in data.columns:
                pdf_names = data['PDF_Name'].tolist()
            else:
                pdf_names = data.iloc[:, 1].tolist()

        # Handle empty values (more comprehensive approach)
        data = data.dropna(subset=['Text_Content', 'PDF_Name'])
        documents = data['Text_Content'].tolist()
        pdf_names = data['PDF_Name'].tolist()

        # Thoroughly check for any remaining nan values
        if any(pd.isna(documents)):
            print("Error: There are still nan values in the documents list.")
            return None

        # TF-IDF vectorization
        vectorizer = TfidfVectorizer()
        tfidf_matrix = vectorizer.fit_transform(documents)

        # Transform user input to the same vector space
        user_tfidf = vectorizer.transform([user_input])

        # Compute cosine similarities
        similarities = cosine_similarity(user_tfidf, tfidf_matrix)

        # Find top N similar documents (adjust N as needed)
        N = 1
        top_indices = similarities.argsort()[0][-N:][::-1]

        # Extract similar PDF names
        similar_pdf_names = [pdf_names[i] for i in top_indices]

        return similar_pdf_names

    except FileNotFoundError as e:
        print(f"Error: CSV file '{csv_file_path}' not found.")
        return None


def main():
    st.title("FileFlow AI")
    st.header("OCR and File Path Generation")

    document_type = st.selectbox("Select Document Type", ["Plans", "General Documents"])
    uploaded_file = st.file_uploader("Upload a PDF file", type=["pdf"])

    if uploaded_file is not None:
        st.text("Performing OCR... Please wait.")
        csv_file_path = r"D:\Projects\Synergy\final1 1.csv"  # Replace with the actual path to your CSV file
        text_with_pyPDF = extract_text_with_pyPDF(uploaded_file)

        text1 = """
            Given an image of a document, extract and provide the following information in the format Version_DrawingNumber_ProjectNumber.pdf:

            1. Identify and extract the version information.(if available)
            2. Identify and extract the drawing number. (if available)
            3. Identify and extract the project number (if available).

            Please format the output as follows:
            Version_DrawingNumber_ProjectNumber.pdf

            Example:
            For a document with Version: V1.2, Drawing Number: ABC123, and Project Number: PRO456,
            the output should be: V1.2_ABC123_PRO456.pdf
            """

        text2 = """
        Train the OCR model to prioritize and extract vital information from plans. Emphasize the identification and labeling of the following:

        Important Data (Priority):

        Crucial project data
        Critical document details
        Key dates and deadlines
        Categorization (e.g., Report, Acoustics)
        Author information
        Any other imperative data
        Static Information:

        Project name
        Project number
        Address
        Client
        Consultant
        Dynamic Content:

        Drawing titles
        Drawing numbers
        Revisions
        Statuses
        Ensure the OCR model adapts to diverse layouts and variations, striving for accurate extraction of all essential plan details.
        """

        generated_text = get_generated_text(text_with_pyPDF, text1)
        generated_text_2 = get_generated_text(text_with_pyPDF, text2)
        st.header("Generated Text:")

        # Set the default value of the text area to the extracted text
        user_input = st.text_area("Enter text for document similarity", generated_text_2, height=100)

        if st.button("Find the Path"):
            similar_pdf_names = find_similar_documents(user_input, csv_file_path)

            if similar_pdf_names:
                st.header("Generated Path:")
                for name in similar_pdf_names:
                    result = replace_last_part(name, generated_text)
                    st.text(f"Document: {result}")

                    if document_type == "Plans":
                        selected_path = st.selectbox("Select Path:", [result, f"Plans//${generated_text}"])
                    else:
                        selected_path = st.selectbox("Select Path:", [result, generated_text])

                    # Split the file path by backslash
                    folders = selected_path.split("\\")
                    for i, folder in enumerate(folders):
                        arrow = "➔" * i
                        folder_icon = '<i class="fa fa-folder"></i>' if i < len(folders) - 1 else ''
                        st.write(arrow, folder_icon, folder, unsafe_allow_html=True)

                if st.button("➡️ Save File", key="save_file_button"):
                    # Save the file and show a toast message
                    with st.spinner("Saving..."):
                        # Perform your file-saving logic here
                        # For demonstration purposes, we'll create a dummy text file
                        file_content = "\n".join(folders)
                        file_path = "saved_file.txt"
                        with open(file_path, "w") as file:
                            file.write(file_content)

                    # Create a download button
                    st.download_button(
                        label="Download File",
                        data=file_content,
                        key="download_button",
                        file_name="saved_file.txt",
                        mime="text/plain",
                    )
                    st.success("File saved successfully!")

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
