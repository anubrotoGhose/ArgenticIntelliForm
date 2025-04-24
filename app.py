import streamlit as st
import pytesseract
from pdf2image import convert_from_bytes
from PIL import Image
import google.generativeai as genai
import docx
import json
import os
import re
import datetime
import base64
import cv2
import numpy as np
from dotenv import load_dotenv
from textblob import TextBlob
import spacy
from dateutil import parser

# Default to None
GEMINI_API_KEY = None

# Use secrets in deployment
# Load .env file (LOCAL DEVELOPMENT)
if os.path.exists(".env"):
    load_dotenv()  # Load environment variables
    GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

# Use Streamlit Secrets (DEPLOYMENT)
elif "GEMINI_API_KEY" in st.secrets:
    GEMINI_API_KEY = st.secrets["GEMINI_API_KEY"]
    
# Ensure API key is set before configuring genai
if GEMINI_API_KEY:
    genai.configure(api_key=GEMINI_API_KEY)
else:
    st.error("❌ Missing API Key! Please set GEMINI_API_KEY in .env or Streamlit Secrets.")

# Load NLP model
nlp = spacy.load("en_core_web_sm")

# Define OCR languages
OCR_LANGUAGES = {
    "English": "eng",
    "Spanish": "spa",
    "French": "fra",
    "German": "deu"
}

import streamlit as st

# Sidebar options
st.sidebar.header("Extraction Features")
auto_mapping = st.sidebar.checkbox("Enable Auto-Mapping", value=False)
select_all = st.sidebar.checkbox("Select All", value=False)

# Features selection
default_features = {
    "Named Entities (NER)": False,
    "Date of Birth & Age": False,
    "Address Extraction": False,
    "Invoice Data Extraction": False,
    "Currency Detection": False,
    "Product Info": False,
    "Sentiment Analysis": False,
}

# If 'Select All' is checked, enable all features
if select_all:
    features = {key: True for key in default_features}
else:
    features = {key: st.sidebar.checkbox(key, value=False) for key in default_features}

# Language selection for OCR
selected_language = st.sidebar.selectbox(
    "Select OCR Language",
    options=list(OCR_LANGUAGES.keys()),
    index=0  # Default to the first language (English)
)

# Image Preprocessing
def preprocess_image(image):
    gray = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2GRAY)
    _, thresh = cv2.threshold(gray, 128, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return Image.fromarray(thresh)

# OCR Extraction
def extract_text_from_image(image_file, lang_code):
    image = Image.open(image_file)
    processed_image = preprocess_image(image)
    return pytesseract.image_to_string(processed_image, lang=lang_code)

def extract_text_from_pdf(pdf_file, lang_code):
    images = convert_from_bytes(pdf_file.read())
    text = ""
    for image in images:
        text += extract_text_from_image(image, lang_code) + "\n"
    return text

def extract_text_from_docx(docx_file):
    doc = docx.Document(docx_file)
    return "\n".join([paragraph.text for paragraph in doc.paragraphs])

def extract_text_from_txt(txt_file):
    return txt_file.read().decode('utf-8')

# NLP Processing Function
def extract_named_entities(text):
    model = genai.GenerativeModel("gemini-1.5-flash")
    prompt = f"Extract named entities from the following text and return them in JSON format:\n{text}"
    response = model.generate_content(prompt)

    try:
        # Convert the response text into a valid JSON format
        extracted_json = json.loads(response.text)
        formatted_json = json.dumps(extracted_json, indent=4)  # Pretty-print JSON
        return formatted_json
    except json.JSONDecodeError:
        return response.text


def extract_invoice_data(text):
    model = genai.GenerativeModel("gemini-1.5-flash")
    prompt = f"Extract invoice details including invoice number, total amount, and currency from this text:\n{text}"
    response = model.generate_content(prompt)
    return response.text

def extract_currency(text):
    """
    Uses Gemini 1.5 Flash to extract currency values from the given text.
    
    Returns:
        List of extracted currency values.
    """
    model = genai.GenerativeModel("gemini-1.5-flash")
    prompt = f"""
    Extract all currency values from the following text. Return only a list of extracted values.
    Text: {text}
    """
    
    response = model.generate_content(prompt)
    
    # Convert response to a structured format
    extracted_currencies = response.text.strip().split("\n")
    
    return extracted_currencies

def extract_product_info(text):
    model = genai.GenerativeModel("gemini-1.5-flash")

    prompt = f"""
    Extract product details from the text below. Return the output in structured JSON format 
    with fields: "Product Name", "SKU", "Price", "Quantity", "Category", and "Brand". If 
    any field is missing, return it as null.

    Text:
    {text}

    Output format:
    {{
        "Product Name": "Example Product",
        "SKU": "ABC123",
        "Price": "$19.99",
        "Quantity": "2",
        "Category": "Electronics",
        "Brand": "BrandName"
    }}
    """

    try:
        response = model.generate_content(prompt)
        return json.loads(response.text)  # Parse JSON response
    except Exception as e:
        return {"error": str(e)}


def extract_dob_and_age(text):
    """
    Uses Gemini 1.5 Flash to extract the Date of Birth and calculate the Age.
    
    Returns:
        Dictionary with "Date of Birth" and "Age".
    """
    model = genai.GenerativeModel("gemini-1.5-flash")
    
    prompt = f"""
    Extract the Date of Birth (DOB) from the given text and calculate the person's age.
    Return the result as a JSON object with keys "Date of Birth" and "Age".
    
    Text: {text}
    """
    
    response = model.generate_content(prompt)
    
    # Convert AI response to a structured format
    try:
        extracted_data = eval(response.text)  # Convert response into a dictionary
        return extracted_data
    except:
        return {"Date of Birth": None, "Age": None}

def extract_address(text):
    model = genai.GenerativeModel("gemini-1.5-flash")
    
    prompt = f"""
    Extract all addresses from the given text.
    Return the addresses as a JSON object with key "Address".
    
    Text: {text}
    """
    
    response = model.generate_content(prompt)
    
    try:
        extracted_data = eval(response.text)  # Convert AI response into a dictionary
        if isinstance(extracted_data, list):  # Handle list case
            return {"Address": extracted_data}
        return extracted_data
    except:
        return {"Address": []}  # Ensure dictionary format


def perform_sentiment_analysis(text):
    model = genai.GenerativeModel("gemini-1.5-flash")
    prompt = f"Analyze the sentiment of the following text and return Positive, Negative, or Neutral:\n{text}"
    response = model.generate_content(prompt)
    return response.text

def extract_json_from_text(text):
    match = re.search(r"\{.*\}", text, re.DOTALL)
    return match.group(0) if match else None


def auto_map_text(text):
    prompt = f"""
    Extract key-value pairs from the following text and return a valid JSON object.
    If a 'Date of Birth' is found, calculate the person's age and include it in the JSON.
    
    Text:
    {text}
    """
    model = genai.GenerativeModel('gemini-1.5-flash')
    response = model.generate_content(prompt)
    json_text = extract_json_from_text(response.text)
    if json_text:
        try:
            data = json.loads(json_text.strip())
            if 'Date of Birth' in data:
                age = calculate_age(data['Date of Birth'])
                if age is not None:
                    data['Age'] = age
            return data
        except json.JSONDecodeError:
            return {"error": "Failed to parse JSON", "response": json_text}
    return {"error": "No JSON found", "response": response.text}


def process_text(text):
    extracted_data = {}

    feature_mapping = {
        "Named Entities (NER)": lambda: {"Named Entities": extract_named_entities(text)},
        "Date of Birth & Age": lambda: extract_dob_and_age(text),
        "Address Extraction": lambda: extract_address(text),
        "Invoice Data Extraction": lambda: extract_invoice_data(text),
        "Currency Detection": lambda: {"Currencies": extract_currency(text)},
        "Product Info": lambda: extract_product_info(text),
        "Sentiment Analysis": lambda: {"Sentiment": perform_sentiment_analysis(text)},
    }

    if auto_mapping:
        # Define auto-mapping logic (e.g., dynamically detect relevant features)
        extracted_data["Auto-Mapped Data"] = auto_map_text(text)
    else:
        # Process only selected features
        for feature, func in feature_mapping.items():
            if features.get(feature, False):
                extracted_data.update(func())

    return extracted_data


# Streamlit UI Components
st.title("📄 AI-Powered NLP Form Processor")
st.subheader("Upload a document or enter text manually")

uploaded_file = st.file_uploader("Choose a file", type=["pdf", "docx", "png", "jpg", "jpeg", "txt"])
manual_text = st.text_area("Or enter text manually", "")

if st.button("Process Text"):
    
    if manual_text.strip():
        extracted_text = manual_text
    
    elif uploaded_file is not None:
        file_type = uploaded_file.type
        lang_code = OCR_LANGUAGES[selected_language]
        
        if "pdf" in file_type:
            extracted_text = extract_text_from_pdf(uploaded_file, lang_code)
        
        elif "word" in file_type or "docx" in uploaded_file.name:
            extracted_text = extract_text_from_docx(uploaded_file)
        
        elif "image" in file_type or uploaded_file.name.split('.')[-1] in ["png", "jpg", "jpeg"]:
            extracted_text = extract_text_from_image(uploaded_file, lang_code)
        
        elif "text/plain" in file_type or uploaded_file.name.endswith(".txt"):
            extracted_text = extract_text_from_txt(uploaded_file)
        
        else:
            st.error("❌ Unsupported file format")
            st.stop()
    
    else:
        st.error("❌ Please enter text or upload a document.")
        st.stop()
    
    st.subheader("📜 Extracted Text")
    st.text_area("Extracted Content", extracted_text, height=300)
    
    structured_data = process_text(extracted_text)
    
    st.subheader("📊 Extracted Data")
    st.json(structured_data)
    
    json_data = json.dumps(structured_data, indent=4)
    
    b64 = base64.b64encode(json_data.encode()).decode()
    
    href = f'<a href="data:file/json;base64,{b64}" download="extracted_data.json">📥 Download JSON</a>'
    
    st.markdown(href, unsafe_allow_html=True)