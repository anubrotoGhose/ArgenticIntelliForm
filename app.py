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

# Load environment variables
load_dotenv()

genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

# Supported OCR languages
OCR_LANGUAGES = {
    "English": "eng",
    "French": "fra",
    "Arabic": "ara",
    "Chinese": "chi_sim",
    "Hindi": "hin"
}

# Image preprocessing to improve OCR accuracy
def preprocess_image(image):
    gray = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2GRAY)
    _, thresh = cv2.threshold(gray, 128, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return Image.fromarray(thresh)

# Function to extract text from an image using OCR
def extract_text_from_image(image_file, lang):
    image = Image.open(image_file)
    processed_image = preprocess_image(image)
    return pytesseract.image_to_string(processed_image, lang=lang)

# Extract text from different file formats
def extract_text_from_pdf(pdf_file, lang):
    images = convert_from_bytes(pdf_file.read())
    text = "".join([pytesseract.image_to_string(preprocess_image(img), lang=lang) for img in images])
    return text

def extract_text_from_docx(docx_file):
    doc = docx.Document(docx_file)
    return "\n".join([para.text for para in doc.paragraphs])

def extract_text_from_txt(txt_file):
    return txt_file.read().decode("utf-8")

def extract_json_from_text(text):
    match = re.search(r"\{.*\}", text, re.DOTALL)
    return match.group(0) if match else None

def calculate_age(dob):
    try:
        dob_date = datetime.datetime.strptime(dob, "%d-%b-%Y")
        today = datetime.datetime.today()
        return today.year - dob_date.year - ((today.month, today.day) < (dob_date.month, dob_date.day))
    except ValueError:
        return None

def parse_text_with_gemini(text):
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

def interactive_mapping(json_data):
    st.subheader("🛠️ Interactive Field Mapping")
    mappings = {}
    for key in json_data.keys():
        mappings[key] = st.text_input(f"Map '{key}' to application field", key)
    return mappings

st.title("📄 AI-Powered Multi-Language Form Processor")
st.subheader("Upload a PDF, Word document, Image, or Text file")

uploaded_file = st.file_uploader("Choose a file", type=["pdf", "docx", "png", "jpg", "jpeg", "txt"])
manual_text = st.text_area("Or enter text manually", "")
st.sidebar.title("⚙️ Admin Configuration")
selected_language = st.sidebar.selectbox("Select OCR Language", list(OCR_LANGUAGES.keys()))
enable_auto_mapping = st.sidebar.checkbox("Enable Auto-Mapping", value=True)

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
    
    structured_data = parse_text_with_gemini(extracted_text)
    st.subheader("📊 Extracted Data")
    st.json(structured_data)
    
    if not enable_auto_mapping:
        field_mappings = interactive_mapping(structured_data)
        st.write("🔄 Mapped Fields:", field_mappings)
    
    json_data = json.dumps(structured_data, indent=4)
    b64 = base64.b64encode(json_data.encode()).decode()
    href = f'<a href="data:file/json;base64,{b64}" download="extracted_data.json">📥 Download JSON</a>'
    st.markdown(href, unsafe_allow_html=True)