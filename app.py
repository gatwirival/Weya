import cv2
import numpy as np
import matplotlib.pyplot as plt
import streamlit as st
from google.generativeai import GenerativeModel, configure
from google_api_key import google_api_key
import logging
from fpdf import FPDF
from googletrans import Translator

# Configure logging
logging.basicConfig(filename='app.log', level=logging.ERROR)

# Configure the Google Generative AI
configure(api_key=google_api_key)

generation_config = {
    "temperature": 1,
    "top_p": 0.95,
    "top_k": 0,
    "max_output_tokens": 8192,
}

safety_settings = [
    {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
    {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
    {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
    {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
]

system_prompt = """
You are a domain expert in medical image analysis. You are tasked with
examining medical images for a renowned hospital.
Your expertise will help in identifying or
discovering any anomalies, diseases, conditions or
any health issues that might be present in the image.

Your key responsibilities:
1. Detailed Analysis : Scrutinize and thoroughly examine each image,
focusing on finding any abnormalities.
2. Analysis Report : Document all the findings and
clearly articulate them in a structured format.
3. Recommendations : Basis the analysis, suggest remedies,
tests or treatments as applicable.
4. Treatments : If applicable, lay out detailed treatments
which can help in faster recovery.

Important Notes to remember:
1. Scope of response : Only respond if the image pertains to
human health issues.
2. Clarity of image : In case the image is unclear,
note that certain aspects are
'Unable to be correctly determined based on the uploaded image'
3. Disclaimer : Accompany your analysis with the disclaimer:
"Consult with a Doctor before making any decisions."
4. Your insights are invaluable in guiding clinical decisions.
Please proceed with the analysis, adhering to the
structured approach outlined above.

Please provide the final response with these 4 headings :
Detailed Analysis, Analysis Report, Recommendations and Treatments
"""

model = GenerativeModel(
    model_name="gemini-1.5-pro-latest",
    generation_config=generation_config,
    safety_settings=safety_settings
)

st.set_page_config(page_title="Visual Medical Assistant", page_icon="🩺", layout="wide")
st.title("Visual Medical Assistant 👨‍⚕️ 🩺 🏥")
st.subheader("An app to help with medical analysis using images")

translator = Translator()
languages = {
    "English": "en",
    "Spanish": "es",
    "French": "fr",
    "German": "de",
    "Chinese": "zh-cn"
}
selected_language = st.selectbox("Choose the language for analysis", list(languages.keys()))

file_uploaded = st.file_uploader('Upload the image for Analysis', type=['png', 'jpg', 'jpeg'])

def highlight_anomalies(image, anomalies):
    highlighted_image = image.copy()
    for anomaly in anomalies:
        x, y, w, h = anomaly
        cv2.rectangle(highlighted_image, (x, y), (x + w, y + h), (255, 0, 0), 2)
    return highlighted_image

def save_report_as_text(report):
    with open("report.txt", "w") as file:
        file.write(report)

def save_report_as_pdf(report):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", size=12)
    for line in report.split('\n'):
        pdf.cell(200, 10, txt=line, ln=True, align='L')
    pdf.output("report.pdf")

if file_uploaded:
    st.image(file_uploaded, width=400, caption='Uploaded Image')

submit = st.button("Generate Analysis")

if submit and file_uploaded:
    with st.spinner("Analyzing the image..."):
        try:
            # Check file type
            if file_uploaded.type not in ['image/png', 'image/jpeg']:
                st.error("Unsupported file type. Please upload a PNG or JPEG image.")
            else:
                image_data = file_uploaded.getvalue()
                image_array = np.frombuffer(image_data, np.uint8)
                image = cv2.imdecode(image_array, cv2.IMREAD_COLOR)

                image_part = {
                    "mime_type": file_uploaded.type,
                    "data": image_data
                }

                prompt_parts = [image_part, system_prompt]

                # Generate content
                response = model.generate_content(prompt_parts)

                if response:
                    report = response.text

                    # Translate report if needed
                    if languages[selected_language] != "en":
                        translated = translator.translate(report, dest=languages[selected_language])
                        report = translated.text

                    st.title('Detailed analysis based on the uploaded image')
                    st.write(report)

                    # Dummy anomalies for demonstration
                    anomalies = [(50, 50, 100, 100), (150, 150, 80, 80)]
                    highlighted_image = highlight_anomalies(image, anomalies)

                    # Display highlighted image
                    st.image(highlighted_image, caption='Highlighted Anomalies', use_column_width=True)

                    save_report_as_text(report)
                    save_report_as_pdf(report)

                    st.download_button(
                        label="Download Report as Text",
                        data=report,
                        file_name='report.txt',
                        mime='text/plain'
                    )

                    with open("report.pdf", "rb") as pdf_file:
                        pdf_bytes = pdf_file.read()
                        st.download_button(
                            label="Download Report as PDF",
                            data=pdf_bytes,
                            file_name='report.pdf',
                            mime='application/octet-stream'
                        )
                else:
                    st.error("Failed to generate a response from the AI model.")
        except Exception as e:
            logging.error(f"Error occurred: {e}")
            st.error("An error occurred while processing the image. Please try again.")


st.markdown("""
    **Disclaimer:** This tool provides AI-generated analysis based on the uploaded image.
    Always consult with a healthcare professional before making any medical decisions.
""")
