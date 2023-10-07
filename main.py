import streamlit as st 
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import PyPDFLoader, DirectoryLoader
from langchain.chains.summarize import load_summarize_chain
from transformers import T5Tokenizer, T5ForConditionalGeneration
from transformers import pipeline
import torch
import base64
import dropbox
from fpdf import FPDF  # Import FPDF library for PDF generation
import datetime
import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.base import MIMEBase
from email import encoders


# Initialize Dropbox client
dbx = dropbox.Dropbox('*****')

# Define the path to your offload folder
offload_folder = "C:\dropboxpro\offload_folder"

checkpoint = "LaMini-Flan-T5-248M"
tokenizer = T5Tokenizer.from_pretrained("MBZUAI/LaMini-Flan-T5-248M", legacy=False)
base_model = T5ForConditionalGeneration.from_pretrained(checkpoint, device_map='auto', torch_dtype=torch.float32)


# File loader and preprocessing
def file_preprocessing(file):
    loader = PyPDFLoader(file)
    pages = loader.load_and_split()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=200, chunk_overlap=50)
    texts = text_splitter.split_documents(pages)
    final_texts = ""
    for text in texts:
        final_texts = final_texts + text.page_content
    return final_texts

# LLM pipeline for summarization
def llm_pipeline(filepath):
    pipe_sum = pipeline(
        'summarization',
        model=base_model,
        tokenizer=tokenizer,
        max_length=500,
        min_length=50)
    input_text = file_preprocessing(filepath)
    result = pipe_sum(input_text)
    result = result[0]['summary_text']
    return result

# Function to display the PDF of a given file
def displayPDF(file):
    with open(file, "rb") as f:
        base64_pdf = base64.b64encode(f.read()).decode('utf-8')
    pdf_display = f'<iframe src="data:application/pdf;base64,{base64_pdf}" width="100%" height="600" type="application/pdf"></iframe>'
    st.markdown(pdf_display, unsafe_allow_html=True)

# Streamlit code
st.set_page_config(layout="wide")

def main():
    st.title("DocuMed")

    uploaded_file = st.file_uploader("Upload your PDF file", type=['pdf'])

    if uploaded_file is not None:
        if st.button("Analyse and Send request"):
            col1, col2 = st.columns(2)
            filepath = "data/" + uploaded_file.name
            with open(filepath, "wb") as temp_file:
                temp_file.write(uploaded_file.read())
            with col1:
                st.info("Uploaded File")
                pdf_view = displayPDF(filepath)

            with col2:
                summary = llm_pipeline(filepath)
                st.info("analysis Complete")
                st.success(summary)

                # Retrieve the existing contract document from Dropbox
                existing_contract_path = "C:\dropboxpro\Request_for_medical_equipments.pdf"
                existing_contract, _ = dbx.files_download(existing_contract_path)

                # Convert the existing contract to a PDF object
                existing_pdf = PyPDF2.PdfFileReader(io.BytesIO(existing_contract.content))

                # Create a new PDF to add the summary as a page
                new_pdf = PyPDF2.PdfFileWriter()

                # Add each page from the existing contract to the new PDF
                for page_num in range(existing_pdf.getNumPages()):
                    page = existing_pdf.getPage(page_num)
                    new_pdf.addPage(page)

                # Create a new page with the summary text
                new_page = PyPDF2.PageObject.createTextPage(summary)
                new_pdf.addPage(new_page)

                # Save the modified document locally
                modified_contract_path = "modified_contract.pdf"
                with open(modified_contract_path, "wb") as f:
                    new_pdf.write(f)

                # Send the modified contract as an email attachment
                send_email(modified_contract_path)

                # Display a success message
                st.success("Summary added to the contract and sent via email!")

def send_email(attachment_path):
    # Define email parameters
    sender_email = "******"
    sender_password = "*****"
    recipient_emails = ["****","*****"]
    subject = "Modified Contract with Summary"
    body = "Please find the attached contract document with the requirements summary."

    # Create the email message
    msg = MIMEMultipart()
    msg["From"] = sender_email
    msg["To"] = ", ".join(recipient_emails)
    msg["Subject"] = subject
    msg.attach(MIMEText(body, "plain"))

    # Attach the modified contract as a file
    with open(attachment_path, "rb") as attachment:
        part = MIMEBase("application", "octet-stream")
        part.set_payload(attachment.read())
        encoders.encode_base64(part)
        part.add_header("Content-Disposition", f"attachment; filename=modified_contract.pdf")
        msg.attach(part)

    # Send the email
    try:
        server = smtplib.SMTP("smtp.gmail.com", 587)
        server.starttls()
        server.login(sender_email, sender_password)
        text = msg.as_string()
        server.sendmail(sender_email, recipient_emails, text)
        server.quit()
        print("Email sent successfully!")
    except Exception as e:
        print(f"Error sending email: {str(e)}")

if __name__ == "__main__":
    main()