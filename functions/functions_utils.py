# File validation
import pdfplumber
import csv
import os

ALLOWED_EXTENSIONS = {'pdf'}

# Check if the file has an allowed extension
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# Extract text from a PDF
def extract_text_from_pdf(pdf_path):
    with pdfplumber.open(pdf_path) as pdf:
        text = ""
        for page in pdf.pages:
            text += page.extract_text()
    return text

# Find all PDFs in the folder
def find_all_pdfs(root_folder):
    pdf_files = []
    for dirpath, _, filenames in os.walk(root_folder):
        for file in filenames:
            if allowed_file(file):
                full_path = os.path.join(dirpath, file)
                pdf_files.append(full_path)
    return pdf_files

# Load file titles from a CSV file
def load_file_titles(csv_file):
    file_titles = {}

    # Read the CSV file with UTF-8 encoding
    with open(csv_file, mode='r', newline='', encoding='utf-8') as file:
        reader = csv.reader(file)
        next(reader)  # Skip the header

        # Process each row
        for row in reader:
            if len(row) != 2:
                print(f"Skipping invalid row: {row}")
                continue

            # Strip leading/trailing whitespace from file path and title
            file_path = row[0].strip()  # Remove whitespace from the file path
            file_title = row[1].strip().strip('"')  # Remove quotation marks and whitespace

            # Add to the dictionary
            file_titles[file_path] = file_title

    return file_titles
