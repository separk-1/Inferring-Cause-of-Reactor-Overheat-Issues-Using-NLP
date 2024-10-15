import os
import pypdf

# Directory path to scan for PDF files
DATABASE_FOLDER = './database'

# Function to check if a PDF is valid
def is_valid_pdf(file_path):
    try:
        with open(file_path, 'rb') as f:
            reader = pypdf.PdfReader(f)
            return True  # The file is a valid PDF
    except Exception as e:
        return False  # The file is invalid or corrupted

# Function to find all invalid PDF files in the specified folder
def find_invalid_pdfs(root_folder):
    invalid_files = []
    for dirpath, _, filenames in os.walk(root_folder):
        for file in filenames:
            if file.endswith('.pdf'):  # Only check files with .pdf extension
                full_path = os.path.join(dirpath, file)
                if not is_valid_pdf(full_path):
                    invalid_files.append(full_path)  # Add invalid files to the list
    return invalid_files

# Execute the PDF check
invalid_files = find_invalid_pdfs(DATABASE_FOLDER)
if invalid_files:
    print("The following files are either corrupted or not valid PDF files:")
    for file in invalid_files:
        print(file)
else:
    print("All PDF files are valid.")
