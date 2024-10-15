import os
import requests
from bs4 import BeautifulSoup
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm  # Progress bar
import time  # For request delay
import csv  # Module for saving CSV files

# Define base URL and chapters
base_url = "https://www.nrc.gov/reading-rm/doc-collections/nuregs/staff/sr0800/ch"
chapters = {f"Chapter{num}": f"{base_url}{num}/index.html" for num in range(5, 6)}

# Base URL for downloads
base_download_url = "https://www.nrc.gov"
failed_downloads = []  # To track failed downloads

# Dictionary to map file paths and titles
file_titles = {}

# Function to download the PDF
def download_pdf(pdf_name, link, root_dir):
    pdf_path = os.path.join(root_dir, pdf_name)

    # Check if file already exists
    if os.path.exists(pdf_path):
        print(f"Skipping {pdf_name}, already exists.")
        return

    try:
        response = requests.get(link)
        response.raise_for_status()  # Check for HTTP errors
        with open(pdf_path, 'wb') as pdf_file:
            pdf_file.write(response.content)
        print(f"Downloaded {pdf_name} to {root_dir}")
    except requests.exceptions.RequestException as e:
        print(f"Failed to download {pdf_name}: {e}")
        failed_downloads.append(pdf_name)  # Track the failed download (file name)

# Function to process each chapter
def process_chapter(chapter_name, chapter_url, progress_bar):
    try:
        # Create a folder for the chapter if it doesn't exist
        root_dir = f"./database/manual/NUREG0800_{chapter_name}"
        if not os.path.exists(root_dir):
            os.makedirs(root_dir)
        
        # Parse the webpage
        response = requests.get(chapter_url)
        response.raise_for_status()  # Check for HTTP errors
        soup = BeautifulSoup(response.content, 'html.parser')

        # Find the section table
        table_rows = soup.find_all('tr')

        # Iterate through the rows to process each section
        for row in table_rows:
            cells = row.find_all('td')
            if len(cells) > 2:  # Ensure there are at least 3 cells (for index 2 to be valid)
                section = cells[0].text.strip().replace('.', '_').replace(' ', '_').replace('-', '_')
                title = cells[1].text.strip()  # Extract the file title
                rev_link = cells[2].find('a')
                if rev_link:
                    pdf_url = base_download_url + rev_link['href']
                    pdf_name = section + ".pdf"
                    download_pdf(pdf_name, pdf_url, root_dir)
                    progress_bar.update(1)  # Update the progress bar

                    # Save the path and title in the dictionary
                    file_path = os.path.join(root_dir, pdf_name)

                    # Normalize the path and replace backslashes with slashes
                    normalized_path = os.path.normpath(file_path).replace("\\", "/")
                    file_titles[normalized_path] = title

                    # Add a delay between requests to avoid overloading the server
                    time.sleep(1)
    except requests.exceptions.RequestException as e:
        print(f"Failed to process {chapter_name}: {e}")

# Function to save file titles to CSV
def save_titles_to_csv():
    csv_file_path = "file_titles.csv"
    with open(csv_file_path, mode="w", newline="") as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(["File Path", "File Title"])  # Write the CSV header
        for file_path, title in file_titles.items():
            writer.writerow([file_path, title])  # Save each file path and title to the CSV
    print(f"File titles saved to {csv_file_path}")

# Process chapters in parallel with a progress bar
def process_all_chapters():
    total_chapters = len(chapters)  # Total number of chapters to process
    with ThreadPoolExecutor() as executor:
        with tqdm(total=total_chapters, desc="Processing Chapters", unit="chapter") as progress_bar:
            for chapter_name, chapter_url in chapters.items():
                print(f"Processing {chapter_name}...")
                executor.submit(process_chapter, chapter_name, chapter_url, progress_bar)

    # Write failed downloads (file names) to a text file
    if failed_downloads:
        with open("failed_downloads.txt", "w") as f:
            for item in failed_downloads:
                f.write(f"{item}\n")
        print("Failed downloads recorded in failed_downloads.txt")

    # Save the file path to title dictionary to a CSV file
    save_titles_to_csv()

process_all_chapters()
