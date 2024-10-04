
# 🔍 RAG-Based Search Engine

This project is a **RAG (Retrieval-Augmented Generation)**-powered search engine designed to provide precise answers and references from regulatory documents related to nuclear safety. The system crawls safety datasets, processes them, and builds a search interface where users can input questions and receive relevant responses with cited references.

---

## 📦 Project Structure

### 0. **Requirements**
Ensure that the required packages and dependencies are installed by referencing the `requirements.txt` file.

### 1. **Dataset Collection** 🗂️

We collect nuclear safety regulation datasets from the NRC website.

- **`get_dataset.py`**:  
   This script crawls the NRC website for regulatory documents and stores them in the `./database/manual` directory. Any failed downloads are logged in `failed_downloads.txt`. Additionally, it saves the file path and corresponding file title in `file_title.csv`.
  
- **`check_dataset.py`**:  
   This script checks the `./database` folder for any corrupted or missing files and ensures that the dataset is intact.

### 2. **Run Search Engine** 🔍

- **`app.py`**:  
   This script runs the web application. When launched, it starts a local web server where users can input queries. By pressing the Enter key, the system provides an answer along with references to relevant documents.

### 3. **Functions** ⚙️

- **`functions/functions_rag.py`**:  
   This file stores all functions related to **RAG (Retrieval-Augmented Generation)** processing, including document chunking, encoding, and retrieval.

- **`functions/functions_utils.py`**:  
   This file stores utility functions such as file loading, dataset handling, and metadata processing.

### 4. **Frontend Design** 🎨

- **`static/`** and **`templates/`**:  
   These folders contain the HTML, CSS, and other static assets for the web application design. The user interface is clean, minimalistic, and responsive.

---

## 🛠️ Setup Guide

### 1. Clone the Repository
```bash
git clone https://github.com/separk-1/Inferring-Cause-of-Reactor-Overheat-Issues-Using-NLP.git
cd Inferring-Cause-of-Reactor-Overheat-Issues-Using-NLP
```

### 2. Install the Required Packages
Make sure you have Python 3.10 and the required libraries installed. Install dependencies using:

```bash
conda env create -f requirements.txt
```

### 3. Download Spacy Language Model
To ensure proper NLP operations, download the Spacy language model:

```bash
python -m spacy download en_core_web_sm
```

### 4. Run Dataset Collection
To crawl and store the dataset from the NRC website:

```bash
python get_dataset.py
```

### 5. Check the Dataset
After downloading, run the script to verify the integrity of the dataset:

```bash
python check_dataset.py
```

### 6. Configure Your OpenAI API Key
Make sure to store your OpenAI API key in the `.env` file.

Create or update the `.env` file in the root directory with the following content:

```bash
OPENAI_API_KEY='your_openai_api_key'
```

### 7. Run the Search Engine
Launch the search engine locally:

```bash
python app.py
```

Open a browser and navigate to `http://127.0.0.1:5001` to access the web interface, ask questions, and view the generated answers with references.

---

## 🗂️ Example File Structure
```
RAG-Search-Engine/
│
├── database/
│   └── manual/
│       ├── NUREG0800_Chapter5/  # Crawled dataset
│       └── ...
├── static/                      # Frontend assets (CSS, etc.)
├── templates/                   # HTML files for the web app
├── functions/
│   ├── functions_rag.py         # RAG-related functions
│   └── functions_utils.py       # Utility functions
├── get_dataset.py               # Script to crawl and download the dataset
├── check_dataset.py             # Script to verify the dataset integrity
├── app.py                       # Main web application script
└── requirements.txt             # List of required dependencies
```

---

## 🎯 Key Components

### 🔑 Dataset Collection & Preprocessing
- **Web Crawling**: The system scrapes regulatory documents from the NRC website and stores them in the local database.
- **Data Integrity Check**: It ensures the downloaded documents are intact.

### 🔑 RAG Search Engine
- **Document Chunking**: Documents are split into manageable chunks for efficient retrieval.
- **Question-Answering**: A combination of retrieval (via FAISS) and generative models provides answers to user queries, ensuring both relevance and accuracy.

---

## 🚀 How It Works

- **Document Encoding**: The documents are split into chunks, and embeddings are created using OpenAI embeddings, which are then stored in the FAISS vector store.
- **User Query**: The user enters a query through the web interface.
- **RAG Processing**: The system retrieves relevant documents based on the query and generates an answer.
- **References**: The response is accompanied by references to the original documents used to generate the answer.

---
## 🎥 Example Usage

![Example GIF](./assets/example_usage.gif)

---

## 📋 Future Enhancements
- **Expand Dataset**: Integrate more comprehensive datasets beyond NRC regulations.
- **Refine Search Algorithm**: Improve retrieval and generation accuracy by experimenting with more advanced vector models.
- **User Feedback**: Implement feedback loops for users to rate the relevance of the results.

---

This project demonstrates how RAG can be used to build powerful question-answering systems based on unstructured text data. By leveraging NRC regulatory documents, this system provides accurate and detailed answers, making it a valuable tool for understanding nuclear safety regulations.
