import os
from flask import Flask, request, render_template, jsonify
from flask_cors import CORS
from dotenv import load_dotenv
from langchain_community.vectorstores import FAISS
from langchain_community.chat_models import ChatOpenAI
import spacy
from tqdm import tqdm
import logging
from functions.functions_rag import (
    encode_pdf,
    retrieve_context_per_question,
    answer_question_from_context,
    create_question_answer_from_context_chain
)
from functions.functions_utils import (
    find_all_pdfs,
    load_file_titles
)
from functions.functions_casual_relation import analyze_text

app = Flask(__name__)
CORS(app)

UPLOAD_FOLDER = './database'
ALLOWED_EXTENSIONS = {'pdf'}
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Load environment variables
load_dotenv()

# Load OpenAI API key from environment
openai_api_key = os.getenv("OPENAI_API_KEY")
os.environ["OPENAI_API_KEY"] = openai_api_key

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

nlp = spacy.load("en_core_web_sm")

def retrieve_relevant_docs(question, retriever, k=5, threshold=0.5):
    docs_with_scores = retriever.get_relevant_documents_with_scores(question, k=k)
    # 유사도 점수가 threshold 이상인 문서만 선택
    filtered_docs = [doc for doc, score in docs_with_scores if score >= threshold]
    return filtered_docs if filtered_docs else [doc for doc, _ in docs_with_scores[:1]]  # 최소 1개 반환

def analyze_database_and_create_graph():
    pdf_files = find_all_pdfs(app.config['UPLOAD_FOLDER'])
    all_text = ""
    for pdf_file in tqdm(pdf_files, desc="Processing PDFs"):
        vectorstore = encode_pdf(pdf_file, chunk_size=1000, chunk_overlap=200)
        text = " ".join([doc.page_content for doc in vectorstore.docstore._dict.values()])
        all_text += text + " "
    
    logger.info("Analyzing text and creating graph...")
    result = analyze_text(all_text)
    if result['visualization_path']:
        logger.info(f"Graph created and saved as {result['visualization_path']}")
        app.config['GRAPH_PATH'] = result['visualization_path']
    else:
        logger.warning("Failed to create graph")
        app.config['GRAPH_PATH'] = None

# 애플리케이션 시작 시 그래프 생성
analyze_database_and_create_graph()

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/ask', methods=['POST'])
def ask_question():
    data = request.get_json()
    question = data.get('question')

    if not question:
        return jsonify({'error': 'Missing question'}), 400

    # Load file paths and titles from the CSV file
    file_titles = load_file_titles('./file_titles.csv')

    # Get all PDF files in the directory
    pdf_files = find_all_pdfs(app.config['UPLOAD_FOLDER'])

    if not pdf_files:
        return jsonify({'error': 'No files uploaded'}), 400

    combined_chunks_vector_store = None
    references = []

    # Progress bar to show file processing
    for pdf_file in tqdm(pdf_files, desc="Processing PDFs"):
        vectorstore = encode_pdf(pdf_file, chunk_size=1000, chunk_overlap=200)

        if combined_chunks_vector_store is None:
            combined_chunks_vector_store = vectorstore
        else:
            combined_chunks_vector_store.merge_from(vectorstore)

    # Retrieve the relevant context (returning the document object itself)
    context_docs = retrieve_relevant_docs(
    question, 
    combined_chunks_vector_store.as_retriever(), 
    k=5, 
    threshold=0
    )

    # Append source of related files to references by accessing metadata
    for doc in context_docs:
        if hasattr(doc, 'metadata') and 'source' in doc.metadata:
            file_path = doc.metadata['source']  # Use the full file path
            normalized_path = os.path.normpath(file_path)  # Normalize the path for matching
            normalized_path = normalized_path.replace("\\", "/")  # Convert backslashes to slashes (if necessary)
            
            # Find file title using the normalized path
            file_title = file_titles.get(normalized_path, "Unknown Title")
            references.append({"file_path": normalized_path, "file_title": file_title})

    # Remove duplicate file paths and titles
    references = [dict(t) for t in {tuple(d.items()) for d in references}]

    # Generate an answer using the context
    llm = ChatOpenAI(temperature=0.3, model_name="gpt-4", max_tokens=2000)
    question_answer_from_context_chain = create_question_answer_from_context_chain(llm)
    result = answer_question_from_context(question, " ".join([doc.page_content for doc in context_docs]), question_answer_from_context_chain)
    return jsonify({'answer': result['answer'], 'context': result['context'], 'references': references}), 200

if __name__ == '__main__':
    if not os.path.exists(UPLOAD_FOLDER):
        os.makedirs(UPLOAD_FOLDER)
    app.run(debug=True, host='0.0.0.0', port=5001)