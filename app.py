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

def retrieve_relevant_docs(question, retriever, k=5):
    # 단순히 k개의 문서를 반환합니다.
    docs = retriever.get_relevant_documents(question, k=k)
    return docs if docs else []

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

    file_titles = load_file_titles('./file_titles.csv')
    pdf_files = find_all_pdfs(app.config['UPLOAD_FOLDER'])

    if not pdf_files:
        return jsonify({'error': 'No files uploaded'}), 400

    combined_chunks_vector_store = None
    references = []

    for pdf_file in tqdm(pdf_files, desc="Processing PDFs"):
        vectorstore = encode_pdf(pdf_file, chunk_size=1000, chunk_overlap=200)

        if combined_chunks_vector_store is None:
            combined_chunks_vector_store = vectorstore
        else:
            combined_chunks_vector_store.merge_from(vectorstore)

    # 유사도 없이 k개의 문서 검색
    context_docs = retrieve_relevant_docs(
        question, 
        combined_chunks_vector_store.as_retriever(), 
        k=5
    )

    for doc in context_docs:
        if hasattr(doc, 'metadata') and 'source' in doc.metadata:
            file_path = doc.metadata['source']
            normalized_path = os.path.normpath(file_path).replace("\\", "/")
            file_title = file_titles.get(normalized_path, "Unknown Title")
            references.append({"file_path": normalized_path, "file_title": file_title})

    references = [dict(t) for t in {tuple(d.items()) for d in references}]

    llm = ChatOpenAI(temperature=0.3, model_name="gpt-4", max_tokens=2000)
    question_answer_from_context_chain = create_question_answer_from_context_chain(llm)
    result = answer_question_from_context(
        question, 
        " ".join([doc.page_content for doc in context_docs]), 
        question_answer_from_context_chain
    )
    return jsonify({'answer': result['answer'], 'context': result['context'], 'references': references}), 200

if __name__ == '__main__':
    if not os.path.exists(UPLOAD_FOLDER):
        os.makedirs(UPLOAD_FOLDER)
    app.run(debug=True, host='0.0.0.0', port=5001)
