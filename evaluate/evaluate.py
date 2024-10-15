import json
import os
from functions.functions_rag import (
    encode_pdf,
    retrieve_context_per_question,
    answer_question_from_context,
    create_question_answer_from_context_chain
)
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
import openai
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from rouge_score import rouge_scorer
from bert_score import score

# Load environment variables
load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")

def calculate_bert_score(expected_answer, model_answer):
    P, R, F1 = score([model_answer], [expected_answer], lang="en", rescale_with_baseline=True)
    return F1.item()

def load_pdf_files(upload_folder):
    pdf_files = []
    for root, _, files in os.walk(upload_folder):
        for file in files:
            if file.endswith('.pdf'):
                pdf_files.append(os.path.join(root, file))
    return pdf_files

def load_sample_qa(filepath):
    with open(filepath, 'r') as f:
        return json.load(f)

def calculate_bleu(expected_answer, model_answer):
    reference = [expected_answer.split()]
    candidate = model_answer.split()
    smoothing = SmoothingFunction().method4
    return sentence_bleu(reference, candidate, smoothing_function=smoothing)

def calculate_rouge(expected_answer, model_answer):
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rougeL'], use_stemmer=True)
    scores = scorer.score(expected_answer, model_answer)
    return scores

def retrieve_relevant_docs(question, retriever, k=5, threshold=0.4):
    docs_with_scores = retriever.invoke(question, k=k)
    filtered_docs = [doc for doc, score in docs_with_scores if score >= threshold]
    return filtered_docs if filtered_docs else [doc for doc, _ in docs_with_scores[:1]]

def evaluate_model(pdf_files, sample_qa):
    combined_chunks_vector_store = None
    for pdf_file in pdf_files:
        vectorstore = encode_pdf(pdf_file, chunk_size=1000, chunk_overlap=200)
        if combined_chunks_vector_store is None:
            combined_chunks_vector_store = vectorstore
        else:
            combined_chunks_vector_store.merge_from(vectorstore)
    
    llm = ChatOpenAI(temperature=0.3, model_name="gpt-4", max_tokens=2000)
    question_answer_from_context_chain = create_question_answer_from_context_chain(llm)

    bleu_scores = []
    rouge_scores = []
    bert_scores = []

    for qa_pair in sample_qa:
        question = qa_pair["question"]
        expected_answer = qa_pair["answer"]
        
        context_docs = retrieve_relevant_docs(
            question, 
            combined_chunks_vector_store.as_retriever(), 
            k=5, 
            threshold=0.4
        )

        context = " ".join([doc.page_content for doc in context_docs])
        result = answer_question_from_context(question, context, question_answer_from_context_chain)
        model_answer = result['answer']
        
        print(f"Question: {question}")
        print(f"Expected Answer: {expected_answer}")
        print(f"Model Answer: {model_answer}")
        
        bleu = calculate_bleu(expected_answer, model_answer)
        rouge = calculate_rouge(expected_answer, model_answer)
        bert = calculate_bert_score(expected_answer, model_answer)

        print(f"BLEU Score: {bleu:.2f}")
        print(f"ROUGE Scores: {rouge}")
        print(f"BERT Score: {bert:.2f}")
        print("\n" + "="*50 + "\n")
        
        bleu_scores.append(bleu)
        rouge_scores.append(rouge['rougeL'].fmeasure)
        bert_scores.append(bert)

    avg_bleu = sum(bleu_scores) / len(bleu_scores)
    avg_rouge = sum(rouge_scores) / len(rouge_scores)
    avg_bert = sum(bert_scores) / len(bert_scores)

    print(f"Average BLEU Score: {avg_bleu:.2f}")
    print(f"Average ROUGE-L Score: {avg_rouge:.2f}")
    print(f"Average BERT Score: {avg_bert:.2f}")

if __name__ == "__main__":
    upload_folder = '../database'  # 경로 수정
    pdf_files = load_pdf_files(upload_folder)
    
    sample_qa_path = "evaluate/simple_qa.json"  # 경로 수정
    sample_qa = load_sample_qa(sample_qa_path)
    
    if pdf_files:
        evaluate_model(pdf_files, sample_qa)
    else:
        print("No PDF files found in the specified folder.")
