import spacy
import networkx as nx
import matplotlib.pyplot as plt
import os
import logging
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import LatentDirichletAllocation
import numpy as np

# 로깅 설정
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

# Load spaCy model
nlp = spacy.load("en_core_web_sm")

def extract_keywords_tfidf(text, max_features=20):
    vectorizer = TfidfVectorizer(max_features=max_features, stop_words='english')
    tfidf_matrix = vectorizer.fit_transform([text])
    feature_names = vectorizer.get_feature_names_out()
    tfidf_scores = tfidf_matrix.toarray()[0]
    keywords = [(word, score) for word, score in zip(feature_names, tfidf_scores) if score > 0]
    keywords.sort(key=lambda x: x[1], reverse=True)
    return keywords

def perform_topic_modeling(text, n_topics=3, n_top_words=5):
    vectorizer = TfidfVectorizer(max_features=1000, stop_words='english')
    tfidf_matrix = vectorizer.fit_transform([text])
    lda = LatentDirichletAllocation(n_components=n_topics, random_state=42)
    lda.fit(tfidf_matrix)
    feature_names = vectorizer.get_feature_names_out()
    topics = []
    for topic_idx, topic in enumerate(lda.components_):
        top_words = [feature_names[i] for i in topic.argsort()[:-n_top_words - 1:-1]]
        topics.append((f"Topic {topic_idx + 1}", top_words))
    return topics

def extract_causal_relations(text):
    doc = nlp(text)
    causal_relations = []
    for sent in doc.sents:
        for token in sent:
            if token.dep_ == "prep" and token.text.lower() in ["because", "due to", "as a result of"]:
                cause = []
                effect = []
                
                # Extract the cause (usually comes after the causal indicator)
                for child in token.children:
                    if child.dep_ in ["pobj", "attr"]:
                        cause.extend([t.text for t in child.subtree])
                
                # Extract the effect (usually the main clause)
                for ancestor in token.ancestors:
                    if ancestor.dep_ == "ROOT":
                        effect = [t.text for t in ancestor.subtree if t != token]
                        break
                
                if cause and effect:
                    causal_relations.append({
                        "cause": " ".join(cause),
                        "effect": " ".join(effect),
                        "indicator": token.text
                    })
    return causal_relations

def visualize_analysis(keywords, topics, causal_relations):
    G = nx.Graph()
    
    # Add keyword nodes
    for word, score in keywords[:10]:  # Top 10 keywords
        G.add_node(word, type='keyword', size=score*1000)
    
    # Add topic nodes
    for topic, words in topics:
        G.add_node(topic, type='topic', size=500)
        for word in words:
            G.add_edge(topic, word)
    
    # Add causal relation nodes and edges
    for relation in causal_relations:
        G.add_edge(relation['cause'], relation['effect'], type='causal', label=relation['indicator'])

    plt.figure(figsize=(12, 8))  # 이미지 크기를 줄임
    pos = nx.spring_layout(G, k=0.5, iterations=50)

    # Draw nodes
    nx.draw_networkx_nodes(G, pos, node_size=[G.nodes[node].get('size', 300) for node in G.nodes()], 
                           node_color=['lightblue' if G.nodes[node].get('type') == 'keyword' else 'lightgreen' for node in G.nodes()])
    
    # Draw edges
    nx.draw_networkx_edges(G, pos)
    
    # Draw labels
    nx.draw_networkx_labels(G, pos, font_size=8)
    
    # Draw edge labels for causal relations
    causal_edges = [(u, v) for (u, v, d) in G.edges(data=True) if d.get('type') == 'causal']
    causal_labels = {(u, v): d['label'] for (u, v, d) in G.edges(data=True) if d.get('type') == 'causal'}
    nx.draw_networkx_edge_labels(G, pos, edge_labels=causal_labels)

    plt.axis('off')
    plt.tight_layout()
    
    # Save the image
    static_folder = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'static')
    if not os.path.exists(static_folder):
        os.makedirs(static_folder)
    file_name = 'text_analysis_graph.png'
    file_path = os.path.join(static_folder, file_name)
    plt.savefig(file_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    logger.info(f"Graph saved to {file_path}")
    return file_path

def analyze_text(text):
    keywords = extract_keywords_tfidf(text)
    topics = perform_topic_modeling(text)
    causal_relations = extract_causal_relations(text)
    visualization_path = visualize_analysis(keywords, topics, causal_relations)
    
    return {
        'keywords': keywords,
        'topics': topics,
        'causal_relations': causal_relations,
        'visualization_path': visualization_path
    }