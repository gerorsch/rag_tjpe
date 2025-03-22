
import os
from pathlib import Path
import docx
import re
import numpy as np
import faiss
import torch
from transformers import AutoTokenizer, AutoModel

# Configurações
BASE_DIR = "D:\Gerorsch\CAP"
MODEL_NAME = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
MAX_TOKENS = 300

# Inicializa modelo e tokenizer
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModel.from_pretrained(MODEL_NAME)

def extract_text_from_docx(docx_path):
    try:
        doc = docx.Document(docx_path)
        full_text = [para.text.strip() for para in doc.paragraphs if para.text.strip()]
        return "\n".join(full_text)
    except Exception as e:
        print(f"Erro ao ler {docx_path}: {e}")
        return ""

def find_relevant_start(text):
    start_keywords = ["SENTENÇA", "DESPACHO", "DECISÃO", "Vistos, etc.", "RELATÓRIO"]
    for keyword in start_keywords:
        match = re.search(rf"\b{re.escape(keyword)}\b", text, re.IGNORECASE)
        if match:
            return text[match.start():]
    return text

def chunk_text(text, max_tokens=MAX_TOKENS):
    paragraphs = text.split("\n")
    chunks, current_chunk = [], ""
    for para in paragraphs:
        if len(tokenizer.tokenize(current_chunk + para)) <= max_tokens:
            current_chunk += para + "\n"
        else:
            chunks.append(current_chunk.strip())
            current_chunk = para + "\n"
    if current_chunk:
        chunks.append(current_chunk.strip())
    return chunks

def generate_embedding(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
    with torch.no_grad():
        output = model(**inputs)
    return output.last_hidden_state.mean(dim=1).squeeze().numpy()

def collect_docx_files(base_path):
    return list(Path(base_path).rglob("*.docx"))

def process_documents(doc_paths):
    all_chunks, embeddings = [], []
    for path in doc_paths:
        raw_text = extract_text_from_docx(path)
        cleaned_text = find_relevant_start(raw_text)
        chunks = chunk_text(cleaned_text)
        for chunk in chunks:
            all_chunks.append((str(path), chunk))
            embeddings.append(generate_embedding(chunk))
    return all_chunks, np.array(embeddings)

def build_faiss_index(embeddings):
    dim = embeddings.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(embeddings)
    return index

def semantic_search(query, index, all_chunks, top_k=5):
    query_embedding = generate_embedding(query)
    D, I = index.search(np.array([query_embedding]), top_k)
    return [all_chunks[i] for i in I[0]]

if __name__ == "__main__":
    doc_paths = collect_docx_files(BASE_DIR)
    print(f"Documentos encontrados: {len(doc_paths)}")
    all_chunks, embeddings = process_documents(doc_paths)
    index = build_faiss_index(embeddings)

    # Exemplo de busca
    query = "falha na prestação de serviço de segurança"
    results = semantic_search(query, index, all_chunks)
    print("\nResultados para a consulta:\n")
    for path, text in results:
        print(f"Arquivo: {path}\nTrecho: {text}\n{'-'*60}")
