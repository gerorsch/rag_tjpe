
import faiss
import pickle
from transformers import AutoTokenizer, AutoModel
import torch
import numpy as np
from pathlib import Path
import docx
import re

# ------------------ Funções utilitárias ------------------

def extract_text_from_docx(docx_path):
    try:
        doc = docx.Document(docx_path)
        full_text = [para.text.strip() for para in doc.paragraphs if para.text.strip()]
        return "\n".join(full_text)
    except Exception as e:
        print(f"Erro ao ler {docx_path}: {e}")
        return ""

def generate_embedding(text, tokenizer, model):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
    with torch.no_grad():
        output = model(**inputs)
    return output.last_hidden_state.mean(dim=1).squeeze().numpy()

def semantic_search_with_full_doc(query, index, all_chunks, tokenizer, model, top_k=5):
    query_embedding = generate_embedding(query, tokenizer, model)
    D, I = index.search(np.array([query_embedding]), top_k)

    results = []
    seen_docs = {}

    for idx in I[0]:
        path, trecho = all_chunks[idx]
        if path not in seen_docs:
            full_text = extract_text_from_docx(path)
            seen_docs[path] = full_text
        results.append({
            "path": path,
            "trecho": trecho,
            "documento_completo": seen_docs[path]
        })

    return results

# ------------------ Inicialização ------------------

print("🔄 Carregando índice e chunks...")
index = faiss.read_index("base_tjpe.index")
with open("all_chunks.pkl", "rb") as f:
    all_chunks = pickle.load(f)

print("🧠 Carregando modelo de embeddings...")
model_name = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name)

# ------------------ Loop interativo ------------------

print("✅ Sistema pronto para busca!")
print("Digite sua pergunta ou 'sair' para encerrar.")

while True:
    query = input("Pergunta: ").strip()
    if query.lower() in ["sair", "exit", "quit"]:
        print("Encerrando.")
        break

    resultados = semantic_search_with_full_doc(query, index, all_chunks, tokenizer, model)

    print(f"🔎 {len(resultados)} resultados encontrados: ")
    
    for r in resultados:
        print(f"📄 Documento: {Path(r['path']).name}")
        print(f"Trecho relevante: {r['trecho']} ")
        print("-" * 80)
