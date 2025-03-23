
import os
import json
import re
from pathlib import Path
from docx import Document
from tqdm import tqdm

# Caminho da pasta com os arquivos .docx
BASE_DIR = "/home/gerorsch/Documents/CAP"
OUTPUT_PATH = "dataset_tjpe.jsonl"

# Padrões por tipo
PADROES = {
    "sentenca": ["sentença", "julgo", "relatório", "dispositivo"],
    "despacho": ["despacho", "intimo", "determino", "encaminhe-se"],
    "decisao": ["decido", "defiro", "indefiro", "determino"],
}

# Função para extrair texto de um .docx
def extrair_texto_docx(path):
    try:
        doc = Document(path)
        texto = "\n".join([p.text for p in doc.paragraphs if p.text.strip()])
        return texto.strip()
    except Exception as e:
        print(f"Erro ao ler {path}: {e}")
        return ""

# Função para classificar o tipo do documento
def classificar_tipo(texto):
    texto_lower = texto.lower()
    for tipo, padroes in PADROES.items():
        for padrao in padroes:
            if padrao in texto_lower:
                return tipo
    return "indefinido"

# Criação do dataset
def processar_documentos(base_dir, output_path):
    arquivos = list(Path(base_dir).rglob("*.docx"))
    print(f"📄 Encontrados {len(arquivos)} arquivos .docx.")

    with open(output_path, "w", encoding="utf-8") as saida:
        for caminho in tqdm(arquivos, desc="Processando documentos"):
            texto = extrair_texto_docx(caminho)
            if not texto:
                continue
            tipo = classificar_tipo(texto)
            if tipo == "indefinido":
                continue  # ignora documentos sem classificação confiável

            prompt = f"Elabore um(a) {tipo.upper()} com base nos autos apresentados:"
            exemplo = {
                "input": prompt,
                "output": texto
            }
            saida.write(json.dumps(exemplo, ensure_ascii=False) + "\n")

    print(f"✅ Dataset salvo em: {output_path}")

if __name__ == "__main__":
    processar_documentos(BASE_DIR, OUTPUT_PATH)
