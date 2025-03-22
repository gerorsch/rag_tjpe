
import os
import subprocess
from pathlib import Path

def convert_doc_to_docx(base_dir):
    doc_files = list(Path(base_dir).rglob("*.doc"))
    print(f"Encontrados {len(doc_files)} arquivos .doc para conversão.")

    for doc_path in doc_files:
        docx_output_dir = doc_path.parent
        try:
            subprocess.run([
                "libreoffice",
                "--headless",
                "--convert-to", "docx",
                "--outdir", str(docx_output_dir),
                str(doc_path)
            ], check=True)
            print(f"✔ Convertido: {doc_path}")
        except subprocess.CalledProcessError as e:
            print(f"❌ Erro ao converter {doc_path}: {e}")

if __name__ == "__main__":
    BASE_DIR = "D:/Gerorsch/CAP"  # <-- Altere para o caminho da sua base
    convert_doc_to_docx(BASE_DIR)
