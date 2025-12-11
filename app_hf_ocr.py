# app_hf_ocr.py
import os
import io
import re
import time
import base64
import requests
import streamlit as st
import pandas as pd
import pdfplumber
from collections import defaultdict
from datetime import datetime
from difflib import SequenceMatcher

# try pypdfium2 for rendering pages to images
try:
    import pypdfium2 as pdfium
    from PIL import Image
    HAS_PDFIUM = True
except Exception:
    pdfium = None
    Image = None
    HAS_PDFIUM = False

st.set_page_config(page_title="Auditor TR - OCR HuggingFace", layout="wide")
st.title("Auditor TR — OCR via Hugging Face + Extração")

# ---------- Helpers ----------
def similarity(a,b):
    if not a or not b: return 0.0
    return SequenceMatcher(None, str(a).lower(), str(b).lower()).ratio()

def render_pdf_page_to_png_bytes(pdf_bytes, page_index, scale=2.0):
    """Render a page to PNG bytes using pypdfium2 (if available)."""
    if not HAS_PDFIUM:
        return None
    pdf = pdfium.PdfDocument(pdf_bytes)
    page = pdf.get_page(page_index)
    bitmap = page.render(scale=scale)
    pil = bitmap.to_pil()
    bio = io.BytesIO()
    pil.save(bio, format="PNG")
    return bio.getvalue()

def hf_ocr_image_bytes(image_bytes, model="microsoft/trocr-base-printed"):
    """Call Hugging Face Inference API for an image. Returns text or raises."""
    hf_token = None
    if st.secrets and "HF_API_TOKEN" in st.secrets:
        hf_token = st.secrets["HF_API_TOKEN"]
    else:
        hf_token = os.environ.get("HF_API_TOKEN")
    if not hf_token:
        raise RuntimeError("HF_API_TOKEN not configured in secrets/env")
    headers = {"Authorization": f"Bearer {hf_token}"}
    api_url = f"https://api-inference.huggingface.co/models/{model}"
    # send as binary image; HF Inference accepts raw image bytes for many models
    resp = requests.post(api_url, headers=headers, data=image_bytes, timeout=120)
    if resp.status_code == 200:
        # many OCR models return simple text in resp.text or JSON
        try:
            # some models return JSON with 'generated_text'
            j = resp.json()
            if isinstance(j, dict) and ("generated_text" in j):
                return j["generated_text"]
            # other models return list of dicts with 'generated_text'
            if isinstance(j, list) and len(j) and "generated_text" in j[0]:
                return j[0]["generated_text"]
            # fallback: use text body
            return resp.text
        except Exception:
            return resp.text
    else:
        raise RuntimeError(f"HuggingFace inference failed ({resp.status_code}): {resp.text[:400]}")

# ---------- Minimal heuristic table builder ----------
def lines_to_rows_from_page_text(text):
    """
    Heuristic: split text by line breaks. Return list of rows (list of cells).
    This is naive — improvements possible: split by multiple spaces or number columns.
    """
    rows = []
    for ln in text.splitlines():
        ln = ln.strip()
        if not ln: continue
        # split by multiple spaces (likely column separator from OCR)
        parts = re.split(r'\s{2,}', ln)
        if len(parts) == 1:
            # try splitting by tab-like spacing
            parts = re.split(r'\s{1,}', ln)
        rows.append([p.strip() for p in parts if p.strip()])
    return rows

# ---------- Simple pipeline ----------
uploaded = st.file_uploader("Envie o TR (PDF) para OCR+extração", type=["pdf"])
model_choice = st.selectbox("Modelo OCR HF (padrão trocr)", ["microsoft/trocr-base-printed", "microsoft/trocr-small-printed"])
if not uploaded:
    st.info("Envie o PDF para começar")
    st.stop()

pdf_bytes = uploaded.read()
# try to determine number of pages with pdfplumber
with pdfplumber.open(io.BytesIO(pdf_bytes)) as tmp_pdf:
    total_pages = len(tmp_pdf.pages)

st.info(f"PDF com {total_pages} páginas. Iniciando OCR via Hugging Face...")

all_page_texts = []
errors = []
for p in range(total_pages):
    st.write(f"Página {p+1}/{total_pages} — convertendo para imagem e enviando ao HF OCR...")
    image_bytes = None
    if HAS_PDFIUM:
        try:
            image_bytes = render_pdf_page_to_png_bytes(pdf_bytes, p, scale=2.0)
        except Exception as e:
            errors.append(f"Render page {p} failed: {e}")
            image_bytes = None
    # fallback: extract page as image via pdfplumber (less reliable)
    if image_bytes is None:
        try:
            with pdfplumber.open(io.BytesIO(pdf_bytes)) as doc:
                page = doc.pages[p]
                pil = page.to_image(resolution=150).original
                bio = io.BytesIO()
                pil.save(bio, format="PNG")
                image_bytes = bio.getvalue()
        except Exception as e:
            errors.append(f"pdfplumber render failed page {p}: {e}")
            image_bytes = None

    if image_bytes is None:
        st.error(f"Não foi possível renderizar a página {p+1} como imagem.")
        continue

    try:
        text = hf_ocr_image_bytes(image_bytes, model=model_choice)
        st.write(f"— OCR página {p+1}: {len(text.splitlines())} linhas extraídas")
        all_page_texts.append({"page": p+1, "text": text})
    except Exception as e:
        errors.append(f"HuggingFace OCR failed page {p+1}: {e}")
        all_page_texts.append({"page": p+1, "text": ""})

# combine texts and show sample
combined_text = "\n\n=== PAGE BREAK ===\n\n".join([pt["text"] for pt in all_page_texts])
st.subheader("Texto OCR combinado (amostra)")
st.text_area("OCR", combined_text[:8000], height=300)

# heuristic: try to find lines that look like table rows (contain numbers, CATMAT-like codes)
candidate_lines = []
for page_info in all_page_texts:
    for ln in page_info["text"].splitlines():
        ln2 = ln.strip()
        if not ln2: continue
        # select lines containing 5-7 digit sequences (possible CATMAT) or price patterns
        if re.search(r"\b\d{5,7}\b", ln2) or re.search(r"R\$\s*\d+[.,]\d{2}", ln2) or re.search(r"\b\d+\s+\w+\b", ln2):
            candidate_lines.append(ln2)

st.write(f"Linhas candidatas à tabela detectadas: {len(candidate_lines)}")
# build rows heuristically
rows = [re.split(r'\s{2,}', ln) for ln in candidate_lines]
# normalize rows to same column count
maxcol = max((len(r) for r in rows), default=1)
normalized_rows = [r + [""]*(maxcol-len(r)) for r in rows]

df = pd.DataFrame(normalized_rows)
st.subheader("Tabela heurística (preview)")
st.dataframe(df.head(200), height=300)

# Optionally: send rows JSON to ChatGPT for clean HTML (if you set OPENAI_API_KEY in secrets)
if st.checkbox("Gerar HTML via OpenAI (opcional)"):
    if "OPENAI_API_KEY" not in (st.secrets or {}):
        st.warning("Adicione OPENAI_API_KEY em Secrets para usar esta opção.")
    else:
        import openai
        openai.api_key = st.secrets["OPENAI_API_KEY"]
        rows_json = df.fillna("").to_dict(orient="records")
        prompt = "Gere um HTML tabelado (<table>) a partir deste JSON. Mantenha colunas e cabeçalhos simples. JSON:\n" + str(rows_json[:200])
        try:
            completion = openai.ChatCompletion.create(
                model="gpt-4o-mini",
                messages=[{"role":"user","content":prompt}],
                max_tokens=1200,
                temperature=0.0
            )
            html = completion.choices[0].message.content
            st.markdown("HTML gerado:")
            st.components.v1.html(html, height=600, scrolling=True)
            st.download_button("Baixar HTML", html.encode("utf-8"), file_name="tabela_chatgpt.html", mime="text/html")
        except Exception as e:
            st.error("Falha gerando HTML via OpenAI: " + str(e))

st.write("Erros detectados durante execução:")
for e in errors:
    st.write("-", e)

st.success("Pipeline HF-OCR concluído. Faça o download do HTML/Excel e valide como quiser.")
