import streamlit as st
import pandas as pd
import pdfplumber
import io
from pdfminer.high_level import extract_text
import pdfminer
import pypdfium2 as pdfium

# Tenta carregar OCR
OCR_AVAILABLE = False
try:
    import pytesseract
    OCR_AVAILABLE = True
except:
    OCR_AVAILABLE = False


st.set_page_config(page_title="Auditor TR", layout="wide")
st.title("📄 Auditor de Termo de Referência — CATMAT / Extração Tabela")


# ============================================================
# NORMALIZAÇÃO
# ============================================================

def clean_text(value):
    if isinstance(value, str):
        return " ".join(value.replace("\n", " ").split())
    return value


# ============================================================
# EXTRAÇÃO COM PDFPLUMBER
# ============================================================

def extract_with_pdfplumber(pdf_file):
    try:
        tables = []
        with pdfplumber.open(pdf_file) as pdf:
            for page in pdf.pages:
                extracted = page.extract_tables()
                for tb in extracted:
                    df = pd.DataFrame(tb)
                    df = df.applymap(clean_text)
                    # ignora tabelas muito pequenas (ruído)
                    if df.shape[1] >= 3:
                        tables.append(df)
        return tables
    except:
        return []


# ============================================================
# EXTRAÇÃO COM PDFMINER (fallback)
# ============================================================

def extract_with_pdfminer(pdf_file):
    try:
        pdf_file.seek(0)
        text = extract_text(pdf_file)
        lines = [l.strip() for l in text.split("\n") if l.strip()]

        # heurística simples para detectar tabela textual
        rows = []
        for l in lines:
            parts = [p for p in l.split(" ") if p]
            if len(parts) >= 3:
                rows.append(parts)

        if len(rows) == 0:
            return []
        return [pd.DataFrame(rows)]
    except:
        return []


# ============================================================
# OCR COM PYPDFIUM2 (corrigido)
# ============================================================

def extract_with_ocr(pdf_bytes):
    if not OCR_AVAILABLE:
        return []

    try:
        pdf = pdfium.PdfDocument(pdf_bytes)
        ocr_text = ""

        for i in range(len(pdf)):
            page = pdf[i]
            bitmap = page.render(scale=2.0)  # 150–200 dpi
            pil_img = bitmap.to_pil()
            txt = pytesseract.image_to_string(pil_img, lang="por+eng")
            ocr_text += txt + "\n"

        # Monta tabela simplificada
        lines = [l.strip() for l in ocr_text.split("\n") if l.strip()]
        rows = []
        for l in lines:
            parts = [p for p in l.split(" ") if p]
            if len(parts) >= 3:
                rows.append(parts)

        if len(rows) == 0:
            return []
        return [pd.DataFrame(rows)]

    except Exception as e:
        st.warning(f"OCR falhou: {e}")
        return []


# ============================================================
# PIPELINE HÍBRIDO
# ============================================================

def extract_table_hybrid(uploaded_file):
    st.info("🔍 Tentando extrair com pdfplumber…")
    tables = extract_with_pdfplumber(uploaded_file)
    if len(tables) > 0:
        return tables

    st.info("🔍 pdfplumber falhou. Tentando PDFMiner…")
    uploaded_file.seek(0)
    tables = extract_with_pdfminer(uploaded_file)
    if len(tables) > 0:
        return tables

    st.info("🔍 PDFMiner falhou. Tentando OCR (somente local)…")
    uploaded_file.seek(0)
    pdf_bytes = uploaded_file.read()
    tables = extract_with_ocr(pdf_bytes)
    if len(tables) > 0:
        return tables

    return []


# ============================================================
# HTML FORMATADO
# ============================================================

def df_to_html(df):
    df = df.reset_index(drop=True)
    df.columns = [f"Col {i+1}" for i in range(len(df.columns))]

    html = "<table border='1' cellpadding='5' cellspacing='0' width='100%'>"
    html += "<thead><tr>"
    for col in df.columns:
        html += f"<th>{col}</th>"
    html += "</tr></thead><tbody>"

    for _, row in df.iterrows():
        html += "<tr>"
        for cell in row:
            html += f"<td>{cell}</td>"
        html += "</tr>"

    html += "</tbody></table>"
    return html


# ============================================================
# INTERFACE
# ============================================================

uploaded_file = st.file_uploader("Envie o PDF do Termo de Referência", type=["pdf"])

if uploaded_file:
    st.success("PDF carregado com sucesso!")

    tables = extract_table_hybrid(uploaded_file)

    if len(tables) == 0:
        st.error(
            "❌ Não foi possível extrair itens do PDF automaticamente.\n"
            "Se for um PDF escaneado, faça OCR antes ou use um PDF pesquisável."
        )
    else:
        st.success(f"✔ {len(tables)} tabela(s) extraída(s) com sucesso!")

        # Combina tudo em um único DataFrame
        df_final = pd.concat(tables, ignore_index=True)
        df_final = df_final.applymap(clean_text)

        # Mostra no Streamlit
        st.dataframe(df_final, height=500, width="stretch")

        # HTML
        html_result = df_to_html(df_final)
        st.download_button(
            "⬇ Baixar HTML",
            html_result,
            file_name="tabela_tr.html",
            mime="text/html"
        )

        # Excel
        buffer = io.BytesIO()
        df_final.to_excel(buffer, index=False, engine="openpyxl")
        st.download_button(
            "⬇ Baixar Excel",
            buffer.getvalue(),
            file_name="tabela_tr.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )
