# app.py
"""
Auditor TR — Modo Híbrido (pdfplumber -> pdfminer -> OCR)
Versão genérica e robusta para uso no Streamlit (compatível com streamlit.app).
Observações importantes sobre OCR: ver README.md (binário Tesseract).
"""

import streamlit as st
import pandas as pd
import pdfplumber
import re
import requests
import io
from datetime import datetime
from bs4 import BeautifulSoup

# Optional OCR imports (wrapped to avoid hard failure if not available)
try:
    import pypdfium2 as pdfium
    from PIL import Image
    import pytesseract
    OCR_AVAILABLE = True
except Exception:
    pdfium = None
    Image = None
    pytesseract = None
    OCR_AVAILABLE = False

# ---------- Config ----------
st.set_page_config(page_title="Auditor TR - Extração Híbrida", layout="wide")
st.title("🛡️ Auditor TR — Extração Híbrida (pdfplumber/pdfminer/OCR)")

# ---------- Utilitários ----------
def clean_number(value):
    if value is None:
        return 0.0
    s = str(value).strip().replace("R$", "").replace("\xa0", " ")
    s = s.replace(".", "").replace(",", ".")
    s = re.sub(r"[^\d\.\-]", "", s)
    try:
        return float(s) if s not in ("", "-", None) else 0.0
    except:
        return 0.0

def normalize_text(v):
    if pd.isna(v):
        return ""
    return str(v).strip()

# ---------- Consulta CATMAT / CATSER ----------
@st.cache_data(show_spinner=False)
def consultar_item_cat(codigo):
    code = re.sub(r"\D", "", str(codigo))
    if not code:
        return {"status_api": "Inválido", "codigo": codigo, "descricao": "", "unidade": "-", "link": ""}
    # Try materials
    try:
        url_mat = f"https://compras.dados.gov.br/materiais/v1/materiais.json?codigo={code}"
        r = requests.get(url_mat, timeout=6)
        if r.status_code == 200:
            j = r.json()
            mats = j.get("_embedded", {}).get("materiais", [])
            if mats:
                item = mats[0]
                unidade = item.get("unidade_medida") or item.get("unidade") or "-"
                return {"status_api":"Encontrado-Mat","tipo":"Material","codigo":code,"descricao":item.get("descricao","").strip(),"unidade":unidade,"link":f"https://catalogo.compras.gov.br/cnbs-web/busca?cod={code}"}
    except Exception:
        pass
    # Try services
    try:
        url_srv = f"https://compras.dados.gov.br/servicos/v1/servicos.json?codigo={code}"
        r = requests.get(url_srv, timeout=6)
        if r.status_code == 200:
            j = r.json()
            srvs = j.get("_embedded", {}).get("servicos", [])
            if srvs:
                item = srvs[0]
                unidade = item.get("unidade") or "UN"
                return {"status_api":"Encontrado-Serv","tipo":"Servico","codigo":code,"descricao":item.get("descricao","").strip(),"unidade":unidade,"link":f"https://catalogo.compras.gov.br/cnbs-web/busca?cod={code}"}
    except Exception:
        pass
    return {"status_api":"NaoEncontrado","tipo":None,"codigo":code,"descricao":"","unidade":"-","link":f"https://catalogo.compras.gov.br/cnbs-web/busca?cod={code}"}

# ---------- Extração Híbrida ----------
HEADER_KEYWORDS = [
    "item","descr","descricao","unidade","catmat","catser","qtd","quant","quantidade",
    "preco","preço","unit","unitario","total","são paulo","sp","rio","recife","manaus","caeté","caete"
]

def find_header_index(rows):
    for i, row in enumerate(rows[:40]):
        if not any(cell for cell in row):
            continue
        row_text = " ".join([str(x).lower() for x in row if x])
        score = sum(1 for kw in HEADER_KEYWORDS if kw in row_text)
        if score >= 2:
            return i
    return None

def extract_with_pdfplumber(file_stream):
    rows = []
    full_text = ""
    try:
        with pdfplumber.open(file_stream) as pdf:
            for page in pdf.pages:
                text = page.extract_text() or ""
                full_text += text + "\n"
                try:
                    tables = page.extract_tables(table_settings={"vertical_strategy":"lines","horizontal_strategy":"lines"})
                    if not tables:
                        tables = page.extract_tables()
                except Exception:
                    tables = page.extract_tables()
                if tables:
                    for t in tables:
                        for r in t:
                            row = [("" if c is None else c) for c in r]
                            if any(str(x).strip() for x in row):
                                rows.append(row)
    except Exception as e:
        st.warning(f"pdfplumber error: {e}")
    return rows, full_text

def extract_text_pdfminer(file_bytes):
    # Use pdfminer.six: fallback to full text extraction, then try regex-based table extraction
    try:
        from pdfminer.high_level import extract_text
        txt = extract_text(io.BytesIO(file_bytes))
        return txt or ""
    except Exception as e:
        st.warning("pdfminer extraction failed: " + str(e))
        return ""

def ocr_pdf_to_text(file_bytes):
    """Render pages to images with pypdfium2 and OCR with pytesseract (if available)."""
    if not OCR_AVAILABLE:
        return ""
    try:
        pdf = pdfium.PdfDocument(file_bytes)
        full_ocr = ""
        for i in range(len(pdf)):
            pil = pdf.render_topil(i, scale=150/72)  # decent resolution
            txt = pytesseract.image_to_string(pil, lang='por+eng')
            full_ocr += txt + "\n"
        return full_ocr
    except Exception as e:
        st.warning("OCR failed: " + str(e))
        return ""

def rows_to_dataframe(rows, full_text):
    std_cols = ["Grupo","Item","Descrição","Unidade","CATMAT","QTD","São Paulo","Rio de Janeiro","Caeté","Manaus","Recife","Preço Unitário (R$)","Preço Total (R$)"]
    if rows:
        header_idx = find_header_index(rows)
        if header_idx is None:
            # fallback: take very first row as header candidate among top 6
            for i, r in enumerate(rows[:6]):
                if any(str(x).strip() for x in r):
                    header_idx = i
                    break
        if header_idx is not None and header_idx < len(rows)-1:
            header_row = [normalize_text(c) for c in rows[header_idx]]
            headers = []
            for i,h in enumerate(header_row):
                name = h if h else f"col{i}"
                headers.append(name)
            data_rows = rows[header_idx+1:]
            processed = []
            for r in data_rows:
                row = [("" if c is None else c) for c in r]
                if len(row) < len(headers):
                    row += [""]*(len(headers)-len(row))
                processed.append(row[:len(headers)])
            df = pd.DataFrame(processed, columns=headers)
            df = df.applymap(lambda x: normalize_text(x) if isinstance(x, str) else x)
            ren = {}
            for c in df.columns:
                lc = c.lower()
                if "descr" in lc or "espec" in lc:
                    ren[c] = "Descrição"
                elif "cat" in lc or "cod" in lc or "cód" in lc:
                    ren[c] = "CATMAT"
                elif "unid" in lc or re.match(r"^u[np]$", lc):
                    ren[c] = "Unidade"
                elif "qtd" in lc or "quant" in lc:
                    ren[c] = "QTD"
                elif "são paulo" in lc or lc.strip() in ("sp","são paulo","sao paulo"):
                    ren[c] = "São Paulo"
                elif "rio" in lc:
                    ren[c] = "Rio de Janeiro"
                elif "recife" in lc:
                    ren[c] = "Recife"
                elif "manaus" in lc:
                    ren[c] = "Manaus"
                elif "caeté" in lc or "caete" in lc:
                    ren[c] = "Caeté"
                elif ("preço unit" in lc or "preco unit" in lc or ("unit" in lc and "total" not in lc)) or ("valor unit" in lc):
                    ren[c] = "Preço Unitário (R$)"
                elif ("preço total" in lc or "preco total" in lc or "valor total" in lc) or (("total" in lc) and ("preço" in lc or "preco" in lc or "valor" in lc)):
                    ren[c] = "Preço Total (R$)"
                elif "item" in lc and lc.strip() != "descricao":
                    ren[c] = "Item"
            df = df.rename(columns=ren)
            for c in std_cols:
                if c not in df.columns:
                    df[c] = ""
            # Try quick group detection
            current_group = "SEM GRUPO"
            m = re.search(r"(GRUPO\s*\d+.*?)\n", full_text, flags=re.IGNORECASE)
            if m:
                current_group = m.group(1).strip()
            df["Grupo"] = current_group
            df = df[["Grupo","Item","Descrição","Unidade","CATMAT","QTD","São Paulo","Rio de Janeiro","Caeté","Manaus","Recife","Preço Unitário (R$)","Preço Total (R$)"]]
            for c in ["QTD","Preço Unitário (R$)","Preço Total (R$)"]:
                if c in df.columns:
                    df[c] = df[c].apply(clean_number)
            return df
    # Fallback: try extracting CATMAT-like codes from text
    codes = list(dict.fromkeys(re.findall(r"\b\d{5,7}\b", full_text)))
    rows_out = []
    for c in codes:
        rows_out.append({"Grupo":"SEM GRUPO","Item":"","Descrição":"","Unidade":"","CATMAT":c,"QTD":0,"São Paulo":0,"Rio de Janeiro":0,"Caeté":0,"Manaus":0,"Recife":0,"Preço Unitário (R$)":0,"Preço Total (R$)":0})
    if rows_out:
        return pd.DataFrame(rows_out)
    return pd.DataFrame(columns=std_cols)

# ---------- UI ----------
st.sidebar.header("Configurações")
st.sidebar.write("Modo híbrido: pdfplumber → pdfminer → OCR (se disponível).")
uploaded = st.file_uploader("📂 Envie o TR (PDF)", type=["pdf"])
if not uploaded:
    st.info("Envie o PDF do Termo de Referência para iniciar.")
    st.stop()

file_bytes = uploaded.read()

# Step 1: pdfplumber
with st.spinner("Tentando extração com pdfplumber..."):
    rows, full_text = extract_with_pdfplumber(io.BytesIO(file_bytes))

df = rows_to_dataframe(rows, full_text)

# Step 2: if empty, try pdfminer
if df.empty or len(df) < 1:
    with st.spinner("pdfplumber não detectou linhas. Tentando extração textual com pdfminer..."):
        text_pdfminer = extract_text_pdfminer(file_bytes)
        if text_pdfminer:
            # crude attempt: split lines and attempt to form rows where numeric columns present
            lines = [l.strip() for l in text_pdfminer.splitlines() if l.strip()]
            # Heuristic: lines containing a CATMAT-like code (5-7 digits) likely item lines
            candidate_rows = []
            for ln in lines:
                if re.search(r"\b\d{5,7}\b", ln):
                    candidate_rows.append([ln])
            if candidate_rows:
                df = rows_to_dataframe(candidate_rows, text_pdfminer)

# Step 3: OCR fallback (only if needed)
if (df.empty or len(df) < 1) and OCR_AVAILABLE:
    with st.spinner("Tentando OCR com pytesseract (pypdfium2)... isso pode demorar..."):
        ocr_text = ocr_pdf_to_text(file_bytes)
        if ocr_text:
            lines = [l.strip() for l in ocr_text.splitlines() if l.strip()]
            candidate = []
            for ln in lines:
                if re.search(r"\b\d{5,7}\b", ln):
                    candidate.append([ln])
            if candidate:
                df = rows_to_dataframe(candidate, ocr_text)

# Final check
if df.empty or len(df) < 1:
    st.error("Não foi possível extrair itens do PDF automaticamente. Se for um PDF escaneado (imagem), ative OCR ou envie uma versão com texto (PDF pesquisável).")
    # show some raw text to help debugging
    st.subheader("Trecho de texto extraído (amostra):")
    sample = full_text[:400] if full_text else ""
    if not sample and OCR_AVAILABLE:
        sample = ocr_pdf_to_text(file_bytes)[:400]
    st.code(sample or "— sem texto extraído —")
    st.stop()

# Show results
st.markdown("### ✅ Tabela extraída (prévia)")
st.write(f"Linhas: {len(df)} — Colunas: {', '.join(df.columns)}")
st.dataframe(df, width="stretch", height=360)

# Downloads and CATMAT check
c1, c2, c3 = st.columns([1,1,1])
with c1:
    if st.button("📄 Gerar/baixar HTML tabulado"):
        html_body = ""
        groups = df["Grupo"].fillna("SEM GRUPO").unique().tolist()
        for g in groups:
            sub = df[df["Grupo"]==g].copy()
            html_body += f"<h3>{g}</h3>" + sub.to_html(index=False, escape=True)
        full_html = f"<!doctype html><html lang='pt-BR'><head><meta charset='utf-8'><title>TR - Tabela</title></head><body><h2>Termo de Referência — Itens (extraído)</h2>{html_body}</body></html>"
        st.download_button("⬇️ Baixar HTML", data=full_html.encode("utf-8"), file_name="tabela_final.html", mime="text/html")

with c2:
    if st.button("⬇️ Gerar e baixar Excel consolidado"):
        buffer = io.BytesIO()
        with pd.ExcelWriter(buffer, engine="openpyxl") as writer:
            df.to_excel(writer, sheet_name="Itens", index=False)
        buffer.seek(0)
        st.download_button("⬇️ Baixar Excel", data=buffer.getvalue(), file_name="tabela_final.xlsx", mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")

with c3:
    if st.button("🔎 Consultar CATMAT para códigos detectados"):
        st.info("Executando varredura — aguarde (cada consulta usa a API pública).")
        code_col = None
        for c in df.columns:
            if c.lower() in ("catmat","catser","codigo","cod","cód"):
                code_col = c; break
        if not code_col:
            for c in df.columns:
                sample = df[c].astype(str).head(30).tolist()
                if any(re.search(r"\b\d{5,7}\b", s) for s in sample):
                    code_col = c; break
        if not code_col:
            st.error("Não localizei uma coluna de códigos (CATMAT/CATSER). Renomeie a coluna ou informe manualmente.")
        else:
            st.info(f"Usando coluna: {code_col}")
            codes = df[code_col].astype(str).fillna("").unique().tolist()
            codes = [re.sub(r"\D","",c) for c in codes if re.search(r"\d{5,7}", str(c))]
            total = len(codes)
            progress = st.progress(0)
            results = []
            for i, code in enumerate(codes):
                progress.progress((i+1)/max(1,total))
                res = consultar_item_cat(code)
                results.append(res)
            df_cat = pd.DataFrame(results)
            buffer = io.BytesIO()
            with pd.ExcelWriter(buffer, engine="openpyxl") as writer:
                df.to_excel(writer, sheet_name="Itens", index=False)
                df_cat.to_excel(writer, sheet_name="CATMAT", index=False)
            buffer.seek(0)
            st.download_button("⬇️ Baixar Excel (Itens + CATMAT)", data=buffer.getvalue(), file_name="auditoria_com_catmat.xlsx", mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
            st.markdown("#### Resultados da consulta CATMAT")
            st.dataframe(df_cat, width="stretch", height=300)

# Quick math checks
st.markdown("### ℹ️ Verificações rápidas")
if "QTD" in df.columns and "Preço Unitário (R$)" in df.columns and "Preço Total (R$)" in df.columns:
    df_check = df.copy()
    df_check["Total Calculado"] = df_check["QTD"].apply(clean_number) * df_check["Preço Unitário (R$)"].apply(clean_number)
    df_check["Diff"] = (df_check["Total Calculado"] - df_check["Preço Total (R$)"].apply(clean_number)).abs()
    df_check["Status Math"] = df_check["Diff"].apply(lambda d: "OK" if d <= 0.1 else "DIVERGENTE")
    problemas = df_check[df_check["Status Math"]!="OK"]
    st.write(f"Linhas com divergência matemática: {len(problemas)}")
    if not problemas.empty:
        st.dataframe(problemas[["Item","CATMAT","QTD","Preço Unitário (R$)","Preço Total (R$)","Total Calculado","Diff"]], width="stretch", height=260)
else:
    st.info("Colunas QTD e Preço Unitário/Total não detectadas juntas — verificação matemática desativada.")

st.markdown("---")
st.caption(f"Gerado em {datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S')} UTC")
