# app.py
"""
Auditor TR — Extração Robusta + Validação com Planilha Oficial + OCR via OpenAI (opcional)
- Extração por coordenadas (pdfplumber)
- Fallbacks: pdfminer (texto) -> OCR local (pytesseract) -> OCR via OpenAI (se OPENAI_API_KEY configurada)
- Validação com planilha oficial CATMAT/CATSER (downloadada e cacheada)
- Opcional: gerar HTML via OpenAI (ChatGPT)
Notes:
 - Para OCR via OpenAI você precisa configurar OPENAI_API_KEY (st.secrets["OPENAI_API_KEY"]) e sua conta precisa ter permissões de file upload / responses que retornem PDF.
 - Se OpenAI OCR não funcionar, habilite OCR local (instale tesseract-ocr) e as libs pytesseract & pypdfium2 no requirements.txt.
"""

import os
import io
import re
import time
import base64
import streamlit as st
import pandas as pd
import pdfplumber
import requests
from collections import defaultdict
from datetime import datetime
from difflib import SequenceMatcher
from bs4 import BeautifulSoup

# Optional libs for local OCR / rendering
try:
    import pypdfium2 as pdfium
    from PIL import Image
    import pytesseract
    OCR_AVAILABLE_LOCAL = True
except Exception:
    pdfium = None
    Image = None
    pytesseract = None
    OCR_AVAILABLE_LOCAL = False

# Optional OpenAI SDK
try:
    import openai
    OPENAI_SDK_AVAILABLE = True
except Exception:
    openai = None
    OPENAI_SDK_AVAILABLE = False

# ---------- Streamlit UI config ----------
st.set_page_config(page_title="Auditor TR — OCR + CATMAT", layout="wide")
st.title("🛡️ Auditor TR — OCR (opcional OpenAI) + Extração + Validação CATMAT")

# ---------- Helpers ----------
def clean_number(value):
    if value is None:
        return 0.0
    s = str(value)
    s = s.replace("R$", "").replace("\xa0", " ").replace(".", "").replace(",", ".")
    s = re.sub(r"[^\d\.\-]", "", s)
    try:
        return float(s) if s != "" else 0.0
    except:
        return 0.0

def normalize_text(v):
    if pd.isna(v):
        return ""
    return " ".join(str(v).split())

def similarity(a, b):
    if not a or not b:
        return 0.0
    return SequenceMatcher(None, str(a).lower(), str(b).lower()).ratio()

# ---------- Catalog download & cache ----------
CATALOG_PAGE = "https://www.gov.br/compras/pt-br/acesso-a-informacao/consulta-detalhada/planilha-catmat-catser"

@st.cache_data(ttl=60*60*6, show_spinner=False)
def download_latest_catalog():
    """Download the official catalog file linked on gov.br and return (catmat_df, catser_df, meta)"""
    r = requests.get(CATALOG_PAGE, timeout=20)
    r.raise_for_status()
    soup = BeautifulSoup(r.text, "html.parser")
    candidates = []
    for a in soup.find_all("a", href=True):
        href = a["href"]
        if href.lower().endswith((".xlsx",".xls",".ods",".zip")):
            candidates.append(href)
    if not candidates:
        # fallback search by anchor text
        for a in soup.find_all("a", href=True):
            txt = (a.get_text() or "").lower()
            if "catmat" in txt or "catser" in txt:
                candidates.append(a["href"])
    if not candidates:
        raise RuntimeError("Não foi possível localizar o link da planilha no site do gov.br")
    link = candidates[0]
    if link.startswith("/"):
        link = "https://www.gov.br" + link
    r2 = requests.get(link, timeout=60)
    r2.raise_for_status()
    content = r2.content
    dfs = {}
    try:
        if link.lower().endswith(".zip") or "zip" in r2.headers.get("content-type",""):
            import zipfile
            z = zipfile.ZipFile(io.BytesIO(content))
            names = [n for n in z.namelist() if n.lower().endswith((".xlsx",".xls",".ods"))]
            for n in names:
                b = z.read(n)
                sheets = pd.read_excel(io.BytesIO(b), sheet_name=None)
                for k,v in sheets.items():
                    dfs[f"{n}::{k}"] = v
        else:
            sheets = pd.read_excel(io.BytesIO(content), sheet_name=None)
            for k,v in sheets.items():
                dfs[k] = v
    except Exception:
        try:
            df_single = pd.read_excel(io.BytesIO(content), sheet_name=0)
            dfs["sheet0"] = df_single
        except Exception as e:
            raise RuntimeError("Falha ao ler a planilha do catálogo: " + str(e))
    catmat_df, catser_df = None, None
    for name, df in dfs.items():
        cols = [str(c).lower() for c in df.columns]
        if any("catmat" in c for c in cols) or "materiais" in " ".join(cols):
            catmat_df = df.copy()
        if any("catser" in c for c in cols) or "servicos" in " ".join(cols):
            catser_df = df.copy()
    if catmat_df is None and len(dfs) == 1:
        catmat_df = next(iter(dfs.values())).copy()
    def clean_cols(df):
        df2 = df.copy()
        df2.columns = [str(c).strip() for c in df2.columns]
        return df2
    if catmat_df is not None: catmat_df = clean_cols(catmat_df)
    if catser_df is not None: catser_df = clean_cols(catser_df)
    meta = {"url": link, "fetched_at": time.time()}
    return catmat_df, catser_df, meta

# ---------- robust extraction by coords ----------
def cluster_positions(values, tol=10):
    if not values: return []
    vals = sorted(values)
    clusters = [[vals[0]]]
    for v in vals[1:]:
        if abs(v - clusters[-1][-1]) <= tol:
            clusters[-1].append(v)
        else:
            clusters.append([v])
    return [sum(c)/len(c) for c in clusters]

def extract_words_table_by_coords(file_stream):
    rows_out = []
    full_text = ""
    try:
        with pdfplumber.open(file_stream) as pdf:
            for page in pdf.pages:
                page_text = page.extract_text() or ""
                full_text += page_text + "\n"
                words = page.extract_words(use_text_flow=True, keep_blank_chars=False)
                if not words:
                    continue
                y_positions = [round(w.get("top",0)) for w in words]
                y_clusters = cluster_positions(y_positions, tol=3)
                lines_map = defaultdict(list)
                for w in words:
                    top = round(w.get("top",0))
                    nearest = min(y_clusters, key=lambda c: abs(c-top)) if y_clusters else top
                    lines_map[nearest].append(w)
                x_centers = [ (w.get("x0",0)+w.get("x1",0))/2.0 for w in words ]
                x_cols = cluster_positions(x_centers, tol=22)
                if not x_cols:
                    for _, ln_words in sorted(lines_map.items(), key=lambda kv: kv[0]):
                        ordered = sorted(ln_words, key=lambda w: w.get("x0",0))
                        rows_out.append([" ".join(w.get("text","") for w in ordered)])
                    continue
                for _, ln_words in sorted(lines_map.items(), key=lambda kv: kv[0]):
                    ordered = sorted(ln_words, key=lambda w: w.get("x0",0))
                    cells = [""] * len(x_cols)
                    for w in ordered:
                        x_center = (w.get("x0",0) + w.get("x1",0))/2.0
                        idx = min(range(len(x_cols)), key=lambda i: abs(x_cols[i] - x_center))
                        if cells[idx]:
                            cells[idx] += " " + w.get("text","")
                        else:
                            cells[idx] = w.get("text","")
                    rows_out.append([c.strip() for c in cells])
    except Exception as e:
        full_text += f"\n[pdfplumber-extract-error] {e}\n"
    return rows_out, full_text

def rows_to_df_from_coords(rows, full_text):
    std_cols = ["Grupo","Item","Descrição","Unidade","CATMAT","QTD","São Paulo","Rio de Janeiro","Caeté","Manaus","Recife","Preço Unitário (R$)","Preço Total (R$)"]
    if not rows:
        return pd.DataFrame(), full_text
    clean = []
    max_cols = 0
    for r in rows:
        r2 = [("" if (c is None) else str(c).strip()) for c in r]
        if any(cell.strip() for cell in r2):
            clean.append(r2)
            max_cols = max(max_cols, len(r2))
    if not clean:
        return pd.DataFrame(), full_text
    padded = []
    for r in clean:
        if len(r) < max_cols:
            r = r + [""]*(max_cols - len(r))
        padded.append(r[:max_cols])
    df_raw = pd.DataFrame(padded)
    hdr_keywords = ["descr", "descricao", "espec", "catmat", "catser", "unid", "unidade", "qtd", "quant", "preco", "preço", "item", "valor", "total"]
    header_idx = None
    for i in range(min(12, len(df_raw))):
        text = " ".join(df_raw.iloc[i].astype(str).str.lower().tolist())
        score = sum(1 for k in hdr_keywords if k in text)
        if score >= 2:
            header_idx = i
            break
    if header_idx is None:
        for i in range(min(6, len(df_raw))):
            if sum(1 for c in df_raw.iloc[i] if str(c).strip()) >= 2:
                header_idx = i
                break
    if header_idx is None:
        df_raw.columns = [f"col_{i+1}" for i in range(df_raw.shape[1])]
        return df_raw, full_text
    header = [str(x).strip() for x in df_raw.iloc[header_idx].tolist()]
    data = df_raw.iloc[header_idx+1:].reset_index(drop=True)
    data.columns = [h if h else f"col_{i+1}" for i,h in enumerate(header)]
    ren = {}
    for c in list(data.columns):
        lc = c.lower()
        if "descr" in lc or "espec" in lc:
            ren[c] = "Descrição"
        elif "cat" in lc or "cod" in lc or "cód" in lc:
            ren[c] = "CATMAT"
        elif "unid" in lc or re.match(r"^u[np]$", lc):
            ren[c] = "Unidade"
        elif "qtd" in lc or "quant" in lc:
            ren[c] = "QTD"
        elif ("preço" in lc or "preco" in lc or "valor unit" in lc) and "total" not in lc:
            ren[c] = "Preço Unitário (R$)"
        elif ("preço total" in lc or "preco total" in lc or "valor total" in lc) or ("total" in lc and ("preço" in lc or "preco" in lc or "valor" in lc)):
            ren[c] = "Preço Total (R$)"
        elif "item" in lc:
            ren[c] = "Item"
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
    data = data.rename(columns=ren)
    for c in std_cols:
        if c not in data.columns:
            data[c] = ""
    current_group = "SEM GRUPO"
    m = re.search(r"(GRUPO\s*\d+.*?)\n", full_text, flags=re.IGNORECASE)
    if m:
        current_group = m.group(1).strip()
    data["Grupo"] = current_group
    ordered = [c for c in std_cols if c in data.columns]
    extras = [c for c in data.columns if c not in ordered]
    data = data[ordered + extras]
    for c in ["QTD","Preço Unitário (R$)","Preço Total (R$)"]:
        if c in data.columns:
            data[c] = data[c].apply(clean_number)
    return data, full_text

# ---------- Match with catalog ----------
def match_item_with_catalog(code_pdf, desc_pdf, unit_pdf, catmat_df):
    if catmat_df is None:
        return {"found": False, "reason": "No catalog"}
    code_clean = re.sub(r"\D", "", str(code_pdf))
    code_cols = [c for c in catmat_df.columns if any(k in c.lower() for k in ("codigo","cod","catmat","cat"))]
    desc_cols = [c for c in catmat_df.columns if "descr" in c.lower() or "descricao" in c.lower()]
    unit_cols = [c for c in catmat_df.columns if any(k in c.lower() for k in ("unid","unidade","medida"))]
    df = catmat_df.copy()
    if code_cols:
        ccol = code_cols[0]
        df["_code_norm"] = df[ccol].astype(str).str.replace(r"\D","", regex=True)
        matches = df[df["_code_norm"] == code_clean]
        if not matches.empty:
            rec = matches.iloc[0].to_dict()
            desc_off = rec.get(desc_cols[0]) if desc_cols else ""
            unit_off = rec.get(unit_cols[0]) if unit_cols else ""
            sim_desc = similarity(desc_pdf, desc_off)
            unit_match = False
            if unit_off and unit_pdf:
                up = str(unit_pdf).strip().upper()
                uo = str(unit_off).strip().upper()
                unit_match = (up in uo) or (uo in up)
            return {"found": True, "matched_by": "code", "record": rec, "desc_official": desc_off, "unit_official": unit_off, "desc_similarity": sim_desc, "unit_match": unit_match}
    if desc_cols:
        dcol = desc_cols[0]
        df["_sim"] = df[dcol].astype(str).apply(lambda d: similarity(d, str(desc_pdf)))
        best = df.sort_values("_sim", ascending=False).head(5)
        top = best.iloc[0]
        rec = top.to_dict()
        return {"found": True, "matched_by": "desc_fuzzy", "record": rec, "desc_official": rec.get(dcol), "unit_official": rec.get(unit_cols[0]) if unit_cols else None, "desc_similarity": float(top.get("_sim", 0.0)), "unit_match": False}
    return {"found": False}

# ---------- OCR via OpenAI (attempt) ----------
def convert_pdf_to_searchable_via_openai(pdf_bytes):
    """
    Attempt to convert scanned PDF to searchable PDF using OpenAI Responses API (file upload).
    NOTE: This relies on your OpenAI account/SDK supporting file uploads to the responses endpoint and returning a PDF.
    There is no universal guarantee — if your account/SDK does not support it, this function will raise/return error.
    """
    if not OPENAI_SDK_AVAILABLE:
        return None, "OpenAI SDK not installed"
    key = None
    if st.secrets and "OPENAI_API_KEY" in st.secrets:
        key = st.secrets["OPENAI_API_KEY"]
    else:
        key = os.environ.get("OPENAI_API_KEY")
    if not key:
        return None, "OPENAI_API_KEY not configured"
    try:
        # Attempt with modern OpenAI python client pattern
        openai.api_key = key
        # Two strategies: (A) client.responses.create with files (if supported),
        # (B) fallback: send base64 in chat message (less ideal, may hit size limits)
        try:
            # Strategy A: responses.create with files (preferred if available)
            # NOTE: many accounts/SDKs vary; change if your SDK uses a client object
            if hasattr(openai, "Responses") or hasattr(openai, "responses"):
                # Build a request that instructs model to return base64 of searchable PDF
                # This block may need adaptation depending on SDK; keep try/except
                file_b64 = base64.b64encode(pdf_bytes).decode()
                prompt = ("Converta este PDF escaneado em um PDF pesquisável (OCR). "
                          "Retorne o PDF resultante codificado em base64, sem texto adicional; responda com apenas o base64.")
                resp = openai.responses.create(
                    model="gpt-4o-mini",
                    input=[{"role":"user","content": prompt + "\n\nArquivo(base64):\n" + file_b64}],
                    max_output_tokens=200000
                )
                # Try to extract text content as base64 from response
                # The exact path depends on the SDK/response shape
                text_out = ""
                try:
                    # new Responses API often returns content in resp.output_text or choices
                    text_out = resp.output_text if hasattr(resp, "output_text") else None
                except Exception:
                    text_out = None
                if not text_out:
                    # try older shape
                    try:
                        text_out = resp["output"][0]["content"][0]["text"]
                    except Exception:
                        text_out = None
                if not text_out:
                    return None, "OpenAI response did not include base64 PDF (response shape mismatch)."
                # decode
                pdf_new = base64.b64decode(text_out.strip())
                return pdf_new, None
            else:
                return None, "OpenAI Responses API with file upload not supported in this SDK"
        except Exception as e:
            return None, f"OpenAI OCR attempt failed: {e}"
    except Exception as e:
        return None, f"OpenAI client error: {e}"

# ---------- Local OCR fallback (pypdfium2 + pytesseract) ----------
def ocr_local_convert_pdf_to_searchable(pdf_bytes):
    """
    Local OCR fallback: render pages with pypdfium2 and run pytesseract, then construct text per page.
    This does not produce a full 'PDF with embedded text' automatically unless you stitch images into a PDF and embed text layers.
    Instead we return the OCR text (which the extractor can use).
    """
    if not OCR_AVAILABLE_LOCAL:
        return None, "Local OCR libs not available (pytesseract/pypdfium2)"
    try:
        pdf = pdfium.PdfDocument(pdf_bytes)
        pages_text = []
        for i in range(len(pdf)):
            page = pdf[i]
            bm = page.render(scale=2.0)
            pil = bm.to_pil()
            txt = pytesseract.image_to_string(pil, lang='por+eng')
            pages_text.append(txt)
        full_text = "\n".join(pages_text)
        return full_text, None
    except Exception as e:
        return None, str(e)

# ---------- OpenAI HTML generator (optional) ----------
def generate_html_with_gpt(rows_json):
    """Generate HTML table using OpenAI (ChatGPT). Requires OPENAI_API_KEY in secrets."""
    if not OPENAI_SDK_AVAILABLE:
        return None, "OpenAI SDK not installed"
    key = None
    if st.secrets and "OPENAI_API_KEY" in st.secrets:
        key = st.secrets["OPENAI_API_KEY"]
    else:
        key = os.environ.get("OPENAI_API_KEY")
    if not key:
        return None, "OPENAI_API_KEY not configured"
    openai.api_key = key
    prompt = ("Receba os dados JSON e gere somente um HTML <table> fiel. JSON:\n" + rows_json)
    try:
        # simple ChatCompletion usage; may be adapted to Responses API if desired
        completion = openai.ChatCompletion.create(
            model="gpt-4o-mini",
            messages=[{"role":"user","content":prompt}],
            max_tokens=2000,
            temperature=0.0
        )
        html = completion.choices[0].message.content
        return html, None
    except Exception as e:
        return None, str(e)

# ---------- UI ----------
st.sidebar.header("Configurações")
st.sidebar.markdown("- OCR: OpenAI (se configurado) → fallback local OCR (se disponível)\n- Cache da planilha: 6h")

uploaded = st.file_uploader("📂 Envie o TR (PDF) — se escaneado, marque OCR abaixo", type=["pdf"])
if not uploaded:
    st.info("Envie o PDF para começar.")
    st.stop()

use_openai_ocr = st.checkbox("Usar OCR remoto via OpenAI (requer OPENAI_API_KEY em Secrets)", value=False)
use_local_ocr = st.checkbox("Permitir OCR local (pytesseract/pypdfium2) se OpenAI não estiver disponível", value=True)

file_bytes = uploaded.read()
file_stream = io.BytesIO(file_bytes)

# If PDF has embedded text, we can extract directly; otherwise attempt OCR routes
with st.spinner("Tentando extrair tabelas com pdfplumber..."):
    rows, full_text = extract_words_table_by_coords(file_stream)
df, full_text_df = rows_to_df_from_coords(rows, full_text)

# If no table found, try OCR via OpenAI then re-extract; else local OCR text
if df.empty or len(df) < 1:
    st.warning("Não foi possível extrair itens automaticamente a partir do PDF atual. Tentando OCR...")
    pdf_searchable_bytes = None
    ocr_text = None
    # 1) Try OpenAI OCR if requested
    if use_openai_ocr:
        st.info("Tentando OCR remoto via OpenAI (isso envia o PDF para a API OpenAI).")
        pdf_searchable_bytes, err = convert_pdf_to_searchable_via_openai(file_bytes)
        if err:
            st.error("OpenAI OCR falhou: " + str(err))
            pdf_searchable_bytes = None
        else:
            st.success("OpenAI OCR retornou um PDF pesquisável (tentando re-extrair).")
            # write to temp stream and re-run extraction
            file_stream = io.BytesIO(pdf_searchable_bytes)
            rows, full_text = extract_words_table_by_coords(file_stream)
            df, full_text_df = rows_to_df_from_coords(rows, full_text)
    # 2) If still empty, try local OCR to get text (if allowed)
    if (df.empty or len(df) < 1) and use_local_ocr:
        st.info("Tentando OCR local (pypdfium2 + pytesseract) — requisição local.")
        ocr_text, err_local = ocr_local_convert_pdf_to_searchable(file_bytes)
        if err_local:
            st.error("OCR local falhou: " + str(err_local))
            ocr_text = None
        else:
            st.success("OCR local obteve texto — vamos tentar extrair linhas por heurística textual.")
            # crude parsing: lines with codes -> feed rows_to_df_from_coords fallback
            candidates = []
            for ln in [l.strip() for l in ocr_text.splitlines() if l.strip()]:
                if re.search(r"\b\d{5,7}\b", ln):
                    candidates.append([ln])
            if candidates:
                df2, _ = rows_to_df_from_coords(candidates, ocr_text)
                if not df2.empty:
                    df = df2
                    full_text_df = ocr_text

# Final check again
if df.empty or len(df) < 1:
    st.error("Não foi possível extrair itens automaticamente. Se o PDF for escaneado, ative OCR local ou forneça OPENAI_API_KEY para OCR remoto.")
    st.subheader("Trecho de texto extraído (amostra):")
    st.code((full_text_df or full_text)[:1000] or "— sem texto extraído —")
    st.stop()

# Display extracted table
st.markdown("### ✅ Tabela extraída (prévia)")
st.write(f"Linhas: {len(df)} — Colunas detectadas: {', '.join(df.columns)}")
st.dataframe(df, width="stretch", height=360)

# Download HTML/Excel and validate with CATMAT
c1, c2, c3 = st.columns([1,1,1])
with c1:
    if st.button("📄 Gerar/baixar HTML (pandas)"):
        html = df.to_html(index=False, escape=True)
        st.download_button("⬇️ Baixar HTML", data=html.encode("utf-8"), file_name="tabela_pandastable.html", mime="text/html")
with c2:
    if st.button("⬇️ Gerar e baixar Excel consolidado"):
        buf = io.BytesIO()
        with pd.ExcelWriter(buf, engine="openpyxl") as writer:
            df.to_excel(writer, sheet_name="Itens", index=False)
        buf.seek(0)
        st.download_button("⬇️ Baixar Excel", data=buf.getvalue(), file_name="tabela_final.xlsx", mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
with c3:
    if st.button("🔎 Validar com planilha oficial (CATMAT)"):
        with st.spinner("Baixando planilha oficial e validando..."):
            try:
                catmat_df, catser_df, meta = download_latest_catalog()
            except Exception as e:
                st.error("Falha ao baixar planilha oficial: " + str(e))
                st.stop()
            st.write(f"Planilha usada: {meta.get('url')} (baixada em {datetime.utcfromtimestamp(meta.get('fetched_at')).isoformat()} UTC)")
            # detect code column heuristically
            code_col = None
            for c in df.columns:
                if c.lower() in ("catmat","catser","codigo","cod","cód"):
                    code_col = c; break
            if not code_col:
                for c in df.columns:
                    sample = df[c].astype(str).head(40).tolist()
                    if any(re.search(r"\b\d{5,7}\b", s) for s in sample):
                        code_col = c; break
            results = []
            for idx, row in df.iterrows():
                code_val = row.get(code_col,"") if code_col else ""
                desc_val = row.get("Descrição","")
                unit_val = row.get("Unidade","")
                match = match_item_with_catalog(code_val, desc_val, unit_val, catmat_df)
                status = "❌ Não encontrado"
                if match.get("found"):
                    sim = match.get("desc_similarity",0.0)
                    unit_ok = match.get("unit_match", False)
                    if unit_ok and sim >= 0.65:
                        status = "✅ OK"
                    elif sim >= 0.45:
                        status = "⚠️ Parcial"
                    else:
                        status = "❌ Divergente"
                results.append({
                    "Item": row.get("Item",""),
                    "Código PDF": code_val,
                    "Descrição PDF": (desc_val[:120]+"...") if len(str(desc_val))>120 else desc_val,
                    "Unid. PDF": unit_val,
                    "Encontrado?": match.get("found", False),
                    "Unid. Oficial": match.get("unit_official",""),
                    "Desc. Oficial (amostra)": (str(match.get("desc_official",""))[:120] + "...") if match.get("desc_official") else "",
                    "SimDesc": float(match.get("desc_similarity",0.0)),
                    "UnitMatch": match.get("unit_match", False),
                    "Status": status
                })
            df_res = pd.DataFrame(results)
            st.dataframe(df_res, width="stretch", height=450)
            # download option
            buf = io.BytesIO()
            with pd.ExcelWriter(buf, engine="openpyxl") as writer:
                df.to_excel(writer, sheet_name="Itens", index=False)
                df_res.to_excel(writer, sheet_name="Validação_CATMAT", index=False)
            buf.seek(0)
            st.download_button("⬇️ Baixar Excel (Itens + Validação CATMAT)", data=buf.getvalue(), file_name="auditoria_catmat.xlsx", mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")

# Optional: generate HTML via ChatGPT
st.markdown("### ✨ (Opcional) Gerar HTML formatado usando ChatGPT")
use_gpt = st.checkbox("Gerar HTML via ChatGPT (opcional)", value=False)
if use_gpt:
    if not OPENAI_SDK_AVAILABLE:
        st.warning("openai SDK não instalado — adicione openai ao requirements.txt")
    else:
        if st.button("🔁 Gerar HTML via ChatGPT"):
            rows_json = df.fillna("").to_dict(orient="records")
            html, err = generate_html_with_gpt(str(rows_json))
            if err:
                st.error("Falha ao gerar HTML via OpenAI: " + str(err))
            else:
                st.components.v1.html(html, height=600, scrolling=True)
                st.download_button("⬇️ Baixar HTML (ChatGPT)", data=html.encode("utf-8"), file_name="tabela_chatgpt.html", mime="text/html")

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
