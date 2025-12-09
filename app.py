# app.py
import streamlit as st
import pandas as pd
import pdfplumber
import re
import requests
from io import BytesIO
from datetime import datetime
from bs4 import BeautifulSoup

st.set_page_config(page_title="Auditor TR - Extração + CATMAT", layout="wide")
st.title("🛡️ Auditor TR — Extração fiel + Consulta CATMAT/CATSER")

# ---------- UTILITÁRIOS ----------
def clean_number(value):
    if value is None: return 0.0
    s = str(value).strip()
    s = s.replace("R$", "").replace("\xa0", " ")
    # remove thousand separators and convert comma decimal to dot
    s = s.replace(".", "").replace(",", ".")
    s = re.sub(r"[^\d\.\-]", "", s)
    try:
        return float(s) if s != "" else 0.0
    except:
        return 0.0

def normalize_text(v):
    if v is None: return ""
    return str(v).strip()

# ---------- CONSULTA CATMAT / CATSER (API pública) ----------
# This function is the one you can swap for an internal API or other proxy.
@st.cache_data(show_spinner=False)
def consultar_item_cat(codigo):
    """
    Consulta nas APIs públicas de compras.dados.gov.br (materiais e serviços).
    Retorna dict: {status_api, tipo, codigo, descricao, unidade, link}
    """
    code = re.sub(r"\D", "", str(codigo))
    if not code:
        return {"status_api": "Inválido", "codigo": codigo, "descricao": "", "unidade": "-", "link": ""}

    # Tenta materiais
    try:
        url_mat = f"https://compras.dados.gov.br/materiais/v1/materiais.json?codigo={code}"
        r = requests.get(url_mat, timeout=6)
        if r.status_code == 200:
            j = r.json()
            mats = j.get("_embedded", {}).get("materiais", [])
            if mats:
                item = mats[0]
                unidade = item.get("unidade_medida") or item.get("unidade") or "-"
                return {
                    "status_api": "Encontrado-Mat",
                    "tipo": "Material",
                    "codigo": code,
                    "descricao": item.get("descricao", ""),
                    "unidade": unidade,
                    "link": f"https://catalogo.compras.gov.br/cnbs-web/busca?cod={code}"
                }
    except Exception:
        pass

    # Tenta servicos
    try:
        url_srv = f"https://compras.dados.gov.br/servicos/v1/servicos.json?codigo={code}"
        r = requests.get(url_srv, timeout=6)
        if r.status_code == 200:
            j = r.json()
            srvs = j.get("_embedded", {}).get("servicos", [])
            if srvs:
                item = srvs[0]
                unidade = item.get("unidade") or "UN"
                return {
                    "status_api": "Encontrado-Serv",
                    "tipo": "Servico",
                    "codigo": code,
                    "descricao": item.get("descricao", ""),
                    "unidade": unidade,
                    "link": f"https://catalogo.compras.gov.br/cnbs-web/busca?cod={code}"
                }
    except Exception:
        pass

    return {
        "status_api": "NaoEncontrado",
        "tipo": None,
        "codigo": code,
        "descricao": "",
        "unidade": "-",
        "link": f"https://catalogo.compras.gov.br/cnbs-web/busca?cod={code}"
    }

# ---------- EXTRAÇÃO / RECONSTRUÇÃO ----------
HEADER_KEYWORDS = [
    "item","descr","descricao","unidade","catmat","catser","qtd","quant","quantidade",
    "preco","preço","unit","unitario","total","são paulo","sp","rio","recife","manaus","caeté","caete"
]

def guess_header_index(rows):
    for i, row in enumerate(rows[:30]):
        if not any(cell for cell in row):
            continue
        row_text = " ".join([str(x).lower() for x in row if x])
        score = sum(1 for kw in HEADER_KEYWORDS if kw in row_text)
        if score >= 2:
            return i
    return None

def extract_tables_from_pdf(file_stream):
    """Extrai linhas tabulares usando pdfplumber e retorna lista de linhas (cada linha = lista de células) e texto completo"""
    tabular_rows = []
    all_text = ""
    with pdfplumber.open(file_stream) as pdf:
        for page in pdf.pages:
            text = page.extract_text() or ""
            all_text += text + "\n"
            # tenta extrair tabelas com estratégia por linhas
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
                            tabular_rows.append(row)
    return tabular_rows, all_text

def rebuild_dataframe(tabular_rows, full_text):
    """Reconstrói um DataFrame com colunas padrão, tentando manter a ordem do cabeçalho original"""
    if tabular_rows:
        header_idx = guess_header_index(tabular_rows)
        if header_idx is None:
            for i, r in enumerate(tabular_rows[:6]):
                if any(str(x).strip() for x in r):
                    header_idx = i
                    break
        if header_idx is not None and header_idx < len(tabular_rows)-1:
            header_row = [normalize_text(c) for c in tabular_rows[header_idx]]
            # create safe header names
            headers = []
            for i,h in enumerate(header_row):
                name = h if h else f"col{i}"
                headers.append(name)
            data_rows = tabular_rows[header_idx+1:]
            # pad/truncate rows
            processed = []
            for r in data_rows:
                row = [("" if c is None else c) for c in r]
                if len(row) < len(headers):
                    row += [""]*(len(headers)-len(row))
                processed.append(row[:len(headers)])
            df = pd.DataFrame(processed, columns=headers)
            df = df.applymap(lambda x: normalize_text(x) if isinstance(x, str) else x)
            # Try to normalize common column names
            # Map likely columns to standard names
            ren = {}
            for c in df.columns:
                lc = c.lower()
                if "descr" in lc or "espec" in lc:
                    ren[c] = "Descrição"
                elif "cat" in lc or "cod" in lc:
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
                elif "preço unit" in lc or "preco unit" in lc or ("unit" in lc and "total" not in lc):
                    ren[c] = "Preço Unitário (R$)"
                elif "total" in lc and ("preço" in lc or "preco" in lc) or ("valor total" in lc):
                    ren[c] = "Preço Total (R$)"
            df = df.rename(columns=ren)
            # Ensure standard columns exist
            std_cols = ["Grupo","Item","Descrição","Unidade","CATMAT","QTD","São Paulo","Rio de Janeiro","Caeté","Manaus","Recife","Preço Unitário (R$)","Preço Total (R$)"]
            for c in std_cols:
                if c not in df.columns:
                    df[c] = ""
            # Try detecting group titles in surrounding text and filling Grupo:
            # Simple heuristic: if a row has all empty numeric columns and long text like "GRUPO X", mark it and propagate
            groups = []
            current_group = "SEM GRUPO"
            # try scan full_text for "GRUPO" headings and basic positions - fallback: use "SEM GRUPO"
            if "GRUPO" in full_text.upper():
                # find lines that contain GRUPO
                lines = full_text.splitlines()
                grp_positions = []
                for ln in lines:
                    if re.search(r"\bGRUPO\b", ln, re.IGNORECASE):
                        grp_positions.append(ln.strip())
                # choose first three if exist
                if grp_positions:
                    current_group = grp_positions[0]
            df["Grupo"] = current_group
            # Reorder to std_cols with Grupo first
            cols_order = ["Grupo","Item","Descrição","Unidade","CATMAT","QTD","São Paulo","Rio de Janeiro","Caeté","Manaus","Recife","Preço Unitário (R$)","Preço Total (R$)"]
            df = df[cols_order]
            # Convert numeric columns where possible
            for c in ["QTD","Preço Unitário (R$)","Preço Total (R$)"]:
                if c in df.columns:
                    df[c] = df[c].apply(clean_number)
            return df
    # Fallback: try regex scan for codes in text
    rows = []
    codes = list(set(re.findall(r"\b\d{5,7}\b", full_text)))
    for c in codes:
        rows.append({"Grupo":"SEM GRUPO","Item":"","Descrição":"","Unidade":"","CATMAT":c,"QTD":0,"São Paulo":0,"Rio de Janeiro":0,"Caeté":0,"Manaus":0,"Recife":0,"Preço Unitário (R$)":0,"Preço Total (R$)":0})
    if rows:
        return pd.DataFrame(rows)
    return pd.DataFrame(columns=["Grupo","Item","Descrição","Unidade","CATMAT","QTD","São Paulo","Rio de Janeiro","Caeté","Manaus","Recife","Preço Unitário (R$)","Preço Total (R$)"])

# ---------- HTML GENERATION ----------
def generate_grouped_html(df):
    html_parts = []
    # if there's Grupo column, group by it
    if "Grupo" in df.columns:
        groups = df["Grupo"].fillna("SEM GRUPO").unique().tolist()
    else:
        groups = ["Todos os Itens"]
    for g in groups:
        sub = df[df["Grupo"].fillna("SEM GRUPO")==g].copy()
        html_parts.append(f"<h3>{g}</h3>")
        # use pandas to_html for table body
        html = sub.to_html(index=False, classes="table", escape=True)
        html_parts.append(html)
    return "\n".join(html_parts)

# ---------- STREAMLIT UI ----------
st.sidebar.header("Configurações")
run_catmat_via_api = st.sidebar.selectbox("Consulta CATMAT via", ["API pública (compras.dados.gov.br)"], index=0)
uploaded = st.file_uploader("📂 Envie o TR (PDF)", type=["pdf"])

if not uploaded:
    st.info("Envie o PDF do Termo de Referência para iniciar.")
    st.stop()

with st.spinner("Extraindo tabelas do PDF..."):
    rows, full_text = extract_tables_from_pdf(uploaded)

with st.spinner("Reconstruindo DataFrame..."):
    df = rebuild_dataframe(rows, full_text)

if df.empty:
    st.error("Não foi possível extrair itens do PDF automaticamente. Se for um PDF escaneado (imagem), ative OCR externamente e reenvie.")
    st.stop()

# Show summary and table
st.markdown("### ✅ Tabela extraída (prévia)")
st.write(f"Linhas: {len(df)} — Colunas: {', '.join(df.columns)}")
st.dataframe(df, use_container_width=True, height=360)

# Buttons: generate HTML, download Excel, run CATMAT sweep
c1, c2, c3 = st.columns([1,1,1])
with c1:
    if st.button("📄 Gerar/baixar HTML tabulado"):
        html_body = generate_grouped_html(df)
        full_html = f"""<!doctype html><html lang='pt-BR'><head><meta charset='utf-8'><title>TR - Tabela</title>
        <style>body{{font-family:Arial}}table{{border-collapse:collapse;width:100%}}th,td{{border:1px solid #bbb;padding:6px}}th{{background:#eee}}</style></head><body>
        <h2>Termo de Referência — Itens (extraído)</h2>{html_body}</body></html>"""
        st.download_button("⬇️ Baixar HTML", data=full_html.encode("utf-8"), file_name="tabela_final.html", mime="text/html")

with c2:
    if st.button("⬇️ Gerar e baixar Excel consolidado"):
        with BytesIO() as buf:
            with pd.ExcelWriter(buf, engine="openpyxl") as writer:
                df.to_excel(writer, sheet_name="Itens", index=False)
            buf.seek(0)
            st.download_button("⬇️ Baixar Excel", data=buf.getvalue(), file_name="tabela_final.xlsx", mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")

with c3:
    if st.button("🔎 Consultar CATMAT para códigos detectados"):
        st.info("Executando varredura — aguarde (cada consulta usa a API pública).")
        # find code column: try CATMAT, then any column with numeric codes
        code_col = None
        for c in df.columns:
            if c.lower() in ("catmat","catser","codigo","cod","cód"):
                code_col = c; break
        if not code_col:
            # try search in column values
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
            # join results back to df on CATMAT where possible
            # Provide combined Excel with two sheets
            with BytesIO() as buf:
                with pd.ExcelWriter(buf, engine="openpyxl") as writer:
                    df.to_excel(writer, sheet_name="Itens", index=False)
                    df_cat.to_excel(writer, sheet_name="CATMAT", index=False)
                buf.seek(0)
                st.download_button("⬇️ Baixar Excel (Itens + CATMAT)", data=buf.getvalue(), file_name="auditoria_com_catmat.xlsx", mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
            st.dataframe(df_cat, use_container_width=True, height=300)

# Additional checks (math and simple flags)
st.markdown("### ℹ️ Verificações rápidas")
if "QTD" in df.columns and "Preço Unitário (R$)" in df.columns and "Preço Total (R$)" in df.columns:
    df_check = df.copy()
    df_check["Total Calculado"] = df_check["QTD"] * df_check["Preço Unitário (R$)"]
    df_check["Diff"] = (df_check["Total Calculado"] - df_check["Preço Total (R$)"]).abs()
    df_check["Status Math"] = df_check["Diff"].apply(lambda d: "OK" if d <= 0.1 else "DIVERGENTE")
    problemas = df_check[df_check["Status Math"]!="OK"]
    st.write(f"Linhas com divergência matemática: {len(problemas)}")
    if not problemas.empty:
        st.dataframe(problemas[["Item","CATMAT","QTD","Preço Unitário (R$)","Preço Total (R$)","Total Calculado","Diff"]], height=260)
else:
    st.info("Colunas QTD e Preço Unitário/Total não detectadas juntas — verificação matemática desativada.")
