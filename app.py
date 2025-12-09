# app.py
"""
Auditor TR - Extração fiel + Consulta CATMAT/CATSER
Versão compatível com Streamlit Cloud (streamlit.app)
- Corrige problemas comuns de deploy (download Excel em buffer, sem lxml, width="stretch")
- Extração por pdfplumber, heurística de cabeçalho e grupos
- Consulta CATMAT/CATSER via API pública compras.dados.gov.br
"""

import streamlit as st
import pandas as pd
import pdfplumber
import re
import requests
import io
from datetime import datetime
from bs4 import BeautifulSoup

# --- Config ---
st.set_page_config(page_title="Auditor TR - Extração + CATMAT", layout="wide")
st.title("🛡️ Auditor TR — Extração fiel + Consulta CATMAT/CATSER")

# ---------- Utilitários ----------
def clean_number(value):
    """Converte textos com formato BR para float. Ex: '1.234,56' -> 1234.56"""
    if value is None:
        return 0.0
    s = str(value).strip()
    # Remove currency R$
    s = s.replace("R$", "").replace("\xa0", " ")
    # Remove thousands dots and convert comma to dot
    s = s.replace(".", "").replace(",", ".")
    # Keep only digits, dot and minus
    s = re.sub(r"[^\d\.\-]", "", s)
    try:
        return float(s) if s != "" else 0.0
    except:
        return 0.0

def normalize_text(v):
    if pd.isna(v):
        return ""
    return str(v).strip()

# ---------- Consulta CATMAT / CATSER (API pública) ----------
@st.cache_data(show_spinner=False)
def consultar_item_cat(codigo):
    """
    Consulta API pública compras.dados.gov.br para materiais e serviços.
    Retorna dict: status_api, tipo, codigo, descricao, unidade, link
    """
    code = re.sub(r"\D", "", str(codigo))
    if not code:
        return {"status_api": "Inválido", "codigo": codigo, "descricao": "", "unidade": "-", "link": ""}

    # Tenta materiais
    try:
        url_mat = f"https://compras.dados.gov.br/materiais/v1/materiais.json?codigo={code}"
        resp = requests.get(url_mat, timeout=6)
        if resp.status_code == 200:
            j = resp.json()
            mats = j.get("_embedded", {}).get("materiais", [])
            if mats:
                item = mats[0]
                unidade = item.get("unidade_medida") or item.get("unidade") or "-"
                return {
                    "status_api": "Encontrado-Mat",
                    "tipo": "Material",
                    "codigo": code,
                    "descricao": item.get("descricao", "").strip(),
                    "unidade": unidade,
                    "link": f"https://catalogo.compras.gov.br/cnbs-web/busca?cod={code}"
                }
    except Exception:
        pass

    # Tenta serviços
    try:
        url_srv = f"https://compras.dados.gov.br/servicos/v1/servicos.json?codigo={code}"
        resp = requests.get(url_srv, timeout=6)
        if resp.status_code == 200:
            j = resp.json()
            srvs = j.get("_embedded", {}).get("servicos", [])
            if srvs:
                item = srvs[0]
                unidade = item.get("unidade") or "UN"
                return {
                    "status_api": "Encontrado-Serv",
                    "tipo": "Servico",
                    "codigo": code,
                    "descricao": item.get("descricao", "").strip(),
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

# ---------- Extração e reconstrução da tabela ----------
HEADER_KEYWORDS = [
    "item","descr","descricao","unidade","catmat","catser","qtd","quant","quantidade",
    "preco","preço","unit","unitario","total","são paulo","sp","rio","recife","manaus","caeté","caete"
]

def guess_header_index(rows):
    """Tenta deduzir o índice da linha de cabeçalho entre as primeiras linhas extraídas."""
    for i, row in enumerate(rows[:30]):
        if not any(cell for cell in row):
            continue
        row_text = " ".join([str(x).lower() for x in row if x])
        score = sum(1 for kw in HEADER_KEYWORDS if kw in row_text)
        if score >= 2:
            return i
    return None

def extract_tables_from_pdf(file_stream):
    """
    Usa pdfplumber para extrair linhas tabulares (lista de listas) e texto completo.
    """
    tabular_rows = []
    all_text = ""
    try:
        with pdfplumber.open(file_stream) as pdf:
            for page in pdf.pages:
                text = page.extract_text() or ""
                all_text += text + "\n"
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
    except Exception as e:
        st.warning("Erro ao abrir PDF com pdfplumber: " + str(e))
    return tabular_rows, all_text

def rebuild_dataframe(tabular_rows, full_text):
    """
    Reconstrói um DataFrame padronizado com colunas:
    Grupo, Item, Descrição, Unidade, CATMAT, QTD, São Paulo, Rio de Janeiro, Caeté, Manaus, Recife, Preço Unitário (R$), Preço Total (R$)
    """
    std_cols = ["Grupo","Item","Descrição","Unidade","CATMAT","QTD","São Paulo","Rio de Janeiro","Caeté","Manaus","Recife","Preço Unitário (R$)","Preço Total (R$)"]
    if tabular_rows:
        header_idx = guess_header_index(tabular_rows)
        if header_idx is None:
            # fallback: assume header is first non-empty row among top 6
            for i, r in enumerate(tabular_rows[:6]):
                if any(str(x).strip() for x in r):
                    header_idx = i
                    break
        if header_idx is not None and header_idx < len(tabular_rows)-1:
            header_row = [normalize_text(c) for c in tabular_rows[header_idx]]
            headers = []
            for i,h in enumerate(header_row):
                name = h if h else f"col{i}"
                headers.append(name)
            data_rows = tabular_rows[header_idx+1:]
            processed = []
            for r in data_rows:
                row = [("" if c is None else c) for c in r]
                if len(row) < len(headers):
                    row += [""]*(len(headers)-len(row))
                processed.append(row[:len(headers)])
            df = pd.DataFrame(processed, columns=headers)
            # normalize strings
            df = df.applymap(lambda x: normalize_text(x) if isinstance(x, str) else x)
            # rename probable columns to standard names
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
            # ensure all std cols present
            for c in std_cols:
                if c not in df.columns:
                    df[c] = ""
            # attempt simple group detection from full_text
            current_group = "SEM GRUPO"
            # if there are explicit "GRUPO" headings in the text, take the first one as current_group
            match = re.search(r"(GRUPO\s*\d+.*?)\n", full_text, flags=re.IGNORECASE)
            if match:
                current_group = match.group(1).strip()
            df["Grupo"] = current_group
            # reorder to standard
            df = df[["Grupo","Item","Descrição","Unidade","CATMAT","QTD","São Paulo","Rio de Janeiro","Caeté","Manaus","Recife","Preço Unitário (R$)","Preço Total (R$)"]]
            # numeric conversion
            for c in ["QTD","Preço Unitário (R$)","Preço Total (R$)"]:
                if c in df.columns:
                    df[c] = df[c].apply(clean_number)
            return df
    # Fallback: scan for numeric codes in text and create rows
    rows = []
    codes = list(dict.fromkeys(re.findall(r"\b\d{5,7}\b", full_text)))
    for c in codes:
        rows.append({"Grupo":"SEM GRUPO","Item":"","Descrição":"","Unidade":"","CATMAT":c,"QTD":0,"São Paulo":0,"Rio de Janeiro":0,"Caeté":0,"Manaus":0,"Recife":0,"Preço Unitário (R$)":0,"Preço Total (R$)":0})
    if rows:
        return pd.DataFrame(rows)
    # empty
    return pd.DataFrame(columns=std_cols)

# ---------- HTML generation ----------
def generate_grouped_html(df):
    """
    Gera HTML por grupo contendo tabelas (pandas.to_html).
    """
    html_parts = []
    if "Grupo" in df.columns:
        groups = df["Grupo"].fillna("SEM GRUPO").unique().tolist()
    else:
        groups = ["Todos os Itens"]
    for g in groups:
        sub = df[df["Grupo"].fillna("SEM GRUPO")==g].copy()
        html_parts.append(f"<h3>{g}</h3>")
        html_parts.append(sub.to_html(index=False, escape=True))
    return "\n".join(html_parts)

# ---------- UI ----------
st.sidebar.header("Configurações")
st.sidebar.markdown("Configurações rápidas do app")

uploaded = st.file_uploader("📂 Envie o TR (PDF)", type=["pdf"])
if not uploaded:
    st.info("Envie o PDF do Termo de Referência para iniciar a extração.")
    st.stop()

with st.spinner("Extraindo tabelas do PDF (isso pode levar alguns segundos)..."):
    rows, full_text = extract_tables_from_pdf(uploaded)

with st.spinner("Reconstruindo DataFrame..."):
    df = rebuild_dataframe(rows, full_text)

if df.empty:
    st.error("Não foi possível extrair itens do PDF automaticamente. Se for um PDF escaneado (imagem), faça OCR antes de enviar.")
    st.stop()

# Exibe resumo
st.markdown("### ✅ Tabela extraída (prévia)")
st.write(f"Linhas: {len(df)} — Colunas: {', '.join(df.columns)}")
st.dataframe(df, width="stretch", height=360)

# download HTML e Excel
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
        buffer = io.BytesIO()
        with pd.ExcelWriter(buffer, engine="openpyxl") as writer:
            df.to_excel(writer, sheet_name="Itens", index=False)
        buffer.seek(0)
        st.download_button("⬇️ Baixar Excel", data=buffer.getvalue(), file_name="tabela_final.xlsx", mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")

with c3:
    if st.button("🔎 Consultar CATMAT para códigos detectados"):
        st.info("Executando varredura — aguarde (cada consulta usa a API pública).")
        # localiza coluna de código
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
            # salva Excel com duas abas: Itens + CATMAT
            buffer = io.BytesIO()
            with pd.ExcelWriter(buffer, engine="openpyxl") as writer:
                df.to_excel(writer, sheet_name="Itens", index=False)
                df_cat.to_excel(writer, sheet_name="CATMAT", index=False)
            buffer.seek(0)
            st.download_button("⬇️ Baixar Excel (Itens + CATMAT)", data=buffer.getvalue(), file_name="auditoria_com_catmat.xlsx", mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
            st.markdown("#### Resultados da consulta CATMAT")
            st.dataframe(df_cat, width="stretch", height=300)

# Checagens matemáticas rápidas
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

# Footer com data/hora
st.markdown("---")
st.caption(f"Gerado em {datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S')} UTC")
