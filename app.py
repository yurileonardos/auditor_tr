# app.py - Extrator flexível + Consulta CATMAT/CATSER + HTML por grupo
import streamlit as st
import pandas as pd
import pdfplumber
import re
import requests
from io import BytesIO
from datetime import datetime

st.set_page_config(page_title="Auditor TR - Extração Flexível + CATMAT", layout="wide")
st.title("🛡️ Auditor TR — Extração Flexível (colunas variáveis) + Consulta CATMAT")

# -------------------------
# Utilitários
# -------------------------
def clean_number(value):
    if value is None:
        return 0.0
    s = str(value)
    s = s.replace('R$', '').replace('\xa0', ' ').strip()
    s = s.replace('.', '').replace(',', '.')
    s = re.sub(r'[^\d\.-]', '', s)
    try:
        return float(s) if s != '' else 0.0
    except:
        return 0.0

def normalize_text(t):
    if t is None:
        return ""
    return str(t).strip()

# -------------------------
# Consulta GOV (API oficial)
# -------------------------
@st.cache_data(show_spinner=False)
def consultar_item_governo(codigo):
    codigo = re.sub(r'\D','', str(codigo))
    if not codigo:
        return {"status_api":"Inválido","descricao":"","unidade":"-","link":""}
    # Tenta materiais
    try:
        url = f"https://compras.dados.gov.br/materiais/v1/materiais.json?codigo={codigo}"
        r = requests.get(url, timeout=5)
        if r.status_code == 200:
            j = r.json()
            lista = j.get("_embedded", {}).get("materiais", [])
            if lista:
                item = lista[0]
                unidade = item.get("unidade_medida") or item.get("unidade") or "-"
                return {"status_api":"Ativo-Material", "descricao": item.get("descricao",""), "unidade": unidade, "link": f"https://catalogo.compras.gov.br/cnbs-web/busca?cod={codigo}"}
    except:
        pass
    # Tenta serviços
    try:
        url = f"https://compras.dados.gov.br/servicos/v1/servicos.json?codigo={codigo}"
        r = requests.get(url, timeout=5)
        if r.status_code == 200:
            j = r.json()
            lista = j.get("_embedded", {}).get("servicos", [])
            if lista:
                item = lista[0]
                unidade = item.get("unidade") or "UN"
                return {"status_api":"Ativo-Servico", "descricao": item.get("descricao",""), "unidade": unidade, "link": f"https://catalogo.compras.gov.br/cnbs-web/busca?cod={codigo}"}
    except:
        pass
    return {"status_api":"Não Encontrado", "descricao":"", "unidade":"-", "link": f"https://catalogo.compras.gov.br/cnbs-web/busca?cod={codigo}"}

# -------------------------
# Heurísticas para identificar cabeçalho de tabela
# -------------------------
HEADER_KEYWORDS = [
    "item","descr","descricao","unidade","catmat","catser","qtd","quant","quantidade",
    "preco","preço","unit","unitario","total","são paulo","são","sp","rio","recife","manaus","caeté","caete"
]

def guess_header_index(rows):
    """
    Recebe lista de linhas (listas de células). Retorna índice da linha que parece ser cabeçalho.
    """
    for i, row in enumerate(rows[:30]):
        if not any(cell for cell in row):
            continue
        row_text = " ".join([str(x).lower() for x in row if x])
        score = sum(1 for kw in HEADER_KEYWORDS if kw in row_text)
        # heurística: se encontrou 2+ palavras-chave, provável header
        if score >= 2:
            return i
    return None

# -------------------------
# Extração híbrida e flexível
# -------------------------
def extract_tables_and_lines(file_stream):
    pages_text = []
    tabular_rows = []  # lista de linhas (cada linha = lista de células)
    with pdfplumber.open(file_stream) as pdf:
        for page in pdf.pages:
            txt = page.extract_text() or ""
            pages_text.append(txt)
            # tenta extrair tabelas (mais tolerante)
            try:
                tables = page.extract_tables(table_settings={"vertical_strategy":"lines", "horizontal_strategy":"lines"})
                if not tables:
                    tables = page.extract_tables()
            except Exception:
                tables = page.extract_tables()
            if tables:
                for t in tables:
                    for r in t:
                        # normalize None->""
                        row = [("" if c is None else c) for c in r]
                        if any(str(x).strip() for x in row):
                            tabular_rows.append(row)
    full_text = "\n".join(pages_text)
    return tabular_rows, full_text

def rebuild_structured_df(tabular_rows, full_text):
    """
    1) Se houver tabular_rows, tenta identificar o cabeçalho e construir DF mantendo ordem original de colunas.
    2) Se falhar ou parcial, faz fallback para análise textual por linhas (regex) para detectar códigos e grupos.
    """
    if tabular_rows:
        # Tenta identificar header
        header_idx = guess_header_index(tabular_rows)
        if header_idx is None:
            # fallback: assume primeira linha não vazia é header
            for i, r in enumerate(tabular_rows[:5]):
                if any(str(x).strip() for x in r):
                    header_idx = i
                    break

        # se header encontrado, monta DF
        if header_idx is not None and header_idx < len(tabular_rows)-1:
            headers_raw = [normalize_text(c) for c in tabular_rows[header_idx]]
            # Normalize headers to safe unique column names while preserving original string
            headers = []
            header_map = {}  # map normalized->original
            for i, h in enumerate(headers_raw):
                hstr = h if h else f"col{i}"
                # preserve original content for HTML later
                headers.append(hstr)
                header_map[hstr] = hstr

            data_rows = tabular_rows[header_idx+1:]
            # ensure uniform length: pad rows to header length
            processed = []
            for r in data_rows:
                row = [("" if c is None else c) for c in r]
                if len(row) < len(headers):
                    row += [""]*(len(headers)-len(row))
                processed.append(row[:len(headers)])
            try:
                df = pd.DataFrame(processed, columns=headers)
                # Basic cleaning: strip strings
                df = df.applymap(lambda x: normalize_text(x) if isinstance(x, str) else x)
                return df, header_idx
            except Exception:
                pass

    # Fallback textual parsing: busca sequências de itens por pagina/linha
    # Identifica grupos e cria colunas padrão quando possível
    pattern_group = re.compile(r"\bGRUPO\s*\d+", flags=re.IGNORECASE)
    pattern_cat = re.compile(r"\b\d{5,7}\b")
    lines = full_text.splitlines()
    current_group = "Sem Grupo Identificado"
    rows = []
    for i, line in enumerate(lines):
        ln = line.strip()
        if not ln:
            continue
        if pattern_group.search(ln):
            current_group = ln
            continue
        # se linha contém código, tenta extrair
        m = pattern_cat.search(ln)
        if m:
            code = m.group(0)
            # descrição heurística: parte até o código
            desc = ln[:m.start()].strip()
            # buscar preços na linha
            prices = re.findall(r"\d{1,3}(?:[\.\,]\d{3})*[\.,]\d{2}", ln)
            vunit = clean_number(prices[-2]) if len(prices) >= 2 else 0.0
            vtotal = clean_number(prices[-1]) if len(prices) >= 1 else 0.0
            # qtd heurística: primeiro inteiro >0 presente
            q = 0
            ints = re.findall(r"\b\d+\b", ln[:m.start()])
            for it in ints[::-1]:
                if int(it) > 0:
                    q = float(it)
                    break
            rows.append({
                "Grupo": current_group,
                "Item": "",
                "CATMAT": code,
                "Descrição": desc,
                "Unid": "",
                "Qtd Total": q,
                "Preço Unitário": vunit,
                "Preço Total": vtotal,
                "Linha Origem": ln
            })
    if rows:
        return pd.DataFrame(rows), None

    # último recurso: procura por qualquer 5-7 dígitos no texto e cria linhas
    codes = list(set(re.findall(r"\b\d{5,7}\b", full_text)))
    rows = [{"Grupo":"Sem Grupo Identificado","Item":"","CATMAT":c,"Descrição":"","Unid":"","Qtd Total":0,"Preço Unitário":0,"Preço Total":0,"Linha Origem":""} for c in codes]
    return pd.DataFrame(rows), None

# -------------------------
# Geração HTML por grupo (mantendo colunas originais)
# -------------------------
def generate_grouped_html(df, original_headers=None):
    """
    Gera HTML com uma tabela por grupo. original_headers: lista de strings que representam os headers a exibir (mantém ordem).
    """
    html_parts = []
    groups = df['Grupo'].fillna("Sem Grupo Identificado").unique().tolist()
    for g in groups:
        sub = df[df['Grupo'].fillna("Sem Grupo Identificado")==g].copy()
        html_parts.append(f'<h3>{g}</h3>')
        if original_headers is None:
            html = sub.to_html(index=False, escape=True)
        else:
            # reorganiza colunas para original headers intersectantes
            cols = [c for c in original_headers if c in sub.columns]
            if not cols:
                cols = sub.columns.tolist()
            html = sub[cols].to_html(index=False, escape=True)
        html_parts.append(html)
    return "\n".join(html_parts)

# -------------------------
# UI e fluxo principal
# -------------------------
st.markdown("Envie o PDF do Termo de Referência. O sistema tentará extrair a(s) tabela(s) preservando a ordem original das colunas sempre que possível. Depois você poderá consultar os códigos no catálogo do governo e baixar o Excel com duas abas (Itens + CATMAT).")

uploaded = st.file_uploader("Upload PDF do TR", type=["pdf"])
if not uploaded:
    st.info("Envie o PDF para iniciar a extração.")
    st.stop()

with st.spinner("Extraindo tabelas e texto do PDF... (pode demorar alguns segundos)"):
    tabular_rows, full_text = extract_tables_and_lines(uploaded)

with st.spinner("Reconstruindo DataFrame (heurística flexível)..."):
    df_rebuilt, header_index = rebuild_structured_df(tabular_rows, full_text)

if df_rebuilt is None or df_rebuilt.empty:
    st.error("Não foi possível extrair tabelas ou códigos automaticamente. O PDF pode ser uma imagem (scan). Se for scan, use OCR primeiro.")
    st.stop()

# Tentativa de identificar uma coluna de grupo caso não exista
if 'Grupo' not in df_rebuilt.columns:
    # detecta padrões "GRUPO" na coluna de descrição ou linha origem
    if 'Descrição' in df_rebuilt.columns:
        # cria coluna Grupo por proximidade de pattern nas Linha Origem (se houver)
        df_rebuilt['Grupo'] = df_rebuilt.get('Grupo', 'Sem Grupo Identificado')
    else:
        df_rebuilt['Grupo'] = df_rebuilt.get('Grupo', 'Sem Grupo Identificado')

# Exibe resumo
st.markdown("### ✅ Tabela Reconstruída (visualização)")
st.info(f"Linhas extraídas: {len(df_rebuilt)} — Colunas detectadas: {', '.join(df_rebuilt.columns.astype(str).tolist())}")

# Exibir DataFrame (opção A: só mostrar o DF organizado)
st.dataframe(df_rebuilt, use_container_width=True, height=420)

# Botão para gerar HTML por grupo (download)
if st.button("📄 Gerar HTML tabulado por GRUPO (download)"):
    original_headers = df_rebuilt.columns.tolist()
    html_body = generate_grouped_html(df_rebuilt, original_headers=original_headers)
    full_html = f"""<!doctype html>
<html lang="pt-BR"><head><meta charset="utf-8"/><title>Tabela extraída - {datetime.now().date()}</title>
<style>body{{font-family:Arial,Helvetica,sans-serif;padding:16px}}table{{border-collapse:collapse;width:100%;margin-bottom:18px}}th,td{{border:1px solid #bbb;padding:6px 8px;text-align:left}}th{{background:#f1f1f1}}</style>
</head><body>
<h2>Termo de Referência - Tabela extraída</h2>
{html_body}
</body></html>"""
    b = full_html.encode('utf-8')
    st.download_button("⬇️ Baixar HTML tabulado", data=b, file_name="tabela_extraida.html", mime="text/html")

# Botão para executar consulta CATMAT para todos os códigos detectados
if st.button("🔎 Consultar todos os códigos no Compras.gov.br (CATMAT/CATSER)"):
    # Identificar coluna de código: procura por col names que contenham catmat/catser/cod
    code_col = None
    for c in df_rebuilt.columns:
        cl = str(c).lower()
        if 'cat' in cl or 'cod' in cl or re.search(r'\b\d{5,7}\b', cl):
            code_col = c
            break
    # fallback: buscar coluna que contem 5-7 dígitos em seus valores
    if not code_col:
        for c in df_rebuilt.columns:
            sample = df_rebuilt[c].astype(str).head(20).tolist()
            if any(re.search(r'\b\d{5,7}\b', s) for s in sample):
                code_col = c
                break
    if not code_col:
        st.error("Não encontrei uma coluna identificável de códigos CATMAT/CATSER. Verifique a tabela extraída.")
        st.stop()

    st.info(f"Coluna usada como código: '{code_col}'")
    progress = st.progress(0)
    results = []
    uniques = df_rebuilt[code_col].astype(str).fillna("").unique().tolist()
    uniques = [u for u in uniques if re.search(r'\d{5,7}', u)]
    total = len(uniques)
    for idx, code in enumerate(uniques):
        progress.progress((idx+1)/max(1,total))
        gov = consultar_item_governo(code)
        results.append({
            "Código": code,
            "Status API": gov.get("status_api",""),
            "Descrição Oficial": gov.get("descricao",""),
            "Unidade Oficial": gov.get("unidade","-"),
            "Link Gov": gov.get("link","")
        })
    df_catmat = pd.DataFrame(results)
    st.subheader("Resultado da varredura CATMAT/CATSER (amostra)")
    st.dataframe(df_catmat.head(200), use_container_width=True, height=300)

    # Preparar Excel com duas abas: Itens (tabela completa) e CATMAT
    with BytesIO() as buffer:
        with pd.ExcelWriter(buffer, engine="openpyxl") as writer:
            # itens: preserva todas as colunas detectadas
            df_rebuilt.to_excel(writer, index=False, sheet_name="Itens")
            # catmat results
            df_catmat.to_excel(writer, index=False, sheet_name="CATMAT")
            writer.save()
        buffer.seek(0)
        st.download_button("⬇️ Baixar Resultado (Excel com abas Itens + CATMAT)", data=buffer.getvalue(), file_name="auditoria_tr_resultados.xlsx", mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")

    st.success("Consulta finalizada. Baixe o arquivo Excel para ver todas as colunas e resultados CATMAT.")
