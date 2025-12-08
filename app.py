import streamlit as st
import pandas as pd
import pdfplumber
import requests
import re
from difflib import SequenceMatcher

# -----------------------------------------------------------------------
# CONFIGURAÇÃO INICIAL
# -----------------------------------------------------------------------
st.set_page_config(page_title="Auditor TR - Validação Completa", layout="wide")
st.title("🛡️ Auditor de TR: Validação Jurídica e Técnica + Consulta CATMAT")

# -----------------------------------------------------------------------
# FUNÇÕES AUXILIARES
# -----------------------------------------------------------------------
def clean_number(value):
    if pd.isna(value): return 0.0
    text = str(value).upper().replace('R$', '').replace(' ', '').strip()
    text = text.replace('.', '').replace(',', '.')
    clean_str = re.sub(r'[^\d\.]', '', text)
    try:
        return float(clean_str)
    except:
        return 0.0

def normalize_text(text):
    return str(text).strip().upper() if text else ""

def find_evidence_in_text(text, keywords_dict):
    results = {}
    text_lower = text.lower()

    for topic, terms in keywords_dict.items():
        found = False
        snippet = "Não identificado"

        for term in terms:
            if term in text_lower:
                found = True
                start = max(0, text_lower.find(term) - 50)
                end = min(len(text), start + 300)
                snippet = "..." + text[start:end].replace("\n", " ") + "..."
                break

        results[topic] = {"found": found, "evidence": snippet}

    return results

# -----------------------------------------------------------------------
# CONSULTA CATMAT / CATSER (API OFICIAL)
# -----------------------------------------------------------------------
@st.cache_data(show_spinner=False)
def consultar_item_governo(codigo):
    codigo = re.sub(r'\D', '', str(codigo))

    if not codigo:
        return {"status_api": "Inválido"}

    # 1) Tenta como MATERIAL
    try:
        url = f"https://compras.dados.gov.br/materiais/v1/materiais.json?codigo={codigo}"
        resp = requests.get(url, timeout=2)

        if resp.status_code == 200:
            data = resp.json()
            lista = data.get("_embedded", {}).get("materiais", [])

            if lista:
                item = lista[0]
                unidade = item.get("unidade_medida", "")
                return {
                    "tipo": "Material",
                    "status_api": "Ativo",
                    "codigo": codigo,
                    "descricao": item.get("descricao", ""),
                    "unidade": unidade,
                    "link": f"https://catalogo.compras.gov.br/cnbs-web/busca?cod={codigo}"
                }
    except:
        pass

    # 2) Tenta como SERVIÇO
    try:
        url = f"https://compras.dados.gov.br/servicos/v1/servicos.json?codigo={codigo}"
        resp = requests.get(url, timeout=2)

        if resp.status_code == 200:
            data = resp.json()
            lista = data.get("_embedded", {}).get("servicos", [])

            if lista:
                item = lista[0]
                return {
                    "tipo": "Serviço",
                    "status_api": "Ativo",
                    "codigo": codigo,
                    "descricao": item.get("descricao", ""),
                    "unidade": "UN",
                    "link": f"https://catalogo.compras.gov.br/cnbs-web/busca?cod={codigo}"
                }
    except:
        pass

    return {
        "status_api": "Não Encontrado",
        "codigo": codigo,
        "descricao": "",
        "unidade": "-",
        "link": f"https://catalogo.compras.gov.br/cnbs-web/busca?cod={codigo}"
    }

# -----------------------------------------------------------------------
# EXTRAÇÃO AVANÇADA DA TABELA DO PDF
# -----------------------------------------------------------------------
def identify_columns_dynamic(row):
    mapping = {}
    for i, txt in enumerate(row):
        txt_clean = str(txt).lower()

        if "item" in txt_clean:
            mapping['item'] = i
        elif "cód" in txt_clean or "catmat" in txt_clean:
            mapping['cod'] = i
        elif "desc" in txt_clean or "especif" in txt_clean:
            mapping['desc'] = i
        elif "unid" in txt_clean:
            mapping['unid'] = i
        elif "qtd" in txt_clean:
            mapping['qtd'] = i
        elif "unit" in txt_clean:
            mapping['unit'] = i
        elif "total" in txt_clean:
            mapping['total'] = i

    return mapping

def extract_advanced_structure(file):
    all_rows = []
    full_text = ""

    with pdfplumber.open(file) as pdf:
        for page in pdf.pages:
            full_text += page.extract_text() or ""

            tables = page.extract_tables({
                "vertical_strategy": "lines",
                "horizontal_strategy": "lines"
            }) or page.extract_tables()

            for table in tables:
                for row in table:
                    clean_row = [c if c else "" for c in row]
                    if any(x.strip() for x in clean_row):
                        all_rows.append(clean_row)

    if not all_rows:
        return pd.DataFrame(), full_text

    # Localizar cabeçalho
    header_idx = -1
    col_map = {}

    for i, row in enumerate(all_rows[:20]):
        temp_map = identify_columns_dynamic(row)
        if 'desc' in temp_map and 'qtd' in temp_map:
            header_idx = i
            col_map = temp_map
            break

    if header_idx == -1:
        return pd.DataFrame(), full_text

    structured = []
    current_group = "Grupo"

    header_ref = [str(x).strip().lower() for x in all_rows[header_idx]]

    for row in all_rows[header_idx+1:]:

        # Detecta repetição do cabeçalho
        rc = [str(x).strip().lower() for x in row]
        if sum(1 for a, b in zip(rc, header_ref) if a == b) > len(header_ref)/2:
            continue

        row_str = " ".join(rc).upper()

        # Grupo/Lote
        if ("GRUPO" in row_str or "LOTE" in row_str) and len(row_str) < 80:
            current_group = row_str
            continue

        try:
            cod = row[col_map.get("cod", -1)] if col_map.get("cod") is not None else ""
            desc = row[col_map.get("desc", -1)]
            unid = row[col_map.get("unid", -1)]
            qtd = clean_number(row[col_map.get("qtd", -1)])
            vunit = clean_number(row[col_map.get("unit", -1)])
            total_pdf = clean_number(row[col_map.get("total", -1)])
        except:
            continue

        if not desc and not cod:
            continue

        if "TOTAL" in str(desc).upper() and len(str(desc)) < 30:
            continue

        structured.append({
            "Grupo": current_group,
            "Código": str(cod).strip(),
            "Descrição": str(desc).strip(),
            "Unid PDF": str(unid).strip(),
            "Qtd": qtd,
            "V. Unit": vunit,
            "Total PDF": total_pdf
        })

    return pd.DataFrame(structured), full_text

# -----------------------------------------------------------------------
# INTERFACE
# -----------------------------------------------------------------------
uploaded_file = st.file_uploader("📂 Envie o PDF do TR", type=["pdf"])

if uploaded_file:

    with st.spinner("Extraindo tabela e texto do PDF..."):
        df_itens, texto_total = extract_advanced_structure(uploaded_file)

    st.markdown("---")
    st.header("1) Validação Textual do TR")

    keywords = {
        "Justificativa Agrupamento": ["agrupamento", "parcelamento", "econômica"],
        "Locais de Entrega": ["local de entrega", "serão entregues"],
        "Garantia": ["garantia", "assistência"],
        "Pagamento": ["pagamento", "nota fiscal"]
    }

    resultados = find_evidence_in_text(texto_total, keywords)

    c1, c2 = st.columns(2)
    for i, (tema, dados) in enumerate(resultados.items()):
        col = c1 if i % 2 == 0 else c2
        with col:
            if dados["found"]:
                st.success(f"✔ {tema}")
                with st.expander("Trecho encontrado"):
                    st.info(dados["evidence"])
            else:
                st.error(f"❌ {tema}")

    st.markdown("---")
    st.header("2) Auditoria de Itens + CATMAT")

    if st.button("🔍 Executar Auditoria Completa"):
        audit = []
        progress = st.progress(0)

        for i, row in df_itens.iterrows():
            progress.progress((i+1)/len(df_itens))

            codigo = re.sub(r'\D', '', str(row["Código"]))
            qtd = row["Qtd"]
            unit = row["V. Unit"]
            total_pdf = row["Total PDF"]
            total_calc = qtd * unit

            diff = abs(total_calc - total_pdf)
            math_status = "OK" if diff < 0.05 else "Erro"

            cat = consultar_item_governo(codigo)

            unidade_pdf = normalize_text(row["Unid PDF"])
            unidade_gov = normalize_text(cat["unidade"])

            # Comparação técnica de unidade
            if cat["status_api"] != "Ativo":
                tec_status = "Código Inexistente"
            elif unidade_gov and unidade_pdf not in unidade_gov:
                tec_status = "Unidade Divergente"
            else:
                tec_status = "OK"

            audit.append({
                "Grupo": row["Grupo"],
                "Código": codigo,
                "Descrição PDF": row["Descrição"][:60] + "...",
                "Unid PDF": unidade_pdf,
                "Unid Gov": unidade_gov,
                "Qtd": qtd,
                "Unitário": unit,
                "Total Calc": total_calc,
                "Total PDF": total_pdf,
                "Status Matemático": math_status,
                "Status Técnico": tec_status,
                "Link Gov": cat["link"]
            })

        df_final = pd.DataFrame(audit)

        st.subheader("📊 Resultado da Auditoria")
        st.dataframe(
            df_final,
            column_config={
                "Link Gov": st.column_config.LinkColumn("Catálogo", display_text="Abrir"),
            },
            use_container_width=True,
            height=600
        )

        # DOWNLOAD
        st.download_button(
            "💾 Baixar Resultado em Excel",
            df_final.to_excel(index=False, engine="openpyxl"),
            file_name="auditoria_catmat.xlsx"
        )
