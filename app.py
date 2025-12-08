# app.py (versão revisada)
import streamlit as st
import pandas as pd
import pdfplumber
import re
import requests
from io import BytesIO

st.set_page_config(page_title="Auditor TR - Validação Completa (Revisado)", layout="wide")
st.title("🛡️ Auditor TR — Extração robusta + Consulta CATMAT")

# -------------------------
# UTILIDADES
# -------------------------
def clean_number(value):
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return 0.0
    text = str(value).upper().replace('R$', '').replace(' ', '').strip()
    text = text.replace('.', '').replace(',', '.')
    clean_str = re.sub(r'[^\d\.]', '', text)
    try:
        return float(clean_str) if clean_str != "" else 0.0
    except:
        return 0.0

def normalize_text(text):
    return str(text).strip() if text else ""

# -------------------------
# CONSULTA API GOV (CATMAT / CATSER)
# -------------------------
@st.cache_data(show_spinner=False)
def consultar_item_governo(codigo):
    codigo = re.sub(r'\D', '', str(codigo))
    if not codigo:
        return {"status_api": "Inválido", "descricao":"", "unidade": "-", "link": ""}

    # Primeiro: tentar materiais
    try:
        url = f"https://compras.dados.gov.br/materiais/v1/materiais.json?codigo={codigo}"
        r = requests.get(url, timeout=5)
        if r.status_code == 200:
            j = r.json()
            lista = j.get("_embedded", {}).get("materiais", [])
            if lista:
                item = lista[0]
                unidade = item.get("unidade_medida") or item.get("unidade") or "-"
                return {
                    "status_api": "Ativo-Material",
                    "descricao": item.get("descricao",""),
                    "unidade": unidade,
                    "link": f"https://catalogo.compras.gov.br/cnbs-web/busca?cod={codigo}"
                }
    except Exception as e:
        # não interrompe; vamos tentar serviço
        pass

    # Segundo: tentar serviços
    try:
        url = f"https://compras.dados.gov.br/servicos/v1/servicos.json?codigo={codigo}"
        r = requests.get(url, timeout=5)
        if r.status_code == 200:
            j = r.json()
            lista = j.get("_embedded", {}).get("servicos", [])
            if lista:
                item = lista[0]
                return {
                    "status_api": "Ativo-Servico",
                    "descricao": item.get("descricao",""),
                    "unidade": item.get("unidade") or "UN",
                    "link": f"https://catalogo.compras.gov.br/cnbs-web/busca?cod={codigo}"
                }
    except:
        pass

    return {"status_api":"Não Encontrado", "descricao":"", "unidade":"-", "link": f"https://catalogo.compras.gov.br/cnbs-web/busca?cod={codigo}"}

# -------------------------
# EXTRAÇÃO HÍBRIDA (tabelas + texto) COM DETECÇÃO DE GRUPOS E ITENS
# -------------------------
def extract_structured_from_pdf(file_stream):
    """
    Vai tentar extrair:
    - reconhecer linhas de tabela via pdfplumber.extract_tables
    - quando falhar, faz varredura no texto para identificar blocos 'GRUPO' e linhas de item
    - retorna DataFrame com colunas: Grupo, Item, Código, Descrição, Unidade, Qtd, VUnit, VTotal
    """
    pages_text = []
    all_tabular_rows = []
    with pdfplumber.open(file_stream) as pdf:
        for page in pdf.pages:
            text = page.extract_text() or ""
            pages_text.append(text)
            # tenta extrair tabelas (com estratégia por linhas)
            try:
                tables = page.extract_tables(table_settings={"vertical_strategy":"lines", "horizontal_strategy":"lines"})
            except Exception:
                tables = page.extract_tables()
            if tables:
                for t in tables:
                    for r in t:
                        # converte None -> ""
                        row = ["" if c is None else c for c in r]
                        if any(str(x).strip() for x in row):
                            all_tabular_rows.append(row)

    full_text = "\n".join(pages_text)

    # 1) Se extração tabular trouxe linhas com muitos campos (é uma boa), tenta mapear por cabeçalho
    df_from_table = None
    if all_tabular_rows:
        # busca linha de cabeçalho - heurística
        header_idx = None
        for i, row in enumerate(all_tabular_rows[:20]):
            row_lower = " ".join([str(x).lower() for x in row])
            if ("descri" in row_lower and ("qtd" in row_lower or "quant" in row_lower)) or ("catmat" in row_lower or "catser" in row_lower):
                header_idx = i
                break

        if header_idx is not None:
            headers = [str(x).strip() or f"col{i}" for i, x in enumerate(all_tabular_rows[header_idx])]
            data_rows = all_tabular_rows[header_idx+1:]
            df_from_table = pd.DataFrame(data_rows, columns=headers[:len(data_rows[0])])
            # Normalize column names to easier keys
            # We'll still fallback to text parsing later if needed

    # 2) Texto profundo: detectar blocos GRUPO e linhas de itens por padrão (mais robusto)
    # Padrões:
    # - "GRUPO n - ..." -> define group
    # - item lines: começam com número do item (1/2 dígitos) possivelmente seguido de descrição, e no final possuem CATMAT (códigos numericos 5-7 dígitos) e preço unitário e total
    groups = []
    rows_struct = []

    current_group = "Sem Grupo Identificado"

    # quebrar texto por linhas e analisar
    text_lines = []
    for p_text in pages_text:
        for ln in p_text.splitlines():
            ln_stripped = ln.strip()
            if ln_stripped:
                text_lines.append(ln_stripped)

    # detecta grupos e itens
    # regex para detectar "GRUPO" ou "GRUPO X -"
    re_group = re.compile(r"\bGRUPO\s*\d+", flags=re.IGNORECASE)
    # regex para detectar linha com código CATMAT (5-7 dígitos) e preços (valor com vírgula ou ponto)
    re_catmat = re.compile(r"(\d{5,7})")
    # regex para preços (ex: 1.234,56 ou 1234.56)
    re_price = re.compile(r"(\d{1,3}(?:[\.\,]\d{3})*[\.,]\d{2})")

    # Vamos percorrer as linhas, agregando quando necessário
    i = 0
    while i < len(text_lines):
        ln = text_lines[i]

        # atualiza grupo
        if re_group.search(ln):
            current_group = ln.strip()
            i += 1
            continue

        # tenta achar código CATMAT na linha
        cat_match = re_catmat.search(ln)
        if cat_match:
            # tenta captar dados na mesma linha
            codigo = cat_match.group(1)
            # Extrair preços finais (pega últimos 2 ocorrências como unit e total se presentes)
            prices = re_price.findall(ln)
            # extrai quantidades: procurar sequência de números inteiros (por ex '5 3 0 1 1 0' ) - heurística
            qtds = re.findall(r"\b\d+\b", ln)
            # descrição: parte do início até onde aparece a unidade/código; heurística: tudo antes do código encontrado
            start_desc = ln[:cat_match.start()].strip()
            description = start_desc

            # tentativa melhor: se a linha começar com item número (ex: "13 BOROHIDRETO ..."), pega item número
            item_num = ""
            m_item = re.match(r"^(\d{1,3})\b", ln)
            if m_item:
                item_num = m_item.group(1)

            # Para caso a linha seja muito curta (só código), agregamos a linha anterior como descrição
            if len(description) < 5 and i > 0:
                description = text_lines[i-1]

            # tenta inferir unidade do pdf procurando tokens curtos como FR, SC, AM, UN, G, KG
            unit_search = re.search(r"\b(FR|FRASCO|SC|SACO|AM|UN|UNIDADE|G|GR|KG|MG|L|ML|CX|CAIXA)\b", ln, flags=re.IGNORECASE)
            unidade = unit_search.group(1) if unit_search else ""

            # tenta pegar qtd e vunit/vtotal se possível
            v_unit = 0.0
            v_total = 0.0
            qtd_val = 0.0

            # heurística: últimos dois preços identificados -> vunit e vtotal (se existirem)
            if len(prices) >= 2:
                try:
                    v_unit = clean_number(prices[-2])
                    v_total = clean_number(prices[-1])
                except:
                    pass
            elif len(prices) == 1:
                v_unit = clean_number(prices[-1])

            # heurística para qtd: se houver uma sequência de 6 inteiros (ex: 7 4 2 0 1 0) pode ser total e por locais
            if len(qtds) >= 3:
                # pegar o primeiro inteiro razoável >0
                for q in qtds:
                    if int(q) > 0 and len(q) <= 4:
                        qtd_val = float(q)
                        break

            # finalmente registra linha
            rows_struct.append({
                "Grupo": current_group,
                "Item": item_num,
                "Código": codigo,
                "Descrição": description,
                "Unid PDF": unidade,
                "Qtd": qtd_val,
                "V. Unit": v_unit,
                "V. Total PDF": v_total,
                "Linha Origem": ln
            })
            i += 1
            continue

        # se não houver código, pode ser continuação da descrição (concatena com próxima linha)
        # heurística: se a linha começa com letra e a próxima contém código, juntamos
        if i+1 < len(text_lines) and re_catmat.search(text_lines[i+1]):
            # juntar com a próxima e re-testar no próximo loop (não consumir agora)
            i += 1
            continue

        i += 1

    # Se extraiu algo via tabela (df_from_table) podemos tentar enriquecer rows_struct com códigos que aparecem no df
    # Mas dado a variedade de layouts, retornamos rows_struct como DataFrame
    if not rows_struct:
        # fallback: se não identificou nada, tenta procurar ANY 5-7 dígitos no full text e criar linhas simples
        fallback = []
        for m in re.findall(r"\b\d{5,7}\b", full_text):
            fallback.append({"Grupo":"Sem Grupo Identificado","Item":"","Código":m,"Descrição":"","Unid PDF":"","Qtd":0,"V. Unit":0,"V. Total PDF":0,"Linha Origem":""})
        return pd.DataFrame(fallback), full_text

    df = pd.DataFrame(rows_struct)
    # Limpeza básica: remover duplicados por (Código, Descrição)
    df = df.drop_duplicates(subset=["Código","Descrição"]).reset_index(drop=True)

    return df, full_text

# -------------------------
# UI
# -------------------------
st.markdown("Envie o PDF do Termo de Referência (TR). O sistema tentará extrair grupos, itens e códigos CATMAT/CATSER, e fará consulta ao Compras.gov.br.")
uploaded = st.file_uploader("Upload PDF", type=["pdf"])

if uploaded:
    with st.spinner("Extraindo..."):
        df_items, full_text = extract_structured_from_pdf(uploaded)

    if df_items.empty:
        st.error("Não foi possível extrair itens automaticamente. Verifique o PDF (scan/imagem) ou envie o arquivo original.")
    else:
        st.success(f"{len(df_items)} itens/linhas extraídas (heurística).")
        # Mostra tabela HTML formatada (usando to_html para manter formatação)
        st.subheader("Tabela extraída (visualização)")
        # monta colunas na ordem amigável
        display_df = df_items[["Grupo","Item","Código","Descrição","Unid PDF","Qtd","V. Unit","V. Total PDF","Linha Origem"]]
        # Exibe como dataframe normal (interativo)
        st.dataframe(display_df, use_container_width=True, height=350)

        # Também fornece versão HTML tabulada (mais próxima do que você pediu)
        st.markdown("### Versão HTML (para copy/paste)")
        html_table = display_df.to_html(index=False, escape=False)
        st.code(html_table, language='html')

        # Botão para iniciar varredura/consulta GOV
        if st.button("🔎 Consultar CATMAT/CATSER no Governo para todos os códigos"):
            # preparar coluna de resultados
            results = []
            progress = st.progress(0)
            for idx, r in df_items.iterrows():
                progress.progress((idx+1)/len(df_items))
                cod = r.get("Código")
                # chamada de API
                gov = consultar_item_governo(cod)
                # comparação simples de unidade
                unid_pdf = normalize_text(r.get("Unid PDF",""))
                unid_gov = normalize_text(gov.get("unidade","-"))
                # status técnico
                if gov["status_api"].startswith("Ativo"):
                    if unid_gov != "-" and unid_pdf and unid_pdf.upper() not in unid_gov.upper():
                        status_tec = "Unid. Divergente"
                    else:
                        status_tec = "OK"
                elif gov["status_api"] == "Não Encontrado":
                    status_tec = "Não Encontrado"
                else:
                    status_tec = gov["status_api"]

                results.append({
                    "Grupo": r.get("Grupo"),
                    "Item": r.get("Item"),
                    "Código": cod,
                    "Descrição PDF": r.get("Descrição"),
                    "Unid PDF": r.get("Unid PDF"),
                    "Qtd": r.get("Qtd"),
                    "V. Unit": r.get("V. Unit"),
                    "V. Total PDF": r.get("V. Total PDF"),
                    "Status Técnico": status_tec,
                    "Desc. Oficial (Gov)": gov.get("descricao",""),
                    "Unid. Oficial (Gov)": gov.get("unidade","-"),
                    "Link Gov": gov.get("link","")
                })

            df_res = pd.DataFrame(results)
            st.subheader("Resultado da Consulta (tabela final)")
            # Exibe com link ativo: cria coluna de HTML com anchor
            df_display = df_res.copy()
            df_display["Link Gov"] = df_display["Link Gov"].apply(lambda x: f'<a href="{x}" target="_blank">Abrir</a>' if x else "")

            # mostra como dataframe interativo
            st.dataframe(df_res.drop(columns=["Desc. Oficial (Gov)","Unid. Oficial (Gov)"]), use_container_width=True, height=420)

            # mostra html completo (inclui descrição oficial)
            st.markdown("### Tabela Final (HTML com descrições oficiais)")
            html_final = df_display.to_html(index=False, escape=False)
            st.code(html_final, language='html')

            # Preparar excel em memória (corrige TypeError do streamlit)
            with BytesIO() as buffer:
                with pd.ExcelWriter(buffer, engine="openpyxl") as writer:
                    df_res.to_excel(writer, index=False, sheet_name="auditoria")
                    writer.save()
                buffer.seek(0)
                st.download_button(
                    label="⬇️ Baixar relatório (Excel)",
                    data=buffer.getvalue(),
                    file_name="auditoria_catmat.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                )

            st.success("Consulta finalizada. Revise as linhas com Status Técnico diferente de 'OK'.")

