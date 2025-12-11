# Auditor TR — Extração + Validação CATMAT/CATSER + (Opcional) HTML via ChatGPT

## O que faz
- Extrai tabelas de Termos de Referência (PDF) usando pdfplumber (método robusto por coordenadas).
- Baixa a planilha oficial CATMAT/CATSER do gov.br e valida códigos/descrições/unidades.
- Gera tabelas HTML e Excel com resultados e relatórios.
- Opcional: converte os dados em HTML "fiel" usando a API do OpenAI (ChatGPT).

## Como rodar localmente
1. Criar virtualenv:
   ```bash
   python -m venv .venv
   source .venv/bin/activate   # Linux/Mac
   .venv\Scripts\activate      # Windows
