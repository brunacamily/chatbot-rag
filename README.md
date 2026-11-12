# Chatbot RAG local (CPU)

Chatbot que responde perguntas com base nos seus próprios documentos (`.md`, `.txt`, `.pdf`), usando modelos do Hugging Face rodando 100% no seu computador, sem GPU e sem API paga.

**Como funciona (RAG):** os documentos são divididos em trechos, cada trecho vira um vetor (embedding), a pergunta recupera os 3 trechos mais parecidos e um LLM pequeno responde usando só esses trechos.

| Etapa | Modelo / técnica |
| --- | --- |
| Embeddings | `intfloat/multilingual-e5-small` (sentence-transformers) |
| Busca | similaridade de cosseno em numpy, com cache de embeddings |
| Geração | `Qwen/Qwen2.5-0.5B-Instruct` (transformers, CPU, fp32) |
| Avaliação | Hit@1, Hit@k, MRR e tempo por consulta (`eval.py`) |

## Requisitos

- Python 3.10+, cerca de 3 GB de RAM livre e 2 GB de disco para os modelos.
- Testado para rodar em CPU (ex.: Intel i5-10210U, 14 GB de RAM).

## Instalação

```bash
cd ~/.../Chatbot
source .venv/bin/activate
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt
```

## Uso

```bash
python chat.py        # chat no terminal
python app.py         # interface web em http://127.0.0.1:7860
python eval.py        # mede a qualidade da busca
```

Na primeira execução os modelos são baixados (cerca de 1,5 GB no total). Depois disso funciona offline.

Para usar seus próprios documentos, copie arquivos para `data/docs/` e, se quiser medir a busca neles, edite `data/eval.json` (pergunta + arquivo esperado).

## Configuração (variáveis de ambiente)

```bash
LLM_MODEL=Qwen/Qwen2.5-0.5B-Instruct   # modelo de geração
EMBED_MODEL=intfloat/multilingual-e5-small
THREADS=4                              # threads da CPU
MAX_NEW_TOKENS=200                     # tamanho máximo da resposta
```

Modelos maiores (1,5B parâmetros ou mais) tendem a não caber na RAM livre em fp32 e ficam muito lentos em CPU.

## Resultados

Preencha com os números que o seu `python eval.py` mostrar (arquivo `results/eval.json`):

| Métrica | Valor |
| --- | --- |
| Hit@1 | |
| Hit@3 | |
| MRR | |
| ms por consulta | |
| s por resposta (`--generate 3`) | |

## Ideias para evoluir

- Trocar o modelo de embeddings e comparar o Hit@k no `eval.py`.
- Adicionar um reranker ou ajustar o tamanho dos trechos e comparar os resultados.
- Fazer fine-tuning leve (LoRA) de um modelo pequeno e publicá-lo no Hugging Face.
- Publicar uma demo no Hugging Face Spaces.
