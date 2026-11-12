"""Nucleo do chatbot RAG: leitura de documentos, busca semantica e geracao local (CPU)."""
from __future__ import annotations

import hashlib
import os
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np

EMBED_MODEL = os.getenv("EMBED_MODEL", "intfloat/multilingual-e5-small")
LLM_MODEL = os.getenv("LLM_MODEL", "Qwen/Qwen2.5-0.5B-Instruct")
DOCS_DIR = Path(os.getenv("DOCS_DIR", "data/docs"))
CACHE_DIR = Path(".cache")
THREADS = int(os.getenv("THREADS", "4"))
MAX_NEW_TOKENS = int(os.getenv("MAX_NEW_TOKENS", "200"))

SYSTEM_PROMPT = (
    "Você é um assistente que responde em português do Brasil, de forma curta e objetiva, "
    "usando SOMENTE o contexto fornecido. Se a resposta não estiver no contexto, responda: "
    "'Não encontrei essa informação nos documentos.'"
)


@dataclass
class Chunk:
    source: str
    text: str


# ----------------------------------------------------------------------------
# Leitura e divisao dos documentos
# ----------------------------------------------------------------------------
def read_file(path: Path) -> str:
    if path.suffix.lower() == ".pdf":
        from pypdf import PdfReader

        reader = PdfReader(str(path))
        return "\n\n".join(page.extract_text() or "" for page in reader.pages)
    return path.read_text(encoding="utf-8")


def _split_long(paragraph: str, max_chars: int) -> list[str]:
    """Quebra um paragrafo muito longo em grupos de frases."""
    if len(paragraph) <= max_chars:
        return [paragraph]
    sentences = re.split(r"(?<=[.!?])\s+", paragraph)
    pieces, current = [], ""
    for sentence in sentences:
        if current and len(current) + len(sentence) + 1 > max_chars:
            pieces.append(current)
            current = sentence
        else:
            current = f"{current} {sentence}".strip()
    if current:
        pieces.append(current)
    return pieces


def chunk_text(text: str, max_chars: int = 700) -> list[str]:
    """Agrupa paragrafos em trechos de ate ~max_chars caracteres."""
    paragraphs = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
    pieces = [piece for p in paragraphs for piece in _split_long(p, max_chars)]
    chunks, current = [], ""
    for piece in pieces:
        if current and len(current) + len(piece) + 2 > max_chars:
            chunks.append(current)
            current = piece
        else:
            current = f"{current}\n\n{piece}" if current else piece
    if current:
        chunks.append(current)
    return chunks


def load_chunks(docs_dir: Path = DOCS_DIR) -> list[Chunk]:
    chunks: list[Chunk] = []
    for path in sorted(Path(docs_dir).rglob("*")):
        if path.suffix.lower() not in {".md", ".txt", ".pdf"}:
            continue
        source = str(path.relative_to(docs_dir))
        for text in chunk_text(read_file(path)):
            chunks.append(Chunk(source=source, text=text))
    return chunks


# ----------------------------------------------------------------------------
# Busca semantica
# ----------------------------------------------------------------------------
class Retriever:
    """Embeddings com sentence-transformers + similaridade de cosseno em numpy."""

    def __init__(self, chunks: list[Chunk], model_name: str = EMBED_MODEL):
        from sentence_transformers import SentenceTransformer

        self.chunks = chunks
        self.model = SentenceTransformer(model_name)
        self.embeddings = self._load_or_build(model_name)

    def _load_or_build(self, model_name: str) -> np.ndarray:
        key = hashlib.sha1(
            (model_name + "\n".join(c.text for c in self.chunks)).encode("utf-8")
        ).hexdigest()[:16]
        cache_file = CACHE_DIR / f"emb_{key}.npy"
        if cache_file.exists():
            return np.load(cache_file)
        # O modelo e5 espera o prefixo "passage: " nos documentos.
        embeddings = self.model.encode(
            [f"passage: {c.text}" for c in self.chunks],
            normalize_embeddings=True,
            batch_size=16,
            show_progress_bar=True,
        )
        CACHE_DIR.mkdir(exist_ok=True)
        np.save(cache_file, embeddings)
        return embeddings

    def search(self, query: str, k: int = 3) -> list[tuple[Chunk, float]]:
        # ... e o prefixo "query: " nas perguntas.
        q = self.model.encode([f"query: {query}"], normalize_embeddings=True)[0]
        scores = self.embeddings @ q
        top = np.argsort(-scores)[:k]
        return [(self.chunks[i], float(scores[i])) for i in top]


# ----------------------------------------------------------------------------
# Geracao com LLM local
# ----------------------------------------------------------------------------
class Generator:
    def __init__(self, model_name: str = LLM_MODEL):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        torch.set_num_threads(THREADS)
        self.torch = torch
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        try:
            self.model = AutoModelForCausalLM.from_pretrained(model_name, dtype=torch.float32)
        except TypeError:  # versoes mais antigas do transformers
            self.model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float32)
        self.model.eval()

    def generate(self, question: str, contexts: list[Chunk], stream: bool = False) -> str:
        context = "\n\n".join(f"[{i}] {c.text}" for i, c in enumerate(contexts, 1))
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": f"Contexto:\n{context}\n\nPergunta: {question}"},
        ]
        prompt = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = self.tokenizer(prompt, return_tensors="pt")

        streamer = None
        if stream:
            from transformers import TextStreamer

            streamer = TextStreamer(self.tokenizer, skip_prompt=True, skip_special_tokens=True)

        with self.torch.inference_mode():
            output = self.model.generate(
                **inputs,
                max_new_tokens=MAX_NEW_TOKENS,
                do_sample=False,
                repetition_penalty=1.1,
                streamer=streamer,
            )
        new_tokens = output[0][inputs["input_ids"].shape[1]:]
        return self.tokenizer.decode(new_tokens, skip_special_tokens=True).strip()


# ----------------------------------------------------------------------------
# Chatbot completo
# ----------------------------------------------------------------------------
class RagChatbot:
    def __init__(self, docs_dir: Path = DOCS_DIR):
        self.chunks = load_chunks(docs_dir)
        if not self.chunks:
            raise ValueError(f"Nenhum documento (.md, .txt ou .pdf) encontrado em {docs_dir}")
        self.retriever = Retriever(self.chunks)
        self.generator = Generator()

    def ask(self, question: str, k: int = 3, stream: bool = False):
        hits = self.retriever.search(question, k)
        answer = self.generator.generate(question, [chunk for chunk, _ in hits], stream)
        return answer, hits
