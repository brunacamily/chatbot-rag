"""Avalia a busca (retrieval) do RAG: Hit@1, Hit@k, MRR e tempo por consulta.

Uso:
    python eval.py                 # so a busca (rapido)
    python eval.py --generate 3    # tambem mede o tempo de geracao em 3 perguntas
"""
import argparse
import json
import time
from pathlib import Path

from rag import DOCS_DIR, Retriever, load_chunks


def rank_of_expected(hits, expected_source):
    """Posicao (1, 2, ...) da primeira fonte esperada nos resultados, ou None."""
    for pos, (chunk, _) in enumerate(hits, start=1):
        if chunk.source == expected_source:
            return pos
    return None


def summarize(ranks, k):
    n = len(ranks)
    return {
        "perguntas": n,
        "hit@1": sum(r == 1 for r in ranks) / n,
        f"hit@{k}": sum(r is not None and r <= k for r in ranks) / n,
        "mrr": sum(1 / r for r in ranks if r) / n,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--eval-file", default="data/eval.json")
    parser.add_argument("--k", type=int, default=3)
    parser.add_argument("--generate", type=int, default=0, help="n de perguntas para medir a geracao")
    args = parser.parse_args()

    perguntas = json.loads(Path(args.eval_file).read_text(encoding="utf-8"))
    chunks = load_chunks(DOCS_DIR)
    retriever = Retriever(chunks)

    ranks, tempos = [], []
    for item in perguntas:
        inicio = time.perf_counter()
        hits = retriever.search(item["question"], k=args.k)
        tempos.append((time.perf_counter() - inicio) * 1000)
        rank = rank_of_expected(hits, item["expected_source"])
        ranks.append(rank)
        marca = "OK " if rank == 1 else ("~  " if rank else "ERR")
        print(f"[{marca}] rank={rank}  {item['question']}")

    resumo = summarize(ranks, args.k)
    resumo["ms_por_consulta"] = sum(tempos) / len(tempos)
    resumo["trechos_indexados"] = len(chunks)

    if args.generate:
        from rag import Generator

        gerador = Generator()
        duracoes = []
        for item in perguntas[: args.generate]:
            hits = retriever.search(item["question"], k=args.k)
            inicio = time.perf_counter()
            gerador.generate(item["question"], [c for c, _ in hits])
            duracoes.append(time.perf_counter() - inicio)
        resumo["s_por_resposta"] = sum(duracoes) / len(duracoes)

    print("\nResumo:")
    for chave, valor in resumo.items():
        print(f"  {chave}: {valor:.3f}" if isinstance(valor, float) else f"  {chave}: {valor}")

    Path("results").mkdir(exist_ok=True)
    Path("results/eval.json").write_text(
        json.dumps(resumo, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print("\nSalvo em results/eval.json")


if __name__ == "__main__":
    main()
