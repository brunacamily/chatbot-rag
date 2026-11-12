"""Chat no terminal: python chat.py"""
from rag import RagChatbot


def main() -> None:
    print("Carregando modelos (na primeira vez baixa cerca de 1,5 GB)...")
    try:
        bot = RagChatbot()
    except ValueError as erro:
        raise SystemExit(erro)
    print(f"{len(bot.chunks)} trechos indexados. Digite 'sair' para encerrar.\n")

    while True:
        try:
            pergunta = input("Você: ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if pergunta.lower() in {"sair", "exit", "quit"}:
            break
        if not pergunta:
            continue
        print("Bot: ", end="", flush=True)
        _, hits = bot.ask(pergunta, stream=True)
        fontes = ", ".join(sorted({chunk.source for chunk, _ in hits}))
        print(f"Fontes: {fontes}\n")


if __name__ == "__main__":
    main()
