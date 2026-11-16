from search import get_answer

BOLD, CYAN, RESET = "\033[1m", "\033[36m", "\033[0m"


def init_chat():
    while True:
        try:
            question = input(f'{BOLD}Pergunta (digite sair para encerrar):{RESET} ').strip()
            if not question:
                continue
            if question.lower() == 'sair':
                break
            answer = get_answer(question)
            print(f'{CYAN}RESPOSTA:{RESET} {answer}\n')
        except (EOFError, KeyboardInterrupt):
            print()
            break
