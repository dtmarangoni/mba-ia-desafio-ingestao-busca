from operator import itemgetter
from textwrap import dedent

from langchain.chat_models import init_chat_model
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate

from config import load_get_config
from vector_store import get_vector_store

config = load_get_config()
prompt_template = PromptTemplate.from_template(dedent(
    """
    CONTEXTO:
    {contexto}

    REGRAS:
    - Responda somente com base no CONTEXTO.
    - Se a informação não estiver explicitamente no CONTEXTO, responda:
    "Não tenho informações necessárias para responder sua pergunta."
    - Nunca invente ou use conhecimento externo.
    - Nunca produza opiniões ou interpretações além do que está escrito.

    EXEMPLOS DE PERGUNTAS FORA DO CONTEXTO:
    Pergunta: "Qual é a capital da França?"
    Resposta: "Não tenho informações necessárias para responder sua pergunta."

    Pergunta: "Quantos clientes temos em 2024?"
    Resposta: "Não tenho informações necessárias para responder sua pergunta."

    Pergunta: "Você acha isso bom ou ruim?"
    Resposta: "Não tenho informações necessárias para responder sua pergunta."

    PERGUNTA DO USUÁRIO:
    {pergunta}

    RESPONDA A "PERGUNTA DO USUÁRIO"
    """
).strip())
llm = init_chat_model(config.llm_model, api_key=config.api_key)
chain = prompt_template | llm | StrOutputParser()


def search(question: str) -> list[tuple[Document, float]]:
    vector_store = get_vector_store()
    return vector_store.similarity_search_with_score(question, k=10)


def get_answer(question: str | None) -> str:
    if not question:
        return ''

    context_docs = search(question)
    context = '\n'.join([p.page_content for p, _ in sorted(context_docs, key=itemgetter(1))])
    return chain.invoke({'contexto': context, 'pergunta': question})
