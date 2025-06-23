import os
import typing

import streamlit as st
import httpx
from openai.types.responses import ResponseTextDeltaEvent
from agents import (
    Agent,
    Runner,
    function_tool,
    trace,
    set_default_openai_key,
)
from langchain_core.documents import Document
from langchain_openai.embeddings import OpenAIEmbeddings
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient, models as qdr

st.set_page_config(
    page_title="Full Fat GPT",
    page_icon=":robot:",
    initial_sidebar_state="collapsed",
)

INSTRUCTIONS = """Eres experto del podcast "Circulo Vicioso". Ayuda al usuario a encontrar momentos en los episodios.
- Responde SIEMPRE construyendo un link con el nombre del video y timestamp donde se encuentra la conversacion.
- Incluye un poco de contexto de cada resultado.
- Utiliza `vector_search` para buscar por contexto semántico.
- Utiliza `fts_search` para buscar por palabras clave (una o dos palabras). 
- El formato del link debe ser: `[title (timestamp)](https://youtube.com/watch?v=<id>&t=<seconds>)`.
- Usa el contexto para enumerar todas las ocurrencias. 
- Refierete al usuario por gordo. """


# Model and API Key configuration in sidebar
with st.sidebar:
    openai_api_key = st.text_input(
        "OpenAI API Key",
        value="",
        type="password",
        help="Enter your OpenAI API key or leave empty to use OPENAI_API_KEY environment variable",
    )
    openai_models = [
        "gpt-4.1",
        "gpt-4.1-nano",
        "gpt-4.1-mini",
        "gpt-4.1-turbo",
        "o3-mini",
        "o4-mini",
        "gpt-3.5-turbo",
    ]
    selected_model = st.selectbox(
        "OpenAI Model",
        openai_models,
        index=openai_models.index(os.getenv("OPENAI_MODEL", "gpt-4.1-nano")),
        disabled=not openai_api_key,
        accept_new_options=True,
    )
    instructions = st.text_area(
        "System Prompt",
        value=INSTRUCTIONS,
        height=200,
        disabled=not openai_api_key,
    )
    debug = st.toggle(
        "Debug",
        value=False,
    )

QDRANT_COLLECTION_NAME = os.getenv("QDRANT_COLLECTION_NAME", "fullfatvector")
QDRANT_LOCATION = os.getenv("QDRANT_LOCATION", None)
QDRANT_PATH = os.getenv("QDRANT_PATH", None)
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY", None)
OPENAI_MODEL = selected_model
OPENAI_API_KEY = openai_api_key if openai_api_key else os.getenv("OPENAI_API_KEY", "")
FFS_DATASETTE = os.getenv("FFS_DATASETTE", "http://localhost:8001")

set_default_openai_key(OPENAI_API_KEY)
# enable_verbose_stdout_logging()


@st.cache_resource
def get_qdrant_client():
    return QdrantClient(
        location=QDRANT_LOCATION,
        api_key=QDRANT_API_KEY,
        path=QDRANT_PATH,
    )


@st.cache_resource
def get_vectorstore():
    return QdrantVectorStore(
        client=get_qdrant_client(),
        collection_name=QDRANT_COLLECTION_NAME,
        embedding=OpenAIEmbeddings(
            model="text-embedding-3-small",
            api_key=OPENAI_API_KEY,
        ),
    )


def format_vector_results(results: list[Document]) -> str:
    return "\n\n".join(
        [
            f"- {r.metadata['title']}\n{r.metadata['url']}\n{r.page_content}"
            for r in results
        ]
    )


@function_tool
def fts_search(
    query: str,
) -> list[dict] | str:
    """
    Recupera documentos de episodios de pódcast relevantes por palabras clave.

    Args:
        query (str): La consulta de búsqueda para encontrar documentos relevantes.
    """
    print(f"[fts_search] {query=}")

    response = httpx.get(
        f"{FFS_DATASETTE}/youtube/fts_context.json",
        params={
            "q": query,
            "limit": 15,
            "context_seconds": 30,
            "_shape": "objects",
        },
    )

    response.raise_for_status()
    response = response.json()

    if not response.get("ok"):
        print(response)

    rows = response.get("rows", [])

    if not rows:
        return "No se encontraron resultados relevantes."

    return rows


@function_tool
def vector_search(
    query: str,
    youtube_video_id: str | None = None,
    named_entities: list[str] | None = None,
):
    """
    Recupera documentos de episodios de pódcast relevantes semánticamente.

    Args:
        query (str): La consulta de búsqueda.
        youtube_video_id (str, opcional): El ID del video de YouTube para filtrar los resultados (en caso de hablar de un video en particular).
        named_entities (list[str], opcional): Lista de entidades (NER). Usa este field exclusivamente con nombres propios o nombres de lugares.
    """
    print(f"[vector_search] {query=} {youtube_video_id=} {named_entities=}")
    fq = {"must": [], "must_not": [], "should": []}
    filters = None

    if youtube_video_id:
        q = qdr.FieldCondition(
            key="metadata.source",
            match=qdr.MatchValue(value=f"{youtube_video_id}.es.md"),
        )
        fq["must"].append(q)

    if named_entities:
        for keyword in named_entities:
            q = qdr.FieldCondition(
                key="page_content",
                match=qdr.MatchText(text=keyword),
            )
            fq["should"].append(q)

    if fq:
        filters = qdr.Filter(**fq)
        print(f"Filters: {filters}")

    results = vectorstore.max_marginal_relevance_search(
        query, k=5, score_threshold=0.1, filter=filters
    )
    docs = format_vector_results(results)
    print(f"Retrieved {len(results)} documents:\n{docs}")

    if not results:
        return "No se encontraron resultados relevantes."

    return docs


@st.cache_resource
def get_agent():
    return Agent(
        name="Buscador de episodios",
        instructions=instructions,
        tools=[fts_search, vector_search],
        model=OPENAI_MODEL,
    )


agent = get_agent()
vectorstore = get_vectorstore()

if "messages" not in st.session_state:
    st.session_state.messages = [
        {
            "role": "assistant",
            "content": "Hola gordo, ¿en qué puedo ayudarte?",
        }
    ]

for message in st.session_state.messages:
    if "role" in message:
        with st.chat_message(message["role"]):
            try:
                st.markdown(message["content"][0]["text"])
            except TypeError:
                st.markdown(message["content"])
    elif debug:
        st.json(message)

# NEW MESSAGE
if q := st.chat_input("Consulta", max_chars=1000):
    st.session_state.messages.append({"role": "user", "content": q})

    with st.chat_message("user"):
        st.markdown(q)

    async def generate_response(
        chat_history: typing.Iterable,
    ) -> typing.AsyncGenerator[str, None]:
        messages = list(chat_history)
        result = Runner.run_streamed(agent, input=messages)
        async for event in result.stream_events():
            if event.type == "raw_response_event" and isinstance(
                event.data, ResponseTextDeltaEvent
            ):
                yield event.data.delta
            elif event.type == "run_item_stream_event":
                st.session_state.messages.append(event.item.to_input_item())

    def get_chat_history(n: int = 2):
        messages = st.session_state.messages
        user_messages_indexes = [
            idx for idx, m in enumerate(messages) if m.get("role") == "user"
        ]
        return messages[user_messages_indexes[-min(len(user_messages_indexes), n)] :]

    with st.chat_message("assistant"):
        with trace("chat"):
            chat_history = get_chat_history()
            st.write_stream(generate_response(chat_history))
