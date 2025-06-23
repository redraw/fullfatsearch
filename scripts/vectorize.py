import os
import argparse
import frontmatter

from langchain_text_splitters import RecursiveCharacterTextSplitter

from langchain_openai.embeddings import OpenAIEmbeddings
from langchain_qdrant import QdrantVectorStore
from langchain_core.documents import Document
from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, VectorParams

QDRANT_COLLECTION_NAME = os.getenv("QDRANT_COLLECTION_NAME", "fullfatvector")
QDRANT_LOCATION = os.getenv("QDRANT_LOCATION", None)
QDRANT_PATH = os.getenv("QDRANT_PATH", None)
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY", None)


def load_frontmatter_docs(filenames: list[str]) -> list[Document]:
    docs = []
    for fn in filenames:
        item = frontmatter.load(fn)
        docs.append(
            Document(
                page_content=item.content,
                metadata={
                    "source": os.path.basename(fn),
                    **item.metadata,
                },
            )
        )
    return docs


def main(args):
    # load
    docs = load_frontmatter_docs(args.files)

    if not docs:
        print("No docs.")
        return

    # split
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    splits = text_splitter.split_documents(docs)

    # embed
    client = QdrantClient(
        location=QDRANT_LOCATION,
        api_key=QDRANT_API_KEY,
        path=QDRANT_PATH,
    )

    if not client.collection_exists(QDRANT_COLLECTION_NAME):
        client.create_collection(
            collection_name=QDRANT_COLLECTION_NAME,
            vectors_config=VectorParams(size=1536, distance=Distance.COSINE),
        )

    embeddings = OpenAIEmbeddings(
        model="text-embedding-3-small",
        show_progress_bar=True,
    )

    vector_store = QdrantVectorStore(
        client=client,
        collection_name=QDRANT_COLLECTION_NAME,
        embedding=embeddings,
    )

    if args.limit:
        splits = splits[: args.limit]

    vector_store.add_documents(splits)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("files", nargs="*", type=str)
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()
    main(args)
