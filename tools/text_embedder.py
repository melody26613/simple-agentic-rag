import requests
import os

# from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.embeddings import Embeddings

TEXT_EMBEDDING_MODEL_DIM = 768


class TextEmbedder(Embeddings):
    __TEXT_EMBEDDER_URL = os.getenv(
        "TEXT_EMBEDDER_URL", "http://localhost:11434/api/embed"
    )
    __TEXT_EMBEDDER_MODEL = os.getenv(
        "TEXT_EMBEDDER_MODEL", "qllama/multilingual-e5-base:latest"
    )

    __TEXT_EMBEDDER_TIMEOUT = 60

    def __init__(self):
        self.api_url = self.__TEXT_EMBEDDER_URL
        self.model_name = self.__TEXT_EMBEDDER_MODEL
        print(
            f"[TextEmbedder] init text embedding model {self.api_url=}, {self.model_name=}"
        )

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        print(f"[embed_documents] {texts=}")

        embeddings = []
        for text in texts:
            payload = {"model": self.model_name, "input": text}
            try:
                response = requests.post(
                    self.api_url,
                    json=payload,
                    timeout=self.__TEXT_EMBEDDER_TIMEOUT,
                )
                response.raise_for_status()
                result = response.json()
                embeddings.append(result["embeddings"][0])
            except Exception as e:
                print(f"Error embedding '{text}': {e}")
                embeddings.append([])  # or raise, depending on your needs
        return embeddings

    def embed_query(self, text: str) -> list[float]:
        return self.embed_documents([text])[0]


if __name__ == "__main__":
    embedder = TextEmbedder()
    embedding = embedder.embed_query("你好")
    print(f"{embedding=}, length of embedding={len(embedding)}")
