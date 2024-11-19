import logging
import colorlog

from langchain_community.document_loaders import PDFPlumberLoader, DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_pinecone import Pinecone
from langchain.schema import Document
from langchain_ollama import OllamaEmbeddings

from .config import get_settings

settings = get_settings()

handler = colorlog.StreamHandler()
handler.setFormatter(
    colorlog.ColoredFormatter(
        "%(log_color)s%(levelname)s:%(name)s:%(message)s",
        log_colors={
            "DEBUG": "cyan",
            "INFO": "green",
            "WARNING": "yellow",
            "ERROR": "red",
            "CRITICAL": "bold_red",
        },
    )
)

logger = logging.getLogger(__name__)
logger.addHandler(handler)
logger.setLevel(logging.INFO)


class VectorStore:
    embed_model = OllamaEmbeddings(model=settings.EMBEDDING_MODEL)

    def load_pdf(self, dir_path):
        try:
            loader = DirectoryLoader(
                path=dir_path,
                glob="*.pdf",
                loader_cls=PDFPlumberLoader,
            )
            return loader.load()
        except Exception as e:
            logger.error(f"Error loading PDFs: {e}")
            raise

    def split_text(self, extracted_data):
        try:
            splitter = RecursiveCharacterTextSplitter(
                chunk_size=settings.CHUNK_SIZE,
                chunk_overlap=settings.CHUNK_OVERLAP,
            )
            return splitter.split_documents(extracted_data)
        except Exception as e:
            logger.error(f"Error splitting text: {e}")
            raise

    @classmethod
    def get_vector_database(cls):
        try:
            return Pinecone(
                pinecone_api_key=settings.PINECONE_API_KEY,
                embedding=cls.embed_model,
                index_name=settings.PINECONE_INDEX_NAME,
            )
        except Exception as e:
            logger.error(f"Error initializing Pinecone: {e}")
            raise

    @classmethod
    def upload_vectors(cls, dir_path):
        try:
            extracted_data = VectorStore.load_pdf(dir_path)
            text_chunks = VectorStore.split_text(extracted_data)
            documents = [
                Document(page_content=chunk.page_content) for chunk in text_chunks
            ]
            docstore = cls.get_vector_database()
            docstore.from_documents(
                documents=documents,
                index_name=settings.PINECONE_INDEX_NAME,
                embedding=cls.embed_model,
            )
            logger.info(
                f"Successfully uploaded {len(documents)} documents to Pinecone."
            )
        except Exception as e:
            logger.error(f"Error uploading vectors: {e}")
            raise


if __name__ == "__main__":
    # vector_store = VectorStore()
    # vector_store.upload_vectors("/path/to/data_directory")
    pass
