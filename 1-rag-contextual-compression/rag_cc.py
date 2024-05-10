import os
import logging
from typing import List, Optional

from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import NLTKTextSplitter
from langchain_community.chat_models import ChatOllama
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import LLMChainExtractor, EmbeddingsFilter, DocumentCompressorPipeline
from langchain.embeddings.sentence_transformer import SentenceTransformerEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_community.document_transformers import EmbeddingsRedundantFilter
from langchain.chains import RetrievalQA
from langchain_openai import AzureChatOpenAI
from langchain.prompts import ChatPromptTemplate, PromptTemplate
from warnings import filterwarnings

# Ignore warnings from LLMChainExtractor
filterwarnings("ignore", category=UserWarning)

# Configure logging
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class RAGContextualCompression:

    """
    RAG pipeline with compressor for filtering and document compression.
    """

    def __init__(self, data_path: str) -> None:
        self.data_path = data_path
        self.vector_store = Chroma
        self.llmchainextractor = LLMChainExtractor
        self.retrieval_qa = RetrievalQA
        self.embedding_model = SentenceTransformerEmbeddings(model_name="thenlper/gte-large")
        self.text_splitter = NLTKTextSplitter()
        self.azure_openai = AzureChatOpenAI(
            openai_api_version=os.getenv('AZURE_OPENAI_VERSION', 'default_version'),
            azure_deployment=os.getenv('AZURE_OPENAI_DEPLOYMENT_NAME', 'default_deployment'),
            temperature=0
        )
        self.llama = ChatOllama(model="llama3", format="json", temperature=0)


    def load_documents(self) -> List[str]:
        """Loads and prepares PDF documents from a directory."""
        try:
            pdf_files = [f for f in os.listdir(self.data_path) if f.endswith('.pdf')]
            docs = [PyPDFLoader(os.path.join(self.data_path, file)).load() for file in pdf_files]
            logger.info(f"Loaded {len(docs)} files")
            return docs
        except Exception as e:
            logger.error("Failed to load documents: %s", e, exc_info=True)
            return []


    def preprocess_documents(self, docs: List[str]) -> List[str]:
        """Preprocesses and chunks documents into manageable parts."""
        try:
            docs_list = [item for sublist in docs for item in sublist]
            chunks = self.text_splitter.split_documents(docs_list)
            logger.info(f"Documents chunked. Total no. of chunks: {len(chunks)}")
            return chunks
        except Exception as e:
            logger.error("Failed to preprocess documents: %s", e, exc_info=True)
            return []


    def setup_vector_store(self, chunks: List[str]) -> Optional[Chroma]:
        """Initializes and returns a ChromaDB vector store with documents and embedding model."""
        try:
            db = self.vector_store.from_documents(documents=chunks, embedding=self.embedding_model)
            logger.info("Vector store successfully created")
            return db
        except Exception as e:
            logger.error("Failed to initialize vector store: %s", e, exc_info=True)
            return None


    def setup_base_retriever(self, db: Chroma):
        try:
            retriever = db.as_retriever(search_type="mmr", search_kwargs={"k": 5})
            logger.info("Retriever successfully created!")
            return retriever
        except Exception as e:
            logger.error("Failed to create retriever: %s", e, exc_info=True)
            return None


    def retrieve_documents(self, db: Chroma, user_query: str) -> List[str]:
        """Retrieves documents relevant to the given user query."""
        try:
            retriever = self.setup_base_retriever(db)
            documents = retriever.invoke(user_query)
            logger.info(f"Successfully retrieved {len(documents)} documents")
            return documents
        except Exception as e:
            logger.error("Failed to retrieve documents: %s", e, exc_info=True)
            return []


    def setup_compressor(self, use_llm: bool = False, k: int = 10, similarity_threshold: float = 0.6):
        """Sets up the document compressor."""
        try:
            if use_llm:
                compressor = self.llmchainextractor.from_llm(self.azure_openai)
            else:
                compressor = EmbeddingsFilter(embeddings=self.embedding_model, k=k, similarity_threshold=similarity_threshold)
            logger.info("Compressor setup successfully")
            return compressor
        except Exception as e:
            logger.error("Failed to set up compressor: %s", e, exc_info=True)
            return None


    def setup_redundant_filter(self, similarity_threshold: float = 0.98):
        """Sets up the redundant document filter."""
        try:
            redundant_filter = EmbeddingsRedundantFilter(embeddings=self.embedding_model, similarity_threshold=similarity_threshold)
            logger.info("Redundant filter setup successfully")
            return redundant_filter
        except Exception as e:
            logger.error("Failed to set up redundant filter: %s", e, exc_info=True)
            return None


    def setup_compression_pipeline_retriever(self, db: Chroma):
        """Sets up the compression pipeline and contextual compression retriever."""
        try:
            base_retriever = self.setup_base_retriever(db)
            compressor = self.setup_compressor(use_llm=False)
            redundant_filter = self.setup_redundant_filter()
            pipeline_compressor = DocumentCompressorPipeline(transformers=[compressor, redundant_filter])
            contextual_comp_retriever = ContextualCompressionRetriever(base_compressor=pipeline_compressor,
                                                                       base_retriever=base_retriever,
                                                                       search_kwargs={"k": 5})
            logger.info("Compression pipeline retriever setup successfully")
            return contextual_comp_retriever
        except Exception as e:
            logger.error("Failed to set up compression pipeline retriever: %s", e, exc_info=True)
            return None


    def generate_answer(self, retriever, user_query: str) -> str:
        """Generate an answer to the user's query using a structured prompt in chat format."""

        system_prompt_template = """

            You are a brilliant assistant tasked with providing concise and accurate answers.
            You always greet the user by saying "Yow yow, nice to see you here, curious mind!"
            Then you provide a detailed answer.
            Do not hallucinate. 
            If there are multiple questions in the query - always break the line for the answer for each question.

            USER QUERY: {question}
            CONTEXT: {context}

            Your response must be maximum 3 sentences.

        """

        SYSTEM_PROMPT = PromptTemplate(
            input_variables=["question", "context"],
            template=system_prompt_template,
        )

        try:
            qa_chain = self.retrieval_qa.from_chain_type(llm=self.azure_openai, 
                                                        chain_type="stuff", 
                                                        retriever=retriever,
                                                        chain_type_kwargs={"prompt": SYSTEM_PROMPT})
            response = qa_chain.invoke({"query": user_query})
            logger.info("Answer generated successfully!")
            return response['result']
        except Exception as e:
            logger.error("Failed to generate answer: %s", e, exc_info=True)
            return "Error generating answer"
