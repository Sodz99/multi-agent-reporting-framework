"""Simple Bedrock RAG pipeline for automated reporting system."""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple

import chromadb
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_aws import BedrockEmbeddings
from dotenv import load_dotenv

from ..data.fixtures import FixtureLoader

# Load environment variables
load_dotenv()


class BedrockRAGPipeline:
    """Simple RAG pipeline using AWS Bedrock Titan V1 embeddings and ChromaDB for vector storage."""

    def __init__(
        self,
        collection_name: str = "automated_reporting",
        persist_directory: Optional[Path] = None,
        verbose: bool = True,
    ):
        self.collection_name = collection_name
        self.verbose = verbose

        if self.verbose:
            print(f"[RAG] Initializing automated reporting RAG Pipeline")

        # Initialize Bedrock embeddings with simple Titan V1
        try:
            self.embeddings = BedrockEmbeddings(
                model_id="amazon.titan-embed-text-v1",
                region_name=os.getenv("BEDROCK_REGION", "us-east-1"),
            )

            if self.verbose:
                print(f"[RAG] Bedrock embeddings initialized (Titan V1)")

        except Exception as e:
            if self.verbose:
                print(f"[RAG] Error: Could not initialize Bedrock embeddings: {e}")
                print("[RAG] Please configure AWS credentials in .env file")
            raise RuntimeError(f"Bedrock embeddings initialization failed: {e}")

        # Initialize ChromaDB
        if persist_directory:
            persist_directory.mkdir(parents=True, exist_ok=True)
            self.chroma_client = chromadb.PersistentClient(path=str(persist_directory))
            if self.verbose:
                print(f"[RAG] Using persistent ChromaDB at: {persist_directory}")
        else:
            self.chroma_client = chromadb.Client()
            if self.verbose:
                print(f"[RAG] Using in-memory ChromaDB")

        self.collection = self.chroma_client.get_or_create_collection(name=collection_name)

        # Simple text splitter
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len,
        )

        if self.verbose:
            print(f"[RAG] Automated reporting RAG Pipeline initialized!")
            print(f"[RAG] Collection: {collection_name} ({self.collection.count()} existing documents)")

    def load_fixtures_to_vector_store(self, fixtures_dir: Path) -> Dict[str, int]:
        """Simple fixture loading to vector store."""
        loader = FixtureLoader(fixtures_dir)
        fixtures = list(loader.iter_fixtures())

        stats = {"total_docs": 0, "total_chunks": 0}
        documents = []
        metadatas = []
        ids = []

        if self.verbose:
            print(f"[RAG] Loading {len(fixtures)} fixtures...")

        for fixture in fixtures:
            try:
                content = fixture.load()
                doc_text = self._extract_text_from_content(content)

                # Skip empty content
                if len(doc_text.strip()) < 50:
                    if self.verbose:
                        print(f"[RAG] Skipping {fixture.fixture_id}: content too short")
                    continue

                # Split document into chunks
                chunks = self.text_splitter.split_text(doc_text)

                for i, chunk in enumerate(chunks):
                    doc_id = f"{fixture.fixture_id}_chunk_{i}"
                    ids.append(doc_id)
                    documents.append(chunk)
                    metadatas.append({
                        "fixture_id": fixture.fixture_id,
                        "description": fixture.description,
                        "file_path": str(fixture.path),
                        "chunk_index": i,
                    })

                stats["total_docs"] += 1
                stats["total_chunks"] += len(chunks)

                if self.verbose:
                    print(f"[RAG] Processed {fixture.fixture_id}: {len(chunks)} chunks")

            except Exception as e:
                if self.verbose:
                    print(f"[RAG] Error processing {fixture.fixture_id}: {e}")
                continue

        # Add documents to vector store
        if documents:
            self.collection.add(
                ids=ids,
                documents=documents,
                metadatas=metadatas
            )

            if self.verbose:
                print(f"[RAG] Successfully indexed {stats['total_chunks']} chunks from {stats['total_docs']} documents")

        return stats

    def similarity_search(
        self,
        query: str,
        top_k: int = 5
    ) -> List[Dict[str, Any]]:
        """Simple similarity search."""
        try:
            if not self.embeddings:
                raise RuntimeError("Bedrock embeddings not available")

            # Search in ChromaDB
            results = self.collection.query(
                query_texts=[query],
                n_results=top_k,
                include=["documents", "metadatas", "distances"]
            )

            # Convert results
            search_results = []
            if results['documents'] and results['documents'][0]:
                for doc_content, metadata, distance in zip(
                    results['documents'][0],
                    results['metadatas'][0],
                    results['distances'][0]
                ):
                    similarity_score = max(0.0, 1.0 - distance)

                    search_results.append({
                        "content": doc_content,
                        "metadata": metadata,
                        "score": similarity_score
                    })

            if self.verbose and search_results:
                print(f"[RAG] Found {len(search_results)} results")

            return search_results

        except Exception as e:
            if self.verbose:
                print(f"[RAG] Error during similarity search: {e}")
            raise

    def get_grounded_context(
        self,
        query: str,
        top_k: int = 3
    ) -> Tuple[str, List[str]]:
        """Get grounded context for a query with citations."""
        search_results = self.similarity_search(query, top_k)

        if not search_results:
            return "No relevant context found.", []

        # Build context from search results
        context_parts = []
        citations = []

        for result in search_results:
            content = result["content"]
            metadata = result["metadata"]
            score = result["score"]

            # Add document content to context
            context_parts.append(
                f"[Source: {metadata.get('fixture_id', 'unknown')}] {content}"
            )

            # Build citation
            citation = (
                f"{metadata.get('fixture_id', 'unknown')} "
                f"({metadata.get('description', 'no description')}) "
                f"- Relevance: {score:.2f}"
            )
            citations.append(citation)

        context = "\n\n".join(context_parts)
        return context, citations

    def _extract_text_from_content(self, content: Any) -> str:
        """Extract text content from various data types."""
        if isinstance(content, str):
            return content
        elif isinstance(content, dict):
            return json.dumps(content, indent=2)
        elif isinstance(content, list):
            return json.dumps(content, indent=2)
        else:
            return str(content)

    def get_collection_stats(self) -> Dict[str, Any]:
        """Get basic collection statistics."""
        try:
            count = self.collection.count()
            return {
                "collection_name": self.collection_name,
                "total_documents": count,
                "embedding_model": "amazon.titan-embed-text-v1",
            }
        except Exception as e:
            return {"error": str(e)}


__all__ = [
    "BedrockRAGPipeline",
]