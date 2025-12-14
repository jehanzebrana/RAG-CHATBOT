"""
Module: rag_engine.py
Task 2: Advanced RAG System with CV

This module implements a sophisticated RAG pipeline with:
1. Hybrid Search (BM25 + Vector Similarity)
2. Cross-Encoder Re-ranking
3. Query Enhancement
4. Source Citations

Features:
- PDF document processing
- Semantic chunking with overlap
- Persistent vector storage (ChromaDB)
- Metadata-rich retrieval
- Confidence scoring
"""

import os
import re
from typing import List, Dict, Optional, Tuple
from pathlib import Path

from loguru import logger
import PyPDF2
from sentence_transformers import SentenceTransformer, CrossEncoder
import chromadb
from chromadb.config import Settings
from rank_bm25 import BM25Okapi
import numpy as np

from modules.llm_handler import LLMHandler


class DocumentChunk:
    """Represents a chunk of text from a document."""
    
    def __init__(
        self,
        text: str,
        metadata: Dict,
        chunk_id: str,
        page_number: int = 0
    ):
        self.text = text
        self.metadata = metadata
        self.chunk_id = chunk_id
        self.page_number = page_number
        self.score: float = 0.0


class AdvancedRAGEngine:
    """
    Advanced Retrieval-Augmented Generation Engine.
    
    Combines multiple retrieval and ranking strategies for optimal results.
    """
    
    def __init__(
        self,
        llm_handler: LLMHandler,
        embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2",
        reranker_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
        persist_directory: str = "./vector_store",
        collection_name: str = "cv_documents",
        chunk_size: int = 500,
        chunk_overlap: int = 50,
        top_k_retrieval: int = 10,
        top_k_rerank: int = 3
    ):
        """
        Initialize Advanced RAG Engine.
        
        Args:
            llm_handler: LLM handler instance
            embedding_model: Sentence transformer model for embeddings
            reranker_model: Cross-encoder model for re-ranking
            persist_directory: Directory for ChromaDB persistence
            collection_name: Name of the vector collection
            chunk_size: Target size for text chunks (characters)
            chunk_overlap: Overlap between chunks (characters)
            top_k_retrieval: Number of chunks to retrieve initially
            top_k_rerank: Number of chunks after re-ranking
        """
        self.llm_handler = llm_handler
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.top_k_retrieval = top_k_retrieval
        self.top_k_rerank = top_k_rerank
        
        # Initialize embedding model
        logger.info(f"Loading embedding model: {embedding_model}")
        self.embedder = SentenceTransformer(embedding_model)
        
        # Initialize re-ranker
        logger.info(f"Loading re-ranker model: {reranker_model}")
        self.reranker = CrossEncoder(reranker_model)
        
        # Initialize ChromaDB
        self.persist_directory = persist_directory
        Path(persist_directory).mkdir(parents=True, exist_ok=True)
        
        self.chroma_client = chromadb.PersistentClient(
            path=persist_directory,
            settings=Settings(anonymized_telemetry=False)
        )
        
        self.collection = self.chroma_client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"}
        )
        
        # BM25 index (will be populated when documents are loaded)
        self.bm25_index: Optional[BM25Okapi] = None
        self.document_chunks: List[DocumentChunk] = []
        
        logger.info("✓ Advanced RAG Engine initialized")
    
    def load_pdf(self, pdf_path: str) -> None:
        """
        Load and process a PDF document.
        
        Args:
            pdf_path: Path to the PDF file
        """
        logger.info(f"Loading PDF: {pdf_path}")
        
        # Extract text from PDF
        text_by_page = self._extract_pdf_text(pdf_path)
        
        # Create chunks
        all_chunks = []
        for page_num, page_text in enumerate(text_by_page, start=1):
            chunks = self._create_chunks(page_text, page_num)
            all_chunks.extend(chunks)
        
        logger.info(f"Created {len(all_chunks)} chunks from PDF")
        
        # Store chunks in vector database
        self._store_chunks(all_chunks, pdf_path)
        
        # Build BM25 index for keyword search
        self._build_bm25_index(all_chunks)
        
        logger.info("✓ PDF loaded and indexed successfully")
    
    def _extract_pdf_text(self, pdf_path: str) -> List[str]:
        """Extract text from PDF, page by page."""
        text_by_page = []
        
        try:
            with open(pdf_path, 'rb') as file:
                pdf_reader = PyPDF2.PdfReader(file)
                num_pages = len(pdf_reader.pages)
                
                for page_num in range(num_pages):
                    page = pdf_reader.pages[page_num]
                    text = page.extract_text()
                    text_by_page.append(text)
                
                logger.info(f"Extracted text from {num_pages} pages")
                
        except Exception as e:
            logger.error(f"Error reading PDF: {e}")
            raise
        
        return text_by_page
    
    def _create_chunks(self, text: str, page_number: int) -> List[DocumentChunk]:
        """
        Split text into overlapping chunks.
        
        Uses a simple character-based chunking strategy with overlap.
        """
        chunks = []
        text = text.strip()
        
        if not text:
            return chunks
        
        start = 0
        chunk_counter = 0
        
        while start < len(text):
            end = start + self.chunk_size
            chunk_text = text[start:end]
            
            # Try to break at sentence boundary
            if end < len(text):
                last_period = chunk_text.rfind('. ')
                if last_period > self.chunk_size * 0.5:  # At least 50% into chunk
                    end = start + last_period + 1
                    chunk_text = text[start:end]
            
            chunk = DocumentChunk(
                text=chunk_text.strip(),
                metadata={
                    "page": page_number,
                    "chunk_index": chunk_counter,
                    "start_char": start,
                    "end_char": end
                },
                chunk_id=f"page_{page_number}_chunk_{chunk_counter}",
                page_number=page_number
            )
            
            chunks.append(chunk)
            chunk_counter += 1
            
            # Move start with overlap
            start = end - self.chunk_overlap
            
            if start >= len(text):
                break
        
        return chunks
    
    def _store_chunks(self, chunks: List[DocumentChunk], source_doc: str) -> None:
        """Store chunks in ChromaDB vector store."""
        if not chunks:
            return
        
        # Prepare data for ChromaDB
        ids = [chunk.chunk_id for chunk in chunks]
        documents = [chunk.text for chunk in chunks]
        metadatas = [
            {
                **chunk.metadata,
                "source_document": source_doc
            }
            for chunk in chunks
        ]
        
        # Generate embeddings
        logger.info("Generating embeddings...")
        embeddings = self.embedder.encode(documents, show_progress_bar=True)
        
        # Store in ChromaDB
        self.collection.add(
            ids=ids,
            documents=documents,
            metadatas=metadatas,
            embeddings=embeddings.tolist()
        )
        
        logger.info(f"✓ Stored {len(chunks)} chunks in vector database")
    
    def _build_bm25_index(self, chunks: List[DocumentChunk]) -> None:
        """Build BM25 index for keyword search."""
        self.document_chunks = chunks
        
        # Tokenize documents for BM25
        tokenized_docs = [
            self._tokenize(chunk.text)
            for chunk in chunks
        ]
        
        self.bm25_index = BM25Okapi(tokenized_docs)
        logger.info(f"✓ Built BM25 index with {len(chunks)} documents")
    
    @staticmethod
    def _tokenize(text: str) -> List[str]:
        """Simple tokenization for BM25."""
        # Convert to lowercase and split on non-alphanumeric
        tokens = re.findall(r'\b\w+\b', text.lower())
        return tokens
    
    async def enhance_query(self, query: str) -> str:
        """
        Enhance the user query for better retrieval.
        
        Techniques:
        - Query expansion: Generate related search terms
        - Step-back prompting: Generate broader conceptual query
        """
        enhancement_prompt = f"""Given the user query: "{query}"

Generate an enhanced version of this query that would be better for searching through a CV/resume.
The enhanced query should:
1. Expand abbreviations and technical terms
2. Add relevant synonyms
3. Make vague queries more specific

Return ONLY the enhanced query, nothing else."""
        
        try:
            enhanced = await self.llm_handler.generate_response(
                query=enhancement_prompt,
                system_prompt="You are a query enhancement expert. Return only the enhanced query."
            )
            
            # Clean up the response
            enhanced = enhanced.strip().strip('"').strip("'")
            logger.info(f"Query enhancement: '{query}' → '{enhanced}'")
            return enhanced
            
        except Exception as e:
            logger.warning(f"Query enhancement failed: {e}, using original query")
            return query
    
    def hybrid_search(
        self,
        query: str,
        alpha: float = 0.5
    ) -> List[Tuple[DocumentChunk, float]]:
        """
        Hybrid search combining BM25 (keyword) and vector similarity.
        
        Args:
            query: Search query
            alpha: Weight for vector search (1-alpha for BM25)
                   0.0 = pure BM25, 1.0 = pure vector
        
        Returns:
            List of (chunk, score) tuples
        """
        # Vector search
        vector_results = self.collection.query(
            query_texts=[query],
            n_results=min(self.top_k_retrieval, self.collection.count())
        )
        
        # BM25 search
        tokenized_query = self._tokenize(query)
        bm25_scores = self.bm25_index.get_scores(tokenized_query)
        
        # Normalize scores to [0, 1]
        vector_scores_dict = {}
        if vector_results['ids'][0]:
            # ChromaDB returns distances, convert to similarities
            for chunk_id, distance in zip(
                vector_results['ids'][0],
                vector_results['distances'][0]
            ):
                # Cosine distance to similarity: similarity = 1 - distance
                similarity = 1 - distance
                vector_scores_dict[chunk_id] = similarity
        
        # Normalize BM25 scores
        if bm25_scores.max() > 0:
            bm25_scores = bm25_scores / bm25_scores.max()
        
        # Combine scores
        combined_results = {}
        
        for i, chunk in enumerate(self.document_chunks):
            vector_score = vector_scores_dict.get(chunk.chunk_id, 0.0)
            bm25_score = bm25_scores[i]
            
            # Weighted combination
            combined_score = alpha * vector_score + (1 - alpha) * bm25_score
            
            chunk.score = combined_score
            combined_results[chunk.chunk_id] = (chunk, combined_score)
        
        # Sort by combined score
        sorted_results = sorted(
            combined_results.values(),
            key=lambda x: x[1],
            reverse=True
        )
        
        # Return top K
        top_results = sorted_results[:self.top_k_retrieval]
        
        logger.info(
            f"Hybrid search: Retrieved {len(top_results)} chunks "
            f"(α={alpha}, vector+BM25)"
        )
        
        return top_results
    
    def rerank_results(
        self,
        query: str,
        results: List[Tuple[DocumentChunk, float]]
    ) -> List[Tuple[DocumentChunk, float]]:
        """
        Re-rank results using cross-encoder.
        
        Cross-encoders are more accurate than bi-encoders for ranking
        because they directly compute relevance scores.
        """
        if not results:
            return results
        
        # Prepare query-document pairs
        pairs = [[query, chunk.text] for chunk, _ in results]
        
        # Get re-ranking scores
        rerank_scores = self.reranker.predict(pairs)
        
        # Update chunks with new scores
        reranked_results = []
        for (chunk, _), new_score in zip(results, rerank_scores):
            chunk.score = float(new_score)
            reranked_results.append((chunk, float(new_score)))
        
        # Sort by new scores
        reranked_results.sort(key=lambda x: x[1], reverse=True)
        
        # Return top K after re-ranking
        top_reranked = reranked_results[:self.top_k_rerank]
        
        logger.info(
            f"Re-ranking: Reduced from {len(results)} to "
            f"{len(top_reranked)} chunks using cross-encoder"
        )
        
        return top_reranked
    
    async def query(
        self,
        question: str,
        use_query_enhancement: bool = True,
        use_hybrid_search: bool = True,
        use_reranking: bool = True
    ) -> Dict:
        """
        Query the RAG system with full pipeline.
        
        Args:
            question: User's question
            use_query_enhancement: Whether to enhance the query
            use_hybrid_search: Whether to use hybrid search (vs pure vector)
            use_reranking: Whether to re-rank results
        
        Returns:
            Dictionary with answer, sources, and metadata
        """
        logger.info(f"Processing query: '{question}'")
        
        # Step 1: Query Enhancement
        search_query = question
        if use_query_enhancement:
            search_query = await self.enhance_query(question)
        
        # Step 2: Hybrid Retrieval
        if use_hybrid_search:
            retrieved_chunks = self.hybrid_search(search_query, alpha=0.7)
        else:
            # Pure vector search
            results = self.collection.query(
                query_texts=[search_query],
                n_results=self.top_k_retrieval
            )
            retrieved_chunks = [
                (
                    self.document_chunks[
                        next(
                            i for i, c in enumerate(self.document_chunks)
                            if c.chunk_id == chunk_id
                        )
                    ],
                    1.0
                )
                for chunk_id in results['ids'][0]
            ]
        
        # Step 3: Re-ranking
        if use_reranking:
            final_chunks = self.rerank_results(question, retrieved_chunks)
        else:
            final_chunks = retrieved_chunks[:self.top_k_rerank]
        
        # Step 4: Generate Answer with Context
        context = self._build_context(final_chunks)
        answer = await self._generate_answer(question, context)
        
        # Step 5: Extract Sources
        sources = self._extract_sources(final_chunks)
        
        return {
            "answer": answer,
            "sources": sources,
            "num_chunks_retrieved": len(retrieved_chunks),
            "num_chunks_used": len(final_chunks),
            "query_enhanced": search_query != question,
            "enhanced_query": search_query if search_query != question else None
        }
    
    def _build_context(self, chunks: List[Tuple[DocumentChunk, float]]) -> str:
        """Build context string from chunks."""
        context_parts = []
        
        for i, (chunk, score) in enumerate(chunks, 1):
            context_parts.append(
                f"[Context {i} - Page {chunk.page_number} - Relevance: {score:.2f}]\n"
                f"{chunk.text}\n"
            )
        
        return "\n".join(context_parts)
    
    async def _generate_answer(self, question: str, context: str) -> str:
        """Generate answer using LLM with retrieved context."""
        prompt = f"""Based on the following context from a CV/resume, answer the question.

Context:
{context}

Question: {question}

Instructions:
- Answer based ONLY on the provided context
- If the context doesn't contain relevant information, say so
- Be specific and cite which parts of the CV you're referencing
- Keep the answer concise but complete

Answer:"""
        
        answer = await self.llm_handler.generate_response(
            query=prompt,
            system_prompt="You are a helpful assistant that answers questions about a person's CV/resume based on provided context."
        )
        
        return answer.strip()
    
    def _extract_sources(
        self,
        chunks: List[Tuple[DocumentChunk, float]]
    ) -> List[Dict]:
        """Extract source citations from chunks."""
        sources = []
        
        for chunk, score in chunks:
            source = {
                "page": chunk.page_number,
                "chunk_id": chunk.chunk_id,
                "confidence": round(float(score), 3),
                "preview": chunk.text[:150] + "..." if len(chunk.text) > 150 else chunk.text
            }
            sources.append(source)
        
        return sources
    
    def get_stats(self) -> Dict:
        """Get statistics about the RAG system."""
        return {
            "total_chunks": len(self.document_chunks),
            "vector_db_count": self.collection.count(),
            "embedding_model": self.embedder.get_sentence_embedding_dimension(),
            "chunk_size": self.chunk_size,
            "chunk_overlap": self.chunk_overlap
        }


# Example usage
if __name__ == "__main__":
    import asyncio
    from dotenv import load_dotenv
    
    load_dotenv()
    
    async def test():
        # Initialize LLM
        llm = LLMHandler(
            provider="groq",
            groq_api_key=os.getenv("GROQ_API_KEY")
        )
        
        # Initialize RAG
        rag = AdvancedRAGEngine(llm_handler=llm)
        
        # Load CV (you need to provide a PDF)
        # rag.load_pdf("data/cv.pdf")
        
        # Query
        # result = await rag.query("What are the candidate's main skills?")
        # print(result)
    
    asyncio.run(test())
