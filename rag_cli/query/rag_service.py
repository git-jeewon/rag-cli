"""Main RAG service that orchestrates retrieval, processing, and generation."""

import logging
from typing import List, Optional, Dict, Any, Tuple
from dataclasses import dataclass
from datetime import datetime, timedelta

from rag_cli.query.retrieval import context_retriever, RetrievalResult
from rag_cli.query.processor import query_processor, QueryType
from rag_cli.query.langchain_integration import create_rag_llm, LLMResponse, LLMProvider

logger = logging.getLogger(__name__)


@dataclass
class RAGResponse:
    """Complete response from the RAG system."""
    
    query: str
    answer: str
    query_type: QueryType
    documents_used: List[Dict[str, Any]]
    retrieval_time: float
    processing_time: float
    generation_time: float
    total_time: float
    token_usage: Optional[Dict[str, int]] = None
    cost_estimate: Optional[float] = None
    metadata: Dict[str, Any] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            'query': self.query,
            'answer': self.answer,
            'query_type': self.query_type.value,
            'documents_used': self.documents_used,
            'timing': {
                'retrieval_time': self.retrieval_time,
                'processing_time': self.processing_time,
                'generation_time': self.generation_time,
                'total_time': self.total_time
            },
            'token_usage': self.token_usage,
            'cost_estimate': self.cost_estimate,
            'metadata': self.metadata or {}
        }


@dataclass
class ConversationTurn:
    """A single turn in a conversation."""
    
    user_query: str
    assistant_response: str
    timestamp: datetime
    documents_used: List[str]
    query_type: QueryType
    metadata: Dict[str, Any] = None


class ConversationManager:
    """Manages conversation history and context."""
    
    def __init__(self, max_history: int = 10):
        """Initialize conversation manager."""
        self.max_history = max_history
        self.conversation_history: List[ConversationTurn] = []
        self.logger = logging.getLogger(__name__)
    
    def add_turn(self, 
                 user_query: str, 
                 assistant_response: str,
                 documents_used: List[str],
                 query_type: QueryType,
                 metadata: Optional[Dict[str, Any]] = None):
        """Add a conversation turn."""
        turn = ConversationTurn(
            user_query=user_query,
            assistant_response=assistant_response,
            timestamp=datetime.now(),
            documents_used=documents_used,
            query_type=query_type,
            metadata=metadata or {}
        )
        
        self.conversation_history.append(turn)
        
        # Keep only recent history
        if len(self.conversation_history) > self.max_history:
            self.conversation_history = self.conversation_history[-self.max_history:]
    
    def get_recent_context(self, num_turns: int = 3) -> List[Dict[str, str]]:
        """Get recent conversation context for LLM."""
        recent_turns = self.conversation_history[-num_turns:]
        return [
            {
                'user': turn.user_query,
                'assistant': turn.assistant_response
            }
            for turn in recent_turns
        ]
    
    def clear_history(self):
        """Clear conversation history."""
        self.conversation_history.clear()
    
    def get_conversation_summary(self) -> Dict[str, Any]:
        """Get summary of conversation statistics."""
        if not self.conversation_history:
            return {'total_turns': 0}
        
        query_types = [turn.query_type.value for turn in self.conversation_history]
        unique_documents = set()
        for turn in self.conversation_history:
            unique_documents.update(turn.documents_used)
        
        return {
            'total_turns': len(self.conversation_history),
            'query_types_used': list(set(query_types)),
            'unique_documents_referenced': len(unique_documents),
            'conversation_start': self.conversation_history[0].timestamp,
            'last_interaction': self.conversation_history[-1].timestamp
        }


class RAGService:
    """Main RAG service for question answering."""
    
    def __init__(self, 
                 llm_provider: str = "mock",
                 model_name: Optional[str] = None,
                 temperature: float = 0.1):
        """
        Initialize RAG service.
        
        Args:
            llm_provider: LLM provider to use
            model_name: Specific model name
            temperature: LLM temperature setting
        """
        self.retriever = context_retriever
        self.processor = query_processor
        self.llm = create_rag_llm(
            provider=llm_provider,
            model_name=model_name,
            temperature=temperature
        )
        self.conversation_manager = ConversationManager()
        self.logger = logging.getLogger(__name__)
        
        self.logger.info(
            f"RAG service initialized with {llm_provider} provider"
        )
    
    def ask_question(self,
                    question: str,
                    context_limit: int = 5,
                    similarity_threshold: float = 0.1,
                    content_types: Optional[List[str]] = None,
                    sources: Optional[List[str]] = None,
                    query_type: Optional[QueryType] = None,
                    use_conversation_context: bool = True) -> RAGResponse:
        """
        Ask a question and get an AI-powered answer.
        
        Args:
            question: The question to answer
            context_limit: Maximum number of context documents
            similarity_threshold: Minimum similarity for relevant documents
            content_types: Filter by content types
            sources: Filter by source names
            query_type: Specify query type (auto-detected if None)
            use_conversation_context: Whether to use conversation history
            
        Returns:
            RAGResponse with answer and metadata
        """
        import time
        start_time = time.time()
        
        self.logger.info(f"Processing question: '{question}'")
        
        # Step 1: Retrieve relevant context
        retrieval_start = time.time()
        retrieval_result = self.retriever.retrieve_context(
            query=question,
            limit=context_limit,
            similarity_threshold=similarity_threshold,
            content_types=content_types,
            sources=sources
        )
        retrieval_time = time.time() - retrieval_start
        
        # Step 2: Process query and format for LLM
        processing_start = time.time()
        processed_query = self.processor.process_query(
            query=question,
            retrieval_result=retrieval_result,
            query_type=query_type
        )
        processing_time = time.time() - processing_start
        
        # Step 3: Generate answer with LLM
        generation_start = time.time()
        if use_conversation_context and self.conversation_manager.conversation_history:
            conversation_context = self.conversation_manager.get_recent_context()
            llm_response = self.llm.answer_with_conversation_history(
                processed_query, conversation_context
            )
        else:
            llm_response = self.llm.answer_question(processed_query)
        generation_time = time.time() - generation_start
        
        total_time = time.time() - start_time
        
        # Prepare document info for response
        documents_used = [
            {
                'id': doc.document_id,
                'title': doc.title,
                'content_type': doc.content_type,
                'similarity_score': doc.similarity_score,
                'source': doc.source_name,
                'url': doc.url
            }
            for doc in retrieval_result.documents
        ]
        
        # Create response
        rag_response = RAGResponse(
            query=question,
            answer=llm_response.content,
            query_type=processed_query.query_type,
            documents_used=documents_used,
            retrieval_time=retrieval_time,
            processing_time=processing_time,
            generation_time=generation_time,
            total_time=total_time,
            token_usage={
                'prompt_tokens': llm_response.prompt_tokens,
                'completion_tokens': llm_response.completion_tokens,
                'total_tokens': llm_response.total_tokens
            } if llm_response.total_tokens else None,
            cost_estimate=llm_response.cost_estimate,
            metadata={
                'llm_provider': llm_response.provider.value,
                'model_name': llm_response.model_name,
                'documents_found': len(retrieval_result.documents),
                'context_length': processed_query.processing_metadata.get('context_length'),
                'similarity_threshold': similarity_threshold
            }
        )
        
        # Add to conversation history
        document_ids = [doc.document_id for doc in retrieval_result.documents]
        self.conversation_manager.add_turn(
            user_query=question,
            assistant_response=llm_response.content,
            documents_used=document_ids,
            query_type=processed_query.query_type,
            metadata={
                'total_time': total_time,
                'documents_count': len(document_ids)
            }
        )
        
        self.logger.info(
            f"Question answered in {total_time:.2f}s "
            f"(retrieval: {retrieval_time:.2f}s, "
            f"processing: {processing_time:.2f}s, "
            f"generation: {generation_time:.2f}s)"
        )
        
        return rag_response
    
    def summarize_content(self,
                         topic: str,
                         content_types: Optional[List[str]] = None,
                         sources: Optional[List[str]] = None,
                         limit: int = 10) -> RAGResponse:
        """
        Summarize content on a specific topic.
        
        Args:
            topic: Topic to summarize
            content_types: Filter by content types
            sources: Filter by sources
            limit: Maximum documents to include
            
        Returns:
            RAGResponse with summary
        """
        summary_query = f"Summarize all information about: {topic}"
        return self.ask_question(
            question=summary_query,
            context_limit=limit,
            similarity_threshold=0.05,  # Lower threshold for broader content
            content_types=content_types,
            sources=sources,
            query_type=QueryType.SUMMARIZATION,
            use_conversation_context=False
        )
    
    def compare_sources(self,
                       topic: str,
                       sources: List[str],
                       limit: int = 5) -> RAGResponse:
        """
        Compare information across different sources.
        
        Args:
            topic: Topic to compare
            sources: Source names to compare
            limit: Documents per source
            
        Returns:
            RAGResponse with comparison
        """
        comparison_query = f"Compare information about {topic} across different sources"
        return self.ask_question(
            question=comparison_query,
            context_limit=limit * len(sources),
            similarity_threshold=0.1,
            sources=sources,
            query_type=QueryType.COMPARISON,
            use_conversation_context=False
        )
    
    def get_recent_insights(self,
                           topic: str,
                           days: int = 30,
                           limit: int = 5) -> RAGResponse:
        """
        Get recent insights on a topic.
        
        Args:
            topic: Topic to analyze
            days: Number of recent days
            limit: Maximum documents
            
        Returns:
            RAGResponse with recent insights
        """
        recent_result = self.retriever.get_recent_context(
            query=topic,
            days=days,
            limit=limit
        )
        
        processed_query = self.processor.process_query(
            query=f"What are the recent insights about {topic}?",
            retrieval_result=recent_result,
            query_type=QueryType.ANALYSIS
        )
        
        import time
        start_time = time.time()
        llm_response = self.llm.answer_question(processed_query)
        total_time = time.time() - start_time
        
        return RAGResponse(
            query=f"Recent insights about {topic}",
            answer=llm_response.content,
            query_type=QueryType.ANALYSIS,
            documents_used=[
                {
                    'id': doc.document_id,
                    'title': doc.title,
                    'content_type': doc.content_type,
                    'similarity_score': doc.similarity_score,
                    'published_at': doc.published_at.isoformat() if doc.published_at else None
                }
                for doc in recent_result.documents
            ],
            retrieval_time=recent_result.retrieval_time,
            processing_time=0,
            generation_time=llm_response.response_time,
            total_time=total_time,
            token_usage={
                'total_tokens': llm_response.total_tokens
            } if llm_response.total_tokens else None,
            metadata={
                'date_range_days': days,
                'recent_documents': len(recent_result.documents)
            }
        )
    
    def clear_conversation(self):
        """Clear conversation history."""
        self.conversation_manager.clear_history()
        self.logger.info("Conversation history cleared")
    
    def get_conversation_summary(self) -> Dict[str, Any]:
        """Get conversation statistics."""
        return self.conversation_manager.get_conversation_summary()
    
    def get_system_status(self) -> Dict[str, Any]:
        """Get overall system status and capabilities."""
        llm_info = self.llm.get_model_info()
        
        return {
            'rag_service': {
                'available': True,
                'components': ['retrieval', 'processing', 'generation']
            },
            'llm': llm_info,
            'conversation': self.get_conversation_summary(),
            'capabilities': [
                'question_answering',
                'summarization', 
                'comparison',
                'analysis',
                'conversation_context'
            ]
        }


# Global RAG service instance
rag_service = RAGService() 