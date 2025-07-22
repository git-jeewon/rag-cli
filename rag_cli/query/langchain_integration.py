"""LangChain integration for RAG question answering."""

import logging
import time
from typing import List, Optional, Dict, Any, Union
from dataclasses import dataclass
from enum import Enum

from rag_cli.query.processor import ProcessedQuery, QueryType
from config.config import config

logger = logging.getLogger(__name__)


class LLMProvider(Enum):
    """Supported LLM providers."""
    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    MOCK = "mock"  # For testing without API calls


@dataclass
class LLMResponse:
    """Response from an LLM."""
    
    content: str
    provider: LLMProvider
    model_name: str
    prompt_tokens: Optional[int] = None
    completion_tokens: Optional[int] = None
    total_tokens: Optional[int] = None
    response_time: float = 0.0
    metadata: Dict[str, Any] = None
    
    @property
    def cost_estimate(self) -> Optional[float]:
        """Estimate cost based on token usage."""
        if not (self.prompt_tokens and self.completion_tokens):
            return None
        
        # Rough cost estimates (as of 2024, may vary)
        cost_per_1k_tokens = {
            ("openai", "gpt-4"): {"prompt": 0.03, "completion": 0.06},
            ("openai", "gpt-3.5-turbo"): {"prompt": 0.001, "completion": 0.002},
            ("anthropic", "claude-3-opus"): {"prompt": 0.015, "completion": 0.075},
            ("anthropic", "claude-3-sonnet"): {"prompt": 0.003, "completion": 0.015},
        }
        
        key = (self.provider.value, self.model_name)
        if key in cost_per_1k_tokens:
            rates = cost_per_1k_tokens[key]
            prompt_cost = (self.prompt_tokens / 1000) * rates["prompt"]
            completion_cost = (self.completion_tokens / 1000) * rates["completion"]
            return prompt_cost + completion_cost
        
        return None


class LangChainRAG:
    """LangChain-based RAG system for question answering."""
    
    def __init__(self, 
                 provider: str = "openai",
                 model_name: Optional[str] = None,
                 temperature: float = 0.1,
                 max_tokens: Optional[int] = None):
        """
        Initialize LangChain RAG system.
        
        Args:
            provider: LLM provider ("openai" or "anthropic")
            model_name: Specific model to use
            temperature: Sampling temperature (0-1)
            max_tokens: Maximum tokens in response
        """
        self.provider = LLMProvider(provider.lower())
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.logger = logging.getLogger(__name__)
        
        # Set default model names
        if model_name is None:
            default_models = {
                LLMProvider.OPENAI: "gpt-3.5-turbo",
                LLMProvider.ANTHROPIC: "claude-3-sonnet-20240229",
                LLMProvider.MOCK: "mock-model"
            }
            model_name = default_models[self.provider]
        
        self.model_name = model_name
        self.llm = self._initialize_llm()
    
    def _initialize_llm(self):
        """Initialize the LangChain LLM."""
        try:
            if self.provider == LLMProvider.OPENAI:
                return self._initialize_openai()
            elif self.provider == LLMProvider.ANTHROPIC:
                return self._initialize_anthropic()
            elif self.provider == LLMProvider.MOCK:
                return self._initialize_mock()
            else:
                raise ValueError(f"Unsupported provider: {self.provider}")
        
        except ImportError as e:
            raise ImportError(
                f"Required packages not installed for {self.provider.value}. "
                f"Install with: pip install langchain-{self.provider.value}"
            ) from e
    
    def _initialize_openai(self):
        """Initialize OpenAI LLM."""
        try:
            from langchain_openai import ChatOpenAI
        except ImportError:
            raise ImportError("Install with: pip install langchain-openai")
        
        api_key = config.openai.api_key
        if not api_key or api_key == "your-openai-api-key":
            raise ValueError(
                "OpenAI API key not configured. Set OPENAI_API_KEY in your .env file"
            )
        
        return ChatOpenAI(
            model=self.model_name,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            api_key=api_key
        )
    
    def _initialize_anthropic(self):
        """Initialize Anthropic LLM."""
        try:
            from langchain_anthropic import ChatAnthropic
        except ImportError:
            raise ImportError("Install with: pip install langchain-anthropic")
        
        # For now, we'll add Anthropic config if needed
        # You can add ANTHROPIC_API_KEY to config if you want to use Claude
        api_key = getattr(config, 'anthropic_api_key', None)
        if not api_key:
            raise ValueError(
                "Anthropic API key not configured. Add to your .env file"
            )
        
        return ChatAnthropic(
            model=self.model_name,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            api_key=api_key
        )
    
    def _initialize_mock(self):
        """Initialize mock LLM for testing."""
        class MockLLM:
            def invoke(self, messages):
                # Extract the user message
                user_content = ""
                for msg in messages:
                    if hasattr(msg, 'content'):
                        user_content = msg.content
                        break
                
                # Generate a simple mock response
                if "summarize" in user_content.lower():
                    response = "**Summary**: This is a mock summary of the provided documents. The key points include various topics from YouTube videos, Twitter posts, and Reddit discussions related to the user's query."
                elif "compare" in user_content.lower():
                    response = "**Comparison**: Mock comparison shows similarities and differences between the documents. Document 1 discusses X while Document 2 focuses on Y."
                else:
                    response = "Based on the provided documents, here's a mock answer to your question. This response demonstrates how the RAG system would work with a real LLM."
                
                class MockResponse:
                    def __init__(self, content):
                        self.content = content
                        self.response_metadata = {
                            'token_usage': {
                                'prompt_tokens': len(user_content) // 4,
                                'completion_tokens': len(response) // 4,
                                'total_tokens': (len(user_content) + len(response)) // 4
                            }
                        }
                
                return MockResponse(response)
        
        return MockLLM()
    
    def answer_question(self, processed_query: ProcessedQuery) -> LLMResponse:
        """
        Answer a question using the processed query and context.
        
        Args:
            processed_query: Processed query with context and formatting
            
        Returns:
            LLMResponse with the answer
        """
        start_time = time.time()
        
        self.logger.info(
            f"Answering {processed_query.query_type.value} query with {self.provider.value}"
        )
        
        try:
            # Create messages for the LLM
            messages = self._create_messages(processed_query)
            
            # Get response from LLM
            response = self.llm.invoke(messages)
            
            response_time = time.time() - start_time
            
            # Extract token usage if available
            token_usage = getattr(response, 'response_metadata', {}).get('token_usage', {})
            
            llm_response = LLMResponse(
                content=response.content,
                provider=self.provider,
                model_name=self.model_name,
                prompt_tokens=token_usage.get('prompt_tokens'),
                completion_tokens=token_usage.get('completion_tokens'),
                total_tokens=token_usage.get('total_tokens'),
                response_time=response_time,
                metadata={
                    'query_type': processed_query.query_type.value,
                    'documents_used': len(processed_query.context_documents),
                    'context_length': processed_query.processing_metadata.get('context_length'),
                    'temperature': self.temperature
                }
            )
            
            self.logger.info(
                f"Generated response in {response_time:.2f}s "
                f"({token_usage.get('total_tokens', 0)} tokens)"
            )
            
            return llm_response
        
        except Exception as e:
            self.logger.error(f"Error generating response: {e}")
            raise
    
    def _create_messages(self, processed_query: ProcessedQuery) -> List:
        """Create messages for the LLM."""
        try:
            from langchain_core.messages import SystemMessage, HumanMessage
        except ImportError:
            raise ImportError("Install with: pip install langchain-core")
        
        messages = [
            SystemMessage(content=processed_query.system_prompt),
            HumanMessage(content=processed_query.formatted_prompt)
        ]
        
        return messages
    
    def answer_with_conversation_history(self,
                                       processed_query: ProcessedQuery,
                                       conversation_history: List[Dict[str, str]]) -> LLMResponse:
        """
        Answer a question with conversation history for follow-up context.
        
        Args:
            processed_query: Current processed query
            conversation_history: Previous conversation turns
            
        Returns:
            LLMResponse with context-aware answer
        """
        try:
            from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
        except ImportError:
            raise ImportError("Install with: pip install langchain-core")
        
        messages = [SystemMessage(content=processed_query.system_prompt)]
        
        # Add conversation history
        for turn in conversation_history[-5:]:  # Keep last 5 turns
            if turn.get('user'):
                messages.append(HumanMessage(content=turn['user']))
            if turn.get('assistant'):
                messages.append(AIMessage(content=turn['assistant']))
        
        # Add current query
        messages.append(HumanMessage(content=processed_query.formatted_prompt))
        
        start_time = time.time()
        response = self.llm.invoke(messages)
        response_time = time.time() - start_time
        
        token_usage = getattr(response, 'response_metadata', {}).get('token_usage', {})
        
        return LLMResponse(
            content=response.content,
            provider=self.provider,
            model_name=self.model_name,
            prompt_tokens=token_usage.get('prompt_tokens'),
            completion_tokens=token_usage.get('completion_tokens'),
            total_tokens=token_usage.get('total_tokens'),
            response_time=response_time,
            metadata={
                'query_type': processed_query.query_type.value,
                'conversation_turns': len(conversation_history),
                'has_history': len(conversation_history) > 0
            }
        )
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the current model configuration."""
        return {
            'provider': self.provider.value,
            'model_name': self.model_name,
            'temperature': self.temperature,
            'max_tokens': self.max_tokens,
            'available': self.llm is not None
        }


def create_rag_llm(provider: Optional[str] = None, **kwargs) -> LangChainRAG:
    """
    Factory function to create a RAG LLM instance.
    
    Args:
        provider: LLM provider to use
        **kwargs: Additional configuration options
        
    Returns:
        Configured LangChainRAG instance
    """
    # Use config defaults if not specified
    if provider is None:
        provider = getattr(config.openai, 'default_provider', 'mock')
    
    return LangChainRAG(provider=provider, **kwargs)


# For easy imports
__all__ = ['LangChainRAG', 'LLMResponse', 'LLMProvider', 'create_rag_llm'] 