"""Query processing and prompt formatting for RAG."""

import logging
from typing import List, Optional, Dict, Any, Tuple
from dataclasses import dataclass
from datetime import datetime
from enum import Enum

from rag_cli.query.retrieval import RetrievalResult, RetrievedDocument

logger = logging.getLogger(__name__)


class QueryType(Enum):
    """Types of queries the system can handle."""
    QUESTION_ANSWERING = "question_answering"
    SUMMARIZATION = "summarization"
    COMPARISON = "comparison"
    ANALYSIS = "analysis"
    SEARCH = "search"


@dataclass
class ProcessedQuery:
    """A processed query ready for LLM."""
    
    original_query: str
    query_type: QueryType
    formatted_prompt: str
    context_documents: List[RetrievedDocument]
    system_prompt: str
    retrieval_metadata: Dict[str, Any]
    processing_metadata: Dict[str, Any]


class PromptTemplate:
    """Templates for different types of RAG prompts."""
    
    SYSTEM_PROMPTS = {
        QueryType.QUESTION_ANSWERING: """You are a helpful AI assistant with access to a knowledge base of documents from YouTube videos, Twitter posts, Reddit discussions, and other sources. Your job is to answer questions accurately based on the provided context.

Guidelines:
- Answer questions directly and concisely based on the provided context
- If information is not in the context, say "I don't have information about that in the provided documents"
- Always cite which document(s) your answer comes from
- If multiple sources have conflicting information, acknowledge this
- Provide specific details when available (dates, numbers, quotes)
- Be conversational but informative""",

        QueryType.SUMMARIZATION: """You are an expert at analyzing and summarizing content from various sources. Your job is to create clear, comprehensive summaries of the provided documents.

Guidelines:
- Create structured summaries that capture key points
- Group related information together
- Highlight important insights or patterns across sources
- Maintain the original meaning and context
- Note the source types (YouTube, Twitter, Reddit, etc.) when relevant
- Be thorough but avoid redundancy""",

        QueryType.COMPARISON: """You are skilled at comparing and contrasting information from multiple sources. Your job is to analyze similarities, differences, and relationships between the provided documents.

Guidelines:
- Identify common themes and divergent viewpoints
- Highlight agreements and disagreements between sources
- Note the credibility and context of different sources
- Structure comparisons clearly (similarities first, then differences)
- Draw meaningful insights from the comparison
- Cite specific sources for each point""",

        QueryType.ANALYSIS: """You are an analytical AI that helps users understand complex topics by examining information from multiple angles. Your job is to provide deep insights and analysis of the provided content.

Guidelines:
- Look for patterns, trends, and underlying themes
- Provide context and background when helpful
- Connect related concepts across different sources
- Offer insights that go beyond surface-level information
- Maintain objectivity while being thorough
- Support conclusions with evidence from the documents"""
    }
    
    PROMPT_TEMPLATES = {
        QueryType.QUESTION_ANSWERING: """Based on the following documents from your knowledge base, please answer this question: {query}

Context Documents:
{context}

Question: {query}

Please provide a clear, accurate answer based on the information in these documents. If you reference specific information, please cite which document it came from.""",

        QueryType.SUMMARIZATION: """Please summarize the following documents about: {query}

Documents to Summarize:
{context}

Create a comprehensive summary that captures the key points, insights, and important details from these documents. Organize the information in a clear, logical structure.""",

        QueryType.COMPARISON: """Please compare and analyze the following documents related to: {query}

Documents to Compare:
{context}

Analyze these documents by:
1. Identifying common themes and shared perspectives
2. Highlighting differences in viewpoints or approaches
3. Noting any conflicting information
4. Drawing insights from the comparison

Please structure your response clearly with sections for similarities, differences, and your analysis.""",

        QueryType.ANALYSIS: """Please provide a detailed analysis of the following documents related to: {query}

Documents for Analysis:
{context}

Provide a thorough analysis that:
1. Identifies key themes and patterns
2. Extracts important insights and takeaways
3. Connects related concepts across sources
4. Offers deeper understanding of the topic
5. Highlights any notable trends or implications

Structure your analysis in a clear, logical format."""
    }


class QueryProcessor:
    """Processes queries and formats them for LLM consumption."""
    
    def __init__(self):
        """Initialize the query processor."""
        self.logger = logging.getLogger(__name__)
        self.prompt_templates = PromptTemplate()
    
    def process_query(self,
                     query: str,
                     retrieval_result: RetrievalResult,
                     query_type: Optional[QueryType] = None,
                     max_context_length: int = 8000,
                     include_metadata: bool = True) -> ProcessedQuery:
        """
        Process a query and format it for LLM consumption.
        
        Args:
            query: The user's query
            retrieval_result: Retrieved context documents
            query_type: Type of query (auto-detected if None)
            max_context_length: Maximum length of context to include
            include_metadata: Whether to include document metadata
            
        Returns:
            ProcessedQuery ready for LLM
        """
        # Auto-detect query type if not specified
        if query_type is None:
            query_type = self._detect_query_type(query)
        
        self.logger.info(f"Processing {query_type.value} query: '{query}'")
        
        # Format context string
        context_string = self._format_context(
            retrieval_result.documents,
            max_context_length,
            include_metadata
        )
        
        # Get appropriate templates
        system_prompt = self.prompt_templates.SYSTEM_PROMPTS[query_type]
        prompt_template = self.prompt_templates.PROMPT_TEMPLATES[query_type]
        
        # Format the main prompt
        formatted_prompt = prompt_template.format(
            query=query,
            context=context_string
        )
        
        # Add context length warning if needed
        if len(context_string) >= max_context_length:
            formatted_prompt += f"\n\nNote: Context was truncated to {max_context_length} characters. Some information may have been omitted."
        
        processing_metadata = {
            'query_type': query_type.value,
            'context_length': len(context_string),
            'max_context_length': max_context_length,
            'documents_included': len(retrieval_result.documents),
            'include_metadata': include_metadata,
            'truncated': len(context_string) >= max_context_length
        }
        
        return ProcessedQuery(
            original_query=query,
            query_type=query_type,
            formatted_prompt=formatted_prompt,
            context_documents=retrieval_result.documents,
            system_prompt=system_prompt,
            retrieval_metadata=retrieval_result.search_metadata,
            processing_metadata=processing_metadata
        )
    
    def _detect_query_type(self, query: str) -> QueryType:
        """Auto-detect the type of query based on keywords and structure."""
        query_lower = query.lower()
        
        # Summarization keywords
        summarization_keywords = [
            'summarize', 'summary', 'sum up', 'overview', 'recap',
            'what are the main points', 'key takeaways', 'highlights'
        ]
        
        # Comparison keywords
        comparison_keywords = [
            'compare', 'contrast', 'difference', 'similarities', 'versus',
            'vs', 'between', 'how do they differ', 'what are the differences'
        ]
        
        # Analysis keywords
        analysis_keywords = [
            'analyze', 'analysis', 'examine', 'evaluate', 'assess',
            'what does this mean', 'implications', 'significance',
            'patterns', 'trends', 'insights'
        ]
        
        # Check for each type
        if any(keyword in query_lower for keyword in summarization_keywords):
            return QueryType.SUMMARIZATION
        
        if any(keyword in query_lower for keyword in comparison_keywords):
            return QueryType.COMPARISON
        
        if any(keyword in query_lower for keyword in analysis_keywords):
            return QueryType.ANALYSIS
        
        # Default to question answering
        return QueryType.QUESTION_ANSWERING
    
    def _format_context(self,
                       documents: List[RetrievedDocument],
                       max_length: int,
                       include_metadata: bool) -> str:
        """Format retrieved documents into context string."""
        if not documents:
            return "No relevant documents found."
        
        context_parts = []
        current_length = 0
        
        for i, doc in enumerate(documents, 1):
            doc_context = doc.to_context_string(include_metadata)
            doc_string = f"Document {i}:\n{doc_context}\n"
            
            # Check if adding this document would exceed max length
            if current_length + len(doc_string) > max_length and context_parts:
                # Stop adding documents if we're at the limit
                break
            
            context_parts.append(doc_string)
            current_length += len(doc_string)
        
        context = "\n".join(context_parts)
        
        # If we're still over the limit, truncate the last document
        if len(context) > max_length:
            context = context[:max_length - 100] + "\n\n[Content truncated...]"
        
        return context
    
    def create_follow_up_query(self,
                              original_query: str,
                              previous_response: str,
                              new_question: str) -> str:
        """
        Create a follow-up query that includes context from previous conversation.
        
        Args:
            original_query: The original question asked
            previous_response: The AI's previous response
            new_question: The follow-up question
            
        Returns:
            Formatted follow-up query with context
        """
        follow_up_template = """Previous conversation context:

User asked: "{original_query}"
Assistant responded: "{previous_response}"

New question: {new_question}

Please answer the new question considering the previous conversation context."""
        
        return follow_up_template.format(
            original_query=original_query,
            previous_response=previous_response[:500] + "..." if len(previous_response) > 500 else previous_response,
            new_question=new_question
        )
    
    def create_specialized_prompts(self,
                                  query: str,
                                  documents: List[RetrievedDocument],
                                  prompt_type: str) -> Dict[str, str]:
        """
        Create specialized prompts for different use cases.
        
        Args:
            query: User query
            documents: Retrieved documents
            prompt_type: Type of specialized prompt
            
        Returns:
            Dictionary with system and user prompts
        """
        context = self._format_context(documents, 6000, True)
        
        specialized_prompts = {
            'learning_summary': {
                'system': "You are a learning assistant that helps users understand and remember information from their saved content. Focus on educational value and actionable insights.",
                'user': f"Based on my saved content below, help me understand and learn about: {query}\n\nContent:\n{context}\n\nPlease provide:\n1. Key concepts explained clearly\n2. Important takeaways\n3. How this applies practically\n4. Any action items or next steps"
            },
            
            'research_assistant': {
                'system': "You are a research assistant that analyzes information from multiple sources to provide comprehensive insights. Focus on accuracy, credibility, and connecting information across sources.",
                'user': f"Research question: {query}\n\nSources to analyze:\n{context}\n\nPlease provide:\n1. Comprehensive answer with evidence\n2. Source credibility assessment\n3. Gaps in information (if any)\n4. Related questions to explore further"
            },
            
            'creative_insights': {
                'system': "You are a creative thinking partner that helps users generate new ideas and connections from their collected information. Focus on innovation and novel perspectives.",
                'user': f"Topic: {query}\n\nInformation sources:\n{context}\n\nPlease help me:\n1. Identify unexpected connections or patterns\n2. Generate creative ideas or solutions\n3. Suggest innovative applications\n4. Propose interesting questions to explore"
            }
        }
        
        return specialized_prompts.get(prompt_type, {
            'system': self.prompt_templates.SYSTEM_PROMPTS[QueryType.QUESTION_ANSWERING],
            'user': self.prompt_templates.PROMPT_TEMPLATES[QueryType.QUESTION_ANSWERING].format(
                query=query, context=context
            )
        })


# Global query processor instance
query_processor = QueryProcessor() 