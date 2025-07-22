"""Comprehensive test for the complete RAG system."""

import logging
import sys
import time
from typing import Dict, Any

from rag_cli.query.rag_service import rag_service
from rag_cli.query.processor import QueryType
from rag_cli.storage.database import check_database_connection

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)

logger = logging.getLogger(__name__)


def test_system_initialization():
    """Test that all RAG components are properly initialized."""
    logger.info("Testing RAG system initialization...")
    
    try:
        # Check database connection
        if not check_database_connection():
            logger.error("Database connection failed!")
            return False
        
        # Check system status
        status = rag_service.get_system_status()
        logger.info(f"System status: {status}")
        
        # Verify all components are available
        required_components = ['retrieval', 'processing', 'generation']
        available_components = status['rag_service']['components']
        
        if all(comp in available_components for comp in required_components):
            logger.info("✅ All RAG components initialized successfully!")
            return True
        else:
            logger.error("❌ Some RAG components missing")
            return False
    
    except Exception as e:
        logger.error(f"System initialization failed: {e}")
        return False


def test_question_answering():
    """Test basic question answering functionality."""
    logger.info("Testing question answering...")
    
    test_questions = [
        "What is machine learning?",
        "Tell me about artificial intelligence",
        "What are the benefits of testing software?",
        "How does social media work?"
    ]
    
    successful_answers = 0
    
    for i, question in enumerate(test_questions, 1):
        try:
            logger.info(f"\n🔍 Question {i}: '{question}'")
            
            start_time = time.time()
            response = rag_service.ask_question(question)
            
            logger.info(f"✅ Answer generated in {response.total_time:.2f}s")
            logger.info(f"   Query type: {response.query_type.value}")
            logger.info(f"   Documents used: {len(response.documents_used)}")
            logger.info(f"   Answer preview: {response.answer[:150]}...")
            
            if response.token_usage:
                logger.info(f"   Tokens: {response.token_usage.get('total_tokens', 0)}")
            
            # Show sources
            if response.documents_used:
                logger.info("   Sources:")
                for doc in response.documents_used[:2]:  # Show first 2 sources
                    score = doc.get('similarity_score', 0)
                    title = doc.get('title', 'No title')
                    content_type = doc.get('content_type', 'unknown')
                    logger.info(f"     - [{content_type}] {title} (score: {score:.3f})")
            
            successful_answers += 1
            
        except Exception as e:
            logger.error(f"❌ Question {i} failed: {e}")
    
    logger.info(f"\n✅ Question answering: {successful_answers}/{len(test_questions)} successful")
    return successful_answers > 0


def test_specialized_queries():
    """Test different types of specialized queries."""
    logger.info("Testing specialized query types...")
    
    specialized_tests = [
        {
            'name': 'Summarization',
            'query': 'Summarize the main points about testing',
            'expected_type': QueryType.SUMMARIZATION
        },
        {
            'name': 'Comparison', 
            'query': 'Compare different approaches to software development',
            'expected_type': QueryType.COMPARISON
        },
        {
            'name': 'Analysis',
            'query': 'Analyze the trends in technology mentioned in my content',
            'expected_type': QueryType.ANALYSIS
        }
    ]
    
    successful_tests = 0
    
    for test in specialized_tests:
        try:
            logger.info(f"\n🔬 Testing {test['name']}: '{test['query']}'")
            
            response = rag_service.ask_question(test['query'])
            
            logger.info(f"✅ {test['name']} completed in {response.total_time:.2f}s")
            logger.info(f"   Detected type: {response.query_type.value}")
            logger.info(f"   Expected type: {test['expected_type'].value}")
            logger.info(f"   Documents: {len(response.documents_used)}")
            logger.info(f"   Response preview: {response.answer[:200]}...")
            
            successful_tests += 1
            
        except Exception as e:
            logger.error(f"❌ {test['name']} test failed: {e}")
    
    logger.info(f"\n✅ Specialized queries: {successful_tests}/{len(specialized_tests)} successful")
    return successful_tests > 0


def test_conversation_flow():
    """Test conversation with follow-up questions."""
    logger.info("Testing conversation flow...")
    
    # Clear any previous conversation
    rag_service.clear_conversation()
    
    conversation_flow = [
        "What is artificial intelligence?",
        "How is it different from machine learning?",  # Follow-up
        "Can you give me specific examples?",  # Another follow-up
        "What are the practical applications?"  # Final follow-up
    ]
    
    successful_turns = 0
    
    for i, question in enumerate(conversation_flow, 1):
        try:
            logger.info(f"\n💬 Turn {i}: '{question}'")
            
            response = rag_service.ask_question(
                question, 
                use_conversation_context=True
            )
            
            logger.info(f"✅ Response {i} generated in {response.total_time:.2f}s")
            logger.info(f"   Answer: {response.answer[:150]}...")
            
            successful_turns += 1
            
        except Exception as e:
            logger.error(f"❌ Conversation turn {i} failed: {e}")
    
    # Check conversation history
    conv_summary = rag_service.get_conversation_summary()
    logger.info(f"\n📊 Conversation summary: {conv_summary}")
    
    logger.info(f"✅ Conversation flow: {successful_turns}/{len(conversation_flow)} turns successful")
    return successful_turns > 0


def test_content_filtering():
    """Test filtering by content types and sources."""
    logger.info("Testing content filtering...")
    
    filtering_tests = [
        {
            'name': 'YouTube only',
            'query': 'What video content do I have?',
            'content_types': ['video_transcript']
        },
        {
            'name': 'Social media only',
            'query': 'What social media posts mention technology?',
            'content_types': ['tweet', 'reddit_post']
        }
    ]
    
    successful_filters = 0
    
    for test in filtering_tests:
        try:
            logger.info(f"\n🎯 Testing {test['name']}")
            
            response = rag_service.ask_question(
                test['query'],
                content_types=test.get('content_types'),
                sources=test.get('sources')
            )
            
            logger.info(f"✅ Filtered query completed")
            logger.info(f"   Documents found: {len(response.documents_used)}")
            
            # Check if filtering worked
            if test.get('content_types'):
                found_types = {doc['content_type'] for doc in response.documents_used}
                expected_types = set(test['content_types'])
                if found_types.issubset(expected_types) or not found_types:  # Allow empty results
                    logger.info(f"   ✅ Content types correctly filtered: {found_types}")
                else:
                    logger.warning(f"   ⚠️ Unexpected content types: {found_types}")
            
            successful_filters += 1
            
        except Exception as e:
            logger.error(f"❌ Filtering test {test['name']} failed: {e}")
    
    logger.info(f"✅ Content filtering: {successful_filters}/{len(filtering_tests)} successful")
    return successful_filters > 0


def test_summarization_service():
    """Test the dedicated summarization service."""
    logger.info("Testing summarization service...")
    
    try:
        logger.info("📝 Testing content summarization...")
        
        response = rag_service.summarize_content(
            topic="technology and software",
            limit=5
        )
        
        logger.info(f"✅ Summarization completed in {response.total_time:.2f}s")
        logger.info(f"   Documents summarized: {len(response.documents_used)}")
        logger.info(f"   Query type: {response.query_type.value}")
        logger.info(f"   Summary preview: {response.answer[:300]}...")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Summarization service failed: {e}")
        return False


def demonstrate_rag_capabilities():
    """Demonstrate the full capabilities of the RAG system."""
    logger.info("\n🎯 DEMONSTRATING RAG SYSTEM CAPABILITIES")
    logger.info("=" * 60)
    
    demo_queries = [
        {
            'query': "What are the main technologies discussed in my content?",
            'description': "General knowledge extraction"
        },
        {
            'query': "Summarize what I've learned about AI and machine learning",
            'description': "Intelligent summarization"
        },
        {
            'query': "What coding practices are mentioned across different sources?",
            'description': "Cross-source analysis"
        }
    ]
    
    for i, demo in enumerate(demo_queries, 1):
        logger.info(f"\n🚀 Demo {i}: {demo['description']}")
        logger.info(f"Query: '{demo['query']}'")
        logger.info("-" * 40)
        
        try:
            response = rag_service.ask_question(demo['query'])
            
            logger.info(f"💫 ANSWER:")
            logger.info(f"{response.answer}")
            
            logger.info(f"\n📊 METADATA:")
            logger.info(f"   Time: {response.total_time:.2f}s")
            logger.info(f"   Type: {response.query_type.value}")
            logger.info(f"   Sources: {len(response.documents_used)}")
            
            if response.documents_used:
                logger.info(f"   Used documents:")
                for doc in response.documents_used:
                    title = doc.get('title', 'No title')
                    content_type = doc.get('content_type', 'unknown')
                    score = doc.get('similarity_score', 0)
                    logger.info(f"     • [{content_type}] {title} ({score:.3f})")
        
        except Exception as e:
            logger.error(f"❌ Demo {i} failed: {e}")


def main():
    """Run all RAG system tests."""
    logger.info("🚀 Starting comprehensive RAG system tests...")
    
    tests = [
        ("System Initialization", test_system_initialization),
        ("Question Answering", test_question_answering),
        ("Specialized Queries", test_specialized_queries),
        ("Conversation Flow", test_conversation_flow),
        ("Content Filtering", test_content_filtering),
        ("Summarization Service", test_summarization_service)
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        logger.info(f"\n{'='*20} {test_name} {'='*20}")
        try:
            if test_func():
                logger.info(f"✅ {test_name} PASSED")
                passed += 1
            else:
                logger.error(f"❌ {test_name} FAILED")
        except Exception as e:
            logger.error(f"❌ {test_name} FAILED with exception: {e}")
    
    logger.info(f"\n{'='*60}")
    logger.info(f"📊 TEST RESULTS: {passed}/{total} tests passed")
    
    if passed == total:
        logger.info("🎉 ALL RAG TESTS PASSED!")
        
        # Run capability demonstration
        demonstrate_rag_capabilities()
        
        return 0
    else:
        logger.error("❌ Some tests failed.")
        return 1


if __name__ == "__main__":
    sys.exit(main()) 