import pytest
from unittest.mock import MagicMock, patch
from rag_system import RAGSystem


@pytest.fixture
def system():
    with patch("rag_system.DocumentProcessor"), \
         patch("rag_system.VectorStore"), \
         patch("rag_system.AIGenerator") as MockAIGen, \
         patch("rag_system.SessionManager") as MockSession:

        config = MagicMock()
        config.CHUNK_SIZE = 800
        config.CHUNK_OVERLAP = 100
        config.CHROMA_PATH = "/tmp/chroma"
        config.EMBEDDING_MODEL = "all-MiniLM-L6-v2"
        config.MAX_RESULTS = 5
        config.ANTHROPIC_API_KEY = "test"
        config.ANTHROPIC_MODEL = "claude-test"
        config.MAX_HISTORY = 2

        rag = RAGSystem(config)

        # Replace tool_manager with a fresh mock
        rag.tool_manager = MagicMock()
        rag.tool_manager.get_tool_definitions.return_value = [{"name": "search_course_content"}]
        rag.tool_manager.get_last_sources.return_value = [{"label": "Course A - Lesson 1", "url": "http://example.com"}]

        # Wire up the generator mock
        rag.ai_generator = MockAIGen.return_value
        rag.ai_generator.generate_response.return_value = "AI response"

        # Wire up session_manager mock
        rag.session_manager = MockSession.return_value
        rag.session_manager.get_conversation_history.return_value = "prior history"

        yield rag


# 1. Query is wrapped: generate_response receives "Answer this question about course materials: {raw_query}"
def test_query_is_wrapped(system):
    system.query("What is Python?", session_id="s1")
    call_kwargs = system.ai_generator.generate_response.call_args[1]
    assert call_kwargs["query"] == "Answer this question about course materials: What is Python?"


# 2. tools and tool_manager passed to generator
def test_tools_and_tool_manager_passed(system):
    system.query("question", session_id="s1")
    call_kwargs = system.ai_generator.generate_response.call_args[1]
    assert call_kwargs["tools"] == system.tool_manager.get_tool_definitions()
    assert call_kwargs["tool_manager"] is system.tool_manager


# 3. Sources come from tool_manager.get_last_sources()
def test_sources_from_tool_manager(system):
    _, sources = system.query("question", session_id="s1")
    assert sources == [{"label": "Course A - Lesson 1", "url": "http://example.com"}]


# 4. tool_manager.reset_sources() called after every query
def test_reset_sources_called(system):
    system.query("question", session_id="s1")
    system.tool_manager.reset_sources.assert_called_once()


# 5. session_manager.add_exchange called with raw query (not wrapped), after response
def test_add_exchange_called_with_raw_query(system):
    system.query("What is Python?", session_id="s1")
    system.session_manager.add_exchange.assert_called_once_with("s1", "What is Python?", "AI response")


# 6. Return value is (str, list)
def test_return_type(system):
    result = system.query("question", session_id="s1")
    assert isinstance(result, tuple)
    assert len(result) == 2
    assert isinstance(result[0], str)
    assert isinstance(result[1], list)


# 7. When session_id=None: history not fetched, add_exchange not called
def test_no_session_id_skips_history(system):
    system.query("question", session_id=None)
    system.session_manager.get_conversation_history.assert_not_called()
    system.session_manager.add_exchange.assert_not_called()
