import pytest
from unittest.mock import MagicMock, patch
from ai_generator import AIGenerator


@pytest.fixture
def mock_client():
    with patch("ai_generator.anthropic.Anthropic") as MockAnthropic:
        client = MagicMock()
        MockAnthropic.return_value = client
        yield client


@pytest.fixture
def generator(mock_client):
    return AIGenerator(api_key="test-key", model="claude-test")


def _make_simple_response(text="Hello world"):
    resp = MagicMock()
    resp.stop_reason = "end_turn"
    block = MagicMock()
    block.text = text
    resp.content = [block]
    return resp


def _make_tool_use_response(tool_name, tool_id, tool_input):
    resp = MagicMock()
    resp.stop_reason = "tool_use"
    block = MagicMock()
    block.type = "tool_use"
    block.name = tool_name
    block.id = tool_id
    block.input = tool_input
    resp.content = [block]
    return resp


# 1. No tool use → single messages.create call, returns content[0].text
def test_no_tool_use_single_call(mock_client, generator):
    mock_client.messages.create.return_value = _make_simple_response("Direct answer")
    result = generator.generate_response(query="What is 2+2?")
    assert mock_client.messages.create.call_count == 1
    assert result == "Direct answer"


# 2. tools list in first API call kwargs; tool_choice={"type":"auto"} included
def test_tools_included_in_first_call(mock_client, generator):
    mock_client.messages.create.return_value = _make_simple_response()
    tools = [{"name": "search_course_content"}]
    generator.generate_response(query="Tell me about lesson 1", tools=tools)
    call_kwargs = mock_client.messages.create.call_args[1]
    assert call_kwargs["tools"] == tools
    assert call_kwargs["tool_choice"] == {"type": "auto"}


# 3. No tools → "tools" and "tool_choice" keys absent from call
def test_no_tools_means_no_tool_keys(mock_client, generator):
    mock_client.messages.create.return_value = _make_simple_response()
    generator.generate_response(query="General question")
    call_kwargs = mock_client.messages.create.call_args[1]
    assert "tools" not in call_kwargs
    assert "tool_choice" not in call_kwargs


# 4. stop_reason == "tool_use" → two messages.create calls total
def test_tool_use_triggers_two_api_calls(mock_client, generator):
    tool_resp = _make_tool_use_response("search_course_content", "tu_123", {"query": "lesson 1"})
    final_resp = _make_simple_response("Final answer")
    mock_client.messages.create.side_effect = [tool_resp, final_resp]

    tool_manager = MagicMock()
    tool_manager.execute_tool.return_value = "Search result text"

    tools = [{"name": "search_course_content"}]
    result = generator.generate_response(query="What is in lesson 1?", tools=tools, tool_manager=tool_manager)
    assert mock_client.messages.create.call_count == 2
    assert result == "Final answer"


# 5. tool_manager.execute_tool called with correct name and **input kwargs
def test_execute_tool_called_with_correct_args(mock_client, generator):
    tool_input = {"query": "python basics", "lesson_number": 2}
    tool_resp = _make_tool_use_response("search_course_content", "tu_456", tool_input)
    final_resp = _make_simple_response("Answer")
    mock_client.messages.create.side_effect = [tool_resp, final_resp]

    tool_manager = MagicMock()
    tool_manager.execute_tool.return_value = "results"

    generator.generate_response(query="query", tools=[{}], tool_manager=tool_manager)
    tool_manager.execute_tool.assert_called_once_with("search_course_content", **tool_input)


# 6. Tool result injected into second call messages as tool_result message
def test_tool_result_injected_into_second_call(mock_client, generator):
    tool_resp = _make_tool_use_response("search_course_content", "tu_789", {"query": "test"})
    final_resp = _make_simple_response("Answer")
    mock_client.messages.create.side_effect = [tool_resp, final_resp]

    tool_manager = MagicMock()
    tool_manager.execute_tool.return_value = "tool output"

    generator.generate_response(query="query", tools=[{}], tool_manager=tool_manager)

    second_call_kwargs = mock_client.messages.create.call_args_list[1][1]
    messages = second_call_kwargs["messages"]
    tool_result_msg = messages[-1]
    assert tool_result_msg["role"] == "user"
    assert tool_result_msg["content"][0]["type"] == "tool_result"
    assert tool_result_msg["content"][0]["tool_use_id"] == "tu_789"
    assert tool_result_msg["content"][0]["content"] == "tool output"


# 7. Second API call (intermediate) includes tools and tool_choice so Claude can search again
def test_intermediate_call_includes_tools_and_tool_choice(mock_client, generator):
    tool_resp = _make_tool_use_response("search_course_content", "tu_001", {"query": "test"})
    final_resp = _make_simple_response("Answer")
    mock_client.messages.create.side_effect = [tool_resp, final_resp]

    tool_manager = MagicMock()
    tool_manager.execute_tool.return_value = "results"

    tools = [{"name": "search_course_content"}]
    generator.generate_response(query="query", tools=tools, tool_manager=tool_manager)

    second_call_kwargs = mock_client.messages.create.call_args_list[1][1]
    assert "tools" in second_call_kwargs
    assert "tool_choice" in second_call_kwargs


# 8. Final answer is final_response.content[0].text
def test_final_answer_is_second_response_text(mock_client, generator):
    tool_resp = _make_tool_use_response("search_course_content", "tu_002", {"query": "test"})
    final_resp = _make_simple_response("The correct final answer")
    mock_client.messages.create.side_effect = [tool_resp, final_resp]

    tool_manager = MagicMock()
    tool_manager.execute_tool.return_value = "results"

    result = generator.generate_response(query="query", tools=[{}], tool_manager=tool_manager)
    assert result == "The correct final answer"


# 9. Two sequential tool calls → three total API calls
def test_two_sequential_tool_calls_makes_three_api_calls(mock_client, generator):
    tool_resp_1 = _make_tool_use_response("search_course_content", "tu_a1", {"query": "first"})
    tool_resp_2 = _make_tool_use_response("search_course_content", "tu_a2", {"query": "second"})
    end_turn_resp = _make_simple_response("Final combined answer")
    mock_client.messages.create.side_effect = [tool_resp_1, tool_resp_2, end_turn_resp]

    tool_manager = MagicMock()
    tool_manager.execute_tool.return_value = "search results"

    tools = [{"name": "search_course_content"}]
    result = generator.generate_response(query="complex query", tools=tools, tool_manager=tool_manager)

    assert mock_client.messages.create.call_count == 3
    assert tool_manager.execute_tool.call_count == 2
    assert result == "Final combined answer"


# 10. After two rounds are exhausted, the final call has no tools or tool_choice
def test_intermediate_call_after_two_rounds_has_no_tools(mock_client, generator):
    tool_resp_1 = _make_tool_use_response("search_course_content", "tu_b1", {"query": "first"})
    tool_resp_2 = _make_tool_use_response("search_course_content", "tu_b2", {"query": "second"})
    end_turn_resp = _make_simple_response("Answer after exhaustion")
    mock_client.messages.create.side_effect = [tool_resp_1, tool_resp_2, end_turn_resp]

    tool_manager = MagicMock()
    tool_manager.execute_tool.return_value = "results"

    tools = [{"name": "search_course_content"}]
    generator.generate_response(query="query", tools=tools, tool_manager=tool_manager)

    third_call_kwargs = mock_client.messages.create.call_args_list[2][1]
    assert "tools" not in third_call_kwargs
    assert "tool_choice" not in third_call_kwargs


# 11. Two rounds: messages thread correctly (5 entries, alternating roles)
def test_two_rounds_messages_thread_correctly(mock_client, generator):
    tool_resp_1 = _make_tool_use_response("search_course_content", "tu_c1", {"query": "first"})
    tool_resp_2 = _make_tool_use_response("search_course_content", "tu_c2", {"query": "second"})
    end_turn_resp = _make_simple_response("Done")
    mock_client.messages.create.side_effect = [tool_resp_1, tool_resp_2, end_turn_resp]

    tool_manager = MagicMock()
    tool_manager.execute_tool.return_value = "results"

    tools = [{"name": "search_course_content"}]
    generator.generate_response(query="complex query", tools=tools, tool_manager=tool_manager)

    third_call_kwargs = mock_client.messages.create.call_args_list[2][1]
    messages = third_call_kwargs["messages"]
    assert len(messages) == 5
    assert messages[0]["role"] == "user"       # original query
    assert messages[1]["role"] == "assistant"  # tool_use round 1
    assert messages[2]["role"] == "user"       # tool_result round 1
    assert messages[3]["role"] == "assistant"  # tool_use round 2
    assert messages[4]["role"] == "user"       # tool_result round 2


# 12. Tool error in round 1 → graceful fallback with 2 total calls
def test_tool_error_terminates_loop_graceful_fallback(mock_client, generator):
    tool_resp_1 = _make_tool_use_response("search_course_content", "tu_d1", {"query": "test"})
    graceful_resp = _make_simple_response("Graceful fallback answer")
    mock_client.messages.create.side_effect = [tool_resp_1, graceful_resp]

    tool_manager = MagicMock()
    tool_manager.execute_tool.side_effect = RuntimeError("db timeout")

    tools = [{"name": "search_course_content"}]
    result = generator.generate_response(query="query", tools=tools, tool_manager=tool_manager)

    assert mock_client.messages.create.call_count == 2
    assert result == "Graceful fallback answer"

    second_call_kwargs = mock_client.messages.create.call_args_list[1][1]
    messages = second_call_kwargs["messages"]
    tool_result_content = messages[-1]["content"][0]["content"]
    assert "db timeout" in tool_result_content


# 13. Two-round scenario: each execute_tool call uses the correct name and input
def test_both_execute_tool_calls_made_in_two_round_scenario(mock_client, generator):
    input_1 = {"query": "python basics"}
    input_2 = {"query": "advanced python"}
    tool_resp_1 = _make_tool_use_response("search_course_content", "tu_e1", input_1)
    tool_resp_2 = _make_tool_use_response("get_course_outline", "tu_e2", input_2)
    end_turn_resp = _make_simple_response("Combined answer")
    mock_client.messages.create.side_effect = [tool_resp_1, tool_resp_2, end_turn_resp]

    tool_manager = MagicMock()
    tool_manager.execute_tool.return_value = "results"

    tools = [{"name": "search_course_content"}, {"name": "get_course_outline"}]
    generator.generate_response(query="query", tools=tools, tool_manager=tool_manager)

    assert tool_manager.execute_tool.call_count == 2
    first_call = tool_manager.execute_tool.call_args_list[0]
    second_call = tool_manager.execute_tool.call_args_list[1]
    assert first_call == (("search_course_content",), input_1)
    assert second_call == (("get_course_outline",), input_2)
