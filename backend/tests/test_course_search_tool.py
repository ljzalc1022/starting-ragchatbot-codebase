from unittest.mock import MagicMock
from vector_store import SearchResults
from search_tools import CourseSearchTool


def _make_tool(search_return=None):
    store = MagicMock()
    if search_return is not None:
        store.search.return_value = search_return
    tool = CourseSearchTool(store)
    return tool, store


def _make_results(docs, metas, distances=None):
    return SearchResults(
        documents=docs,
        metadata=metas,
        distances=distances or [0.1] * len(docs),
    )


# 1. Successful search → formatted string contains header and doc text
def test_successful_search_format():
    results = _make_results(
        ["Content about Python"], [{"course_title": "Python 101", "lesson_number": 1}]
    )
    tool, store = _make_tool(search_return=results)
    store.get_lesson_link.return_value = "http://example.com/lesson1"

    output = tool.execute(query="python basics")
    assert "[Python 101 - Lesson 1]" in output
    assert "Content about Python" in output


# 2. Empty results, no filter → "No relevant content found."
def test_empty_results_no_filter():
    tool, store = _make_tool(search_return=_make_results([], []))
    output = tool.execute(query="something")
    assert output == "No relevant content found."


# 3. Empty results, course filter → message contains course name
def test_empty_results_with_course_filter():
    tool, store = _make_tool(search_return=_make_results([], []))
    output = tool.execute(query="something", course_name="MCP Course")
    assert "MCP Course" in output


# 4. Empty results, lesson filter → message contains lesson number
def test_empty_results_with_lesson_filter():
    tool, store = _make_tool(search_return=_make_results([], []))
    output = tool.execute(query="something", lesson_number=3)
    assert "3" in output


# 5. results.error set → returns error string before is_empty() check
def test_error_returned_before_is_empty():
    error_results = SearchResults(documents=[], metadata=[], distances=[], error="DB error")
    tool, store = _make_tool(search_return=error_results)
    output = tool.execute(query="something")
    assert output == "DB error"


# 6. course_name passed through to store.search() correctly
def test_course_name_passed_to_store():
    tool, store = _make_tool(search_return=_make_results([], []))
    tool.execute(query="hello", course_name="MCP")
    store.search.assert_called_once_with(query="hello", course_name="MCP", lesson_number=None)


# 7. lesson_number passed through to store.search() correctly
def test_lesson_number_passed_to_store():
    tool, store = _make_tool(search_return=_make_results([], []))
    tool.execute(query="hello", lesson_number=2)
    store.search.assert_called_once_with(query="hello", course_name=None, lesson_number=2)


# 8. last_sources populated: label = "Course - Lesson N", url from get_lesson_link()
def test_last_sources_populated():
    results = _make_results(
        ["Content"], [{"course_title": "ML Basics", "lesson_number": 2}]
    )
    tool, store = _make_tool(search_return=results)
    store.get_lesson_link.return_value = "http://example.com/lesson2"

    tool.execute(query="neural networks")
    assert len(tool.last_sources) == 1
    assert tool.last_sources[0]["label"] == "ML Basics - Lesson 2"
    assert tool.last_sources[0]["url"] == "http://example.com/lesson2"


# 9. last_sources url is None when lesson_number absent from metadata
def test_last_sources_url_none_without_lesson():
    results = _make_results(
        ["Content"], [{"course_title": "ML Basics"}]  # no lesson_number
    )
    tool, store = _make_tool(search_return=results)

    tool.execute(query="something")
    assert tool.last_sources[0]["url"] is None


# 10. get_lesson_link called with correct args (course_title, lesson_num)
def test_get_lesson_link_called_with_correct_args():
    results = _make_results(
        ["Content"], [{"course_title": "Deep Learning", "lesson_number": 5}]
    )
    tool, store = _make_tool(search_return=results)
    store.get_lesson_link.return_value = None

    tool.execute(query="something")
    store.get_lesson_link.assert_called_once_with("Deep Learning", 5)
