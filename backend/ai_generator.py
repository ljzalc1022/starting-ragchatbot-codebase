import anthropic
from typing import List, Optional, Dict, Any

class AIGenerator:
    """Handles interactions with Anthropic's Claude API for generating responses"""
    
    # Static system prompt to avoid rebuilding on each call
    SYSTEM_PROMPT = """ You are an AI assistant specialized in course materials and educational content with access to a comprehensive search tool for course information.

Search Tool Usage:
- Use `search_course_content` **only** for questions about specific course content or detailed educational materials
- Use `get_course_outline` **only** for questions about a course's structure, outline, syllabus, or lesson list
- **Up to two sequential tool calls per query** — use a second tool call only when the first result is insufficient to fully answer the question
- Synthesize search results into accurate, fact-based responses
- If search yields no results, state this clearly without offering alternatives
- When returning an outline, always include: course title, course link (if available), and each lesson's number and title

Response Protocol:
- **General knowledge questions**: Answer using existing knowledge without searching
- **Course-specific questions**: Search first, then answer
- **No meta-commentary**:
 - Provide direct answers only — no reasoning process, search explanations, or question-type analysis
 - Do not mention "based on the search results"


All responses must be:
1. **Brief, Concise and focused** - Get to the point quickly
2. **Educational** - Maintain instructional value
3. **Clear** - Use accessible language
4. **Example-supported** - Include relevant examples when they aid understanding
Provide only the direct answer to what was asked.
"""
    
    def __init__(self, api_key: str, model: str):
        self.client = anthropic.Anthropic(api_key=api_key)
        self.model = model
        
        # Pre-build base API parameters
        self.base_params = {
            "model": self.model,
            "temperature": 0,
            "max_tokens": 800
        }
    
    def generate_response(self, query: str,
                         conversation_history: Optional[str] = None,
                         tools: Optional[List] = None,
                         tool_manager=None) -> str:
        """
        Generate AI response with optional tool usage and conversation context.
        
        Args:
            query: The user's question or request
            conversation_history: Previous messages for context
            tools: Available tools the AI can use
            tool_manager: Manager to execute tools
            
        Returns:
            Generated response as string
        """
        
        # Build system content efficiently - avoid string ops when possible
        system_content = (
            f"{self.SYSTEM_PROMPT}\n\nPrevious conversation:\n{conversation_history}"
            if conversation_history 
            else self.SYSTEM_PROMPT
        )
        
        # Prepare API call parameters efficiently
        api_params = {
            **self.base_params,
            "messages": [{"role": "user", "content": query}],
            "system": system_content
        }
        
        # Add tools if available
        if tools:
            api_params["tools"] = tools
            api_params["tool_choice"] = {"type": "auto"}
        
        # Get response from Claude
        response = self.client.messages.create(**api_params)
        
        # Handle tool execution if needed
        if response.stop_reason == "tool_use" and tool_manager:
            return self._run_tool_loop(response, api_params, tool_manager)
        
        # Return direct response
        return response.content[0].text
    
    def _run_tool_loop(self, initial_response, api_params: Dict[str, Any], tool_manager, max_rounds=2) -> str:
        messages = api_params["messages"].copy()
        current_response = initial_response
        rounds_done = 0

        while rounds_done < max_rounds and current_response.stop_reason == "tool_use":
            # Append assistant's tool_use content to history
            messages.append({"role": "assistant", "content": current_response.content})

            # Execute all tool calls in this round
            tool_results = []
            error_occurred = False
            for block in current_response.content:
                if block.type == "tool_use":
                    try:
                        result = tool_manager.execute_tool(block.name, **block.input)
                    except Exception as e:
                        result = f"Tool execution failed: {e}"
                        error_occurred = True
                    tool_results.append({
                        "type": "tool_result",
                        "tool_use_id": block.id,
                        "content": result,
                    })

            messages.append({"role": "user", "content": tool_results})
            rounds_done += 1

            if error_occurred:
                break  # Exit loop; fall through to final no-tools call below

            # If budget remains: make intermediate call WITH tools so Claude may search again
            if rounds_done < max_rounds:
                intermediate_params = {
                    **self.base_params,
                    "messages": messages,
                    "system": api_params["system"],
                    "tools": api_params["tools"],
                    "tool_choice": {"type": "auto"},
                }
                current_response = self.client.messages.create(**intermediate_params)
                # If Claude answered directly (end_turn), return immediately — no extra final call
                if current_response.stop_reason != "tool_use":
                    return current_response.content[0].text

        # Rounds exhausted (or tool error): make final call WITHOUT tools to force text answer
        final_params = {
            **self.base_params,
            "messages": messages,
            "system": api_params["system"],
        }
        final_response = self.client.messages.create(**final_params)
        return final_response.content[0].text