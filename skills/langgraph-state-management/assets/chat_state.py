"""
Chat application state template.

Use this template for chat applications with message history.
"""

from langgraph.graph import MessagesState


class ChatState(MessagesState):
    """State for basic chat application."""
