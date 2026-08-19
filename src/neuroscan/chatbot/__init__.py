"""Bilingual (English / Nepali) conversational interface.

Requirement: a bilingual chatbot for conversational purposes in English and
Nepali, able to give context-based answers to the queries of patients as well
as medical personnel.
"""

from neuroscan.chatbot.engine import ChatbotEngine, ChatResponse, ChatTurn
from neuroscan.chatbot.language import detect_language, get_ui_strings

__all__ = ["ChatResponse", "ChatTurn", "ChatbotEngine", "detect_language", "get_ui_strings"]
