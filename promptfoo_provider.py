# promptfoo_provider.py
#
# Promptfoo custom Python provider: wraps the actual deployed chatbot
# pipeline (ChatbotManager: retrieval + persona + generation), same
# principle as the Inspect solver and the Garak function-generator,
# Promptfoo probes the real app, not a bare Ollama model.

from chatbot import ChatbotManager

_manager = None


def call_api(prompt, options, context):
    global _manager
    if _manager is None:
        _manager = ChatbotManager()
    response = _manager.get_response(prompt)
    return {"output": response}
