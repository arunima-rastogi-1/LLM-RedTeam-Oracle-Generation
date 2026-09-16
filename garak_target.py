# garak_target.py
#
# Wraps the actual deployed chatbot pipeline (ChatbotManager: retrieval +
# persona + generation) as a Garak function-generator target, same
# principle as the Inspect solver: Garak probes the real app, not a bare
# Ollama model.
#
# Used via: garak --target_type function --target_name garak_target#nimbus_chat --probes <probes>

from typing import List, Union

from chatbot import ChatbotManager

_manager = None


def nimbus_chat(prompt: str, **kwargs) -> List[Union[str, None]]:
    global _manager
    if _manager is None:
        _manager = ChatbotManager()
    response = _manager.get_response(prompt)
    return [response]
