"""Module to handle LLM-agnostic chat model instantiation using dynamic configuration."""

import os
from typing import Any
from langchain_core.language_models.chat_models import BaseChatModel

class LLMFactory:
    """Dynamic provider mapping factory for abstracting frontier LLM backends."""

    @staticmethod
    def get_model(provider: str, model_name: str, temperature: float) -> BaseChatModel:
        """Initializes a specified LangChain chat model wrapper based on provider type.

        Args:
            provider (str): Name of provider ('openai', 'anthropic', 'gemini').
            model_name (str): Specific target model string identifier.
            temperature (float): Sampling randomness parameter.

        Returns:
            BaseChatModel: An executable LangChain model instance.
        """
        prov = provider.lower()
        if prov == "openai":
            from langchain_openai import ChatOpenAI
            return ChatOpenAI(model=model_name, temperature=temperature)
            
        elif prov == "anthropic":
            from langchain_anthropic import ChatAnthropic
            return ChatAnthropic(model=model_name, temperature=temperature)
            
        elif prov == "gemini":
            from langchain_google_genai import ChatGoogleGenerativeAI
            return ChatGoogleGenerativeAI(model=model_name, temperature=temperature)
            
        else:
            raise ValueError(f"Unsupported LLM provider: '{provider}' requested.")
