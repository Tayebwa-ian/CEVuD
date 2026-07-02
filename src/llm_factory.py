"""Module to handle LLM-agnostic chat model instantiation using dynamic configuration."""

import os
from typing import Any
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic
from langchain_google_genai import ChatGoogleGenerativeAI

class LLMFactory:
    """
    Factory pattern implementation for initializing LangChain Chat Models.
    Allows the pipeline to switch between providers (OpenAI, Anthropic, Gemini, Uni Passau Qwen)
    via configuration without changing the core agent logic.
    """

    @staticmethod
    def get_model(provider: str, model_name: str, temperature: float) -> BaseChatModel:
        """Initializes a specified LangChain chat model wrapper based on provider type.

        Args:
            provider (str): Name of provider ('openai', 'anthropic', 'gemini', 'unipassau').
            model_name (str): Specific target model string identifier.
            temperature (float): Sampling randomness parameter.

        Returns:
            BaseChatModel: An executable LangChain model instance.
        """
        prov = provider.lower()
        if prov == "openai":
            return ChatOpenAI(model=model_name, temperature=temperature)
            
        elif prov == "anthropic":
            return ChatAnthropic(model=model_name, temperature=temperature)
            
        elif prov == "gemini":
            return ChatGoogleGenerativeAI(model=model_name, temperature=temperature)
            
        elif prov == "unipassau":
            return ChatOpenAI(
                model="qwen3-next-80b-a3b-instruct",
                temperature=temperature,
                base_url="https://llms.innkube.fim.uni-passau.de",
                api_key=os.getenv("UNIPASSAU_API_KEY") or os.getenv("OPENAI_API_KEY")
            )
            
        else:
            raise ValueError(f"Unsupported LLM provider: '{provider}' requested.")
