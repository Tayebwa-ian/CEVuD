"""
Unit Tests: LLMFactory
======================
Validates that LangChain provider initializations map correctly or error out gracefully.
"""

import pytest
from unittest.mock import patch, MagicMock
import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

from llm_factory import LLMFactory

def test_llm_factory_initializes_openai_with_mock():
    """Verifies that the factory returns a correct instance for the OpenAI provider string."""
    with patch("llm_factory.ChatOpenAI") as mock_openai:
        mock_instance = MagicMock()
        mock_openai.return_value = mock_instance
        
        model = LLMFactory.get_model(provider="openai", model_name="gpt-4o", temperature=0.2)
        
        mock_openai.assert_called_once_with(model="gpt-4o", temperature=0.2)
        assert model == mock_instance

def test_llm_factory_initializes_anthropic_with_mock():
    """Verifies that the factory returns a correct instance for the Anthropic provider string."""
    with patch("llm_factory.ChatAnthropic") as mock_anthropic:
        mock_instance = MagicMock()
        mock_anthropic.return_value = mock_instance
        
        model = LLMFactory.get_model(provider="anthropic", model_name="claude-3", temperature=0.0)
        
        mock_anthropic.assert_called_once_with(model="claude-3", temperature=0.0)
        assert model == mock_instance

def test_llm_factory_initializes_gemini_with_mock():
    """Verifies that the factory returns a correct instance for the Gemini provider string."""
    with patch("llm_factory.ChatGoogleGenerativeAI") as mock_gemini:
        mock_instance = MagicMock()
        mock_gemini.return_value = mock_instance
        
        model = LLMFactory.get_model(provider="gemini", model_name="gemini-pro", temperature=0.5)
        
        mock_gemini.assert_called_once_with(model="gemini-pro", temperature=0.5)
        assert model == mock_instance

def test_llm_factory_initializes_unipassau_with_mock():
    """Verifies that the factory returns a correct instance for the Uni Passau Qwen provider string."""
    with patch("llm_factory.ChatOpenAI") as mock_unipassau:
        mock_instance = MagicMock()
        mock_unipassau.return_value = mock_instance
        
        # Test with UNIPASSAU_API_KEY set
        os.environ["UNIPASSAU_API_KEY"] = "test-unipassau-key"
        model = LLMFactory.get_model(provider="unipassau", model_name="qwen-7b", temperature=0.3)
        
        mock_unipassau.assert_called_once_with(
            model="qwen3-next-80b-a3b-instruct",
            temperature=0.3,
            base_url="https://llms.innkube.fim.uni-passau.de",
            api_key="test-unipassau-key"
        )
        assert model == mock_instance
        
        # Clean up
        del os.environ["UNIPASSAU_API_KEY"]

def test_llm_factory_unipassau_falls_back_to_openai_key():
    """Verifies that Uni Passau provider falls back to OPENAI_API_KEY when UNIPASSAU_API_KEY is not set."""
    with patch("llm_factory.ChatOpenAI") as mock_unipassau:
        mock_instance = MagicMock()
        mock_unipassau.return_value = mock_instance
        
        # Set OPENAI_API_KEY but not UNIPASSAU_API_KEY
        os.environ["OPENAI_API_KEY"] = "test-openai-key"
        model = LLMFactory.get_model(provider="unipassau", model_name="qwen-7b", temperature=0.3)
        
        mock_unipassau.assert_called_once_with(
            model="qwen3-next-80b-a3b-instruct",
            temperature=0.3,
            base_url="https://llms.innkube.fim.uni-passau.de",
            api_key="test-openai-key"
        )
        assert model == mock_instance
        
        # Clean up
        del os.environ["OPENAI_API_KEY"]

def test_llm_factory_raises_value_error_on_invalid_provider():
    """Ensures an explicit error is thrown when requesting an unknown provider."""
    with pytest.raises(ValueError, match="Unsupported LLM provider: 'invalid-ai'"):
        LLMFactory.get_model(provider="invalid-ai", model_name="brain-v1", temperature=0.7)