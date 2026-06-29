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
    with patch("langchain_openai.ChatOpenAI") as mock_openai:
        mock_instance = MagicMock()
        mock_openai.return_value = mock_instance
        
        model = LLMFactory.get_model(provider="openai", model_name="gpt-4o", temperature=0.2)
        
        mock_openai.assert_called_once_with(model="gpt-4o", temperature=0.2)
        assert model == mock_instance

def test_llm_factory_initializes_anthropic_with_mock():
    """Verifies that the factory returns a correct instance for the Anthropic provider string."""
    with patch("langchain_anthropic.ChatAnthropic") as mock_anthropic:
        mock_instance = MagicMock()
        mock_anthropic.return_value = mock_instance
        
        model = LLMFactory.get_model(provider="anthropic", model_name="claude-3", temperature=0.0)
        
        mock_anthropic.assert_called_once_with(model="claude-3", temperature=0.0)
        assert model == mock_instance

def test_llm_factory_raises_value_error_on_invalid_provider():
    """Ensures an explicit error is thrown when requesting an unknown provider."""
    with pytest.raises(ValueError, match="Unsupported LLM provider: 'invalid-ai'"):
        LLMFactory.get_model(provider="invalid-ai", model_name="brain-v1", temperature=0.7)