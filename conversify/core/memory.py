import logging
import os
from typing import List, Any, Dict, Optional, Callable

import numpy as np
from pydantic import BaseModel, Field

from memoripy import MemoryManager, JSONStorage, ChatModel, EmbeddingModel

from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from livekit.agents import ChatMessage, ChatContext

logger = logging.getLogger(__name__)


class ConceptExtractionResponse(BaseModel):
    """Model for structured response from concept extraction."""
    concepts: List[str] = Field(description="List of key concepts extracted from the text.")


class ChatCompletionsModel(ChatModel):
    """Implementation of ChatModel for concept extraction using LLM."""
    
    def __init__(self, llm_config: Dict[str, Any]):
        """Initialize the ChatCompletionsModel with configuration."""
        api_endpoint = llm_config['base_url']
        api_key = llm_config['api_key']
        model_name = llm_config['model']

        logger.info(f"Initializing ChatCompletionsModel with endpoint: {api_endpoint}, model: {model_name}")
        try:
            self.llm = ChatOpenAI(
                openai_api_base=api_endpoint, 
                openai_api_key=api_key, 
                model_name=model_name,
                request_timeout=30.0,  
                max_retries=2         
            )
            self.parser = JsonOutputParser(pydantic_object=ConceptExtractionResponse)
            self.prompt_template = PromptTemplate(
                template=(
                    "Extract key concepts from the following text in a concise, context-specific manner. "
                    "Include only the most highly relevant and specific core concepts that best capture the text's meaning. "
                    "Return nothing but the JSON string.\n"
                    "{format_instructions}\n{text}"
                ),
                input_variables=["text"],
                partial_variables={"format_instructions": self.parser.get_format_instructions()},
            )
            logger.info("ChatCompletionsModel initialized successfully.")
        except Exception as e:
            logger.error(f"Failed to initialize ChatCompletionsModel components: {e}", exc_info=True)
            raise

    def invoke(self, messages: List[Dict[str, Any]]) -> str:
        """Invoke the LLM with a list of messages."""
        if not messages:
            logger.warning("Empty messages list provided to ChatCompletionsModel.invoke()")
            return ""
        
        try:
            response = self.llm.invoke(messages)
            return str(response.content) if response and hasattr(response, 'content') else ""
        except Exception as e:
            logger.error(f"Error during ChatCompletionsModel invocation: {e}", exc_info=True)
            return "Error processing request."

    def extract_concepts(self, text: str) -> List[str]:
        """Extract key concepts from the input text."""
        if not text or not isinstance(text, str) or not text.strip():
            logger.warning("Empty or whitespace-only text provided to extract_concepts()")
            return []
        
        try:
            chain = self.prompt_template | self.llm | self.parser
            response = chain.invoke({"text": text})
            concepts = response.get("concepts", [])
            
            valid_concepts = []
            for concept in concepts:
                if isinstance(concept, str) and concept.strip():
                    valid_concepts.append(concept.strip())
                    
            logger.debug(f"Concepts extracted: {valid_concepts}")
            return valid_concepts
        except Exception as e:
            logger.error(f"Error during concept extraction: {e}", exc_info=True)
            return []


class DummyEmbeddingModel(EmbeddingModel):
    """Dummy implementation of EmbeddingModel for Mac (no vllm)."""
    
    def __init__(self, embedding_config: Dict[str, Any]):
        """Initialize the DummyEmbeddingModel."""
        self.dimension = 768  # Standard embedding dimension
        logger.warning("Using DummyEmbeddingModel - embeddings will be random. Enable a real embedding service for production use.")

    def initialize_embedding_dimension(self) -> int:
        """Return a standard embedding dimension."""
        return self.dimension

    def get_embedding(self, text: str) -> np.ndarray:
        """Return a dummy embedding (random vector)."""
        # In production, you'd call an embedding API here
        # For now, return a random vector
        return np.random.randn(self.dimension)


class AgentMemoryManager:
    """Manages agent memory - simplified version for Mac."""
    
    def __init__(self, participant_identity: str, config: Dict[str, Any]): 
        """Initialize the AgentMemoryManager."""
        self.participant_identity = participant_identity
        self.config = config
        self.memory_config = config['memory']
        self.memory_manager = None
        logger.warning("Memory feature is disabled on Mac due to vllm dependency")

    async def load_memory(self, update_chat_ctx_func: Callable) -> None:
        """Load conversation history - no-op for Mac."""
        logger.info(f"Memory loading skipped for {self.participant_identity} (not supported on Mac)")
        return

    async def save_memory(self, chat_ctx: ChatContext) -> None:
        """Save conversation history - no-op for Mac."""
        logger.info(f"Memory saving skipped for {self.participant_identity} (not supported on Mac)")
        return
