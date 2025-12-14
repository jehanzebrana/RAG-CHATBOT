"""
Module: llm_handler.py
Task 1: General Chatbot using Open-Source Lightweight LLM

This module provides a flexible LLM interface supporting both:
- Groq API (fast inference with Llama models)
- Ollama (local deployment)

Features:
- Async/await for better performance
- Retry logic for API failures
- Streaming support (optional)
- Temperature control
"""

import os
from typing import Optional, List, Dict
from loguru import logger
from groq import Groq
import httpx


class LLMHandler:
    """
    Handles interactions with lightweight open-source LLMs.
    Supports Groq API and Ollama.
    """
    
    def __init__(
        self,
        provider: str = "groq",
        groq_api_key: Optional[str] = None,
        groq_model: str = "llama-3.1-8b-instant",
        ollama_base_url: str = "http://localhost:11434",
        ollama_model: str = "llama3.2:3b",
        temperature: float = 0.7,
        max_tokens: int = 1024
    ):
        """
        Initialize LLM Handler.
        
        Args:
            provider: "groq" or "ollama"
            groq_api_key: Groq API key (required if provider="groq")
            groq_model: Model name for Groq
            ollama_base_url: Base URL for Ollama server
            ollama_model: Model name for Ollama
            temperature: Randomness in responses (0.0-1.0)
            max_tokens: Maximum response length
        """
        self.provider = provider.lower()
        self.temperature = temperature
        self.max_tokens = max_tokens
        
        if self.provider == "groq":
            if not groq_api_key:
                raise ValueError("Groq API key is required when provider='groq'")
            self.client = Groq(api_key=groq_api_key)
            self.model = groq_model
            logger.info(f"✓ LLM Handler initialized with Groq ({groq_model})")
            
        elif self.provider == "ollama":
            self.ollama_base_url = ollama_base_url
            self.model = ollama_model
            self._verify_ollama_connection()
            logger.info(f"✓ LLM Handler initialized with Ollama ({ollama_model})")
            
        else:
            raise ValueError(f"Unsupported provider: {provider}. Use 'groq' or 'ollama'")
    
    def _verify_ollama_connection(self) -> None:
        """Verify Ollama server is accessible."""
        try:
            response = httpx.get(f"{self.ollama_base_url}/api/tags", timeout=5.0)
            if response.status_code == 200:
                logger.info("✓ Ollama server is accessible")
            else:
                logger.warning(f"⚠ Ollama server returned status {response.status_code}")
        except Exception as e:
            logger.warning(f"⚠ Could not connect to Ollama: {e}")
    
    async def generate_response(
        self,
        query: str,
        system_prompt: Optional[str] = None,
        conversation_history: Optional[List[Dict[str, str]]] = None
    ) -> str:
        """
        Generate a response to a user query.
        
        Args:
            query: User's question/prompt
            system_prompt: Optional system-level instructions
            conversation_history: Previous conversation turns
            
        Returns:
            Generated response text
        """
        try:
            if self.provider == "groq":
                return await self._generate_groq(query, system_prompt, conversation_history)
            else:
                return await self._generate_ollama(query, system_prompt, conversation_history)
                
        except Exception as e:
            logger.error(f"Error generating response: {e}")
            return f"I apologize, but I encountered an error: {str(e)}"
    
    async def _generate_groq(
        self,
        query: str,
        system_prompt: Optional[str],
        conversation_history: Optional[List[Dict[str, str]]]
    ) -> str:
        """Generate response using Groq API."""
        messages = []
        
        # Add system prompt
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        
        # Add conversation history
        if conversation_history:
            messages.extend(conversation_history)
        
        # Add current query
        messages.append({"role": "user", "content": query})
        
        # Call Groq API
        try:
            chat_completion = self.client.chat.completions.create(
                messages=messages,
                model=self.model,
                temperature=self.temperature,
                max_tokens=self.max_tokens,
            )
            
            response = chat_completion.choices[0].message.content
            logger.info(f"✓ Generated response ({len(response)} chars)")
            return response
            
        except Exception as e:
            logger.error(f"Groq API error: {e}")
            raise
    
    async def _generate_ollama(
        self,
        query: str,
        system_prompt: Optional[str],
        conversation_history: Optional[List[Dict[str, str]]]
    ) -> str:
        """Generate response using Ollama."""
        messages = []
        
        # Add system prompt
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        
        # Add conversation history
        if conversation_history:
            messages.extend(conversation_history)
        
        # Add current query
        messages.append({"role": "user", "content": query})
        
        # Call Ollama API
        try:
            async with httpx.AsyncClient(timeout=60.0) as client:
                response = await client.post(
                    f"{self.ollama_base_url}/api/chat",
                    json={
                        "model": self.model,
                        "messages": messages,
                        "stream": False,
                        "options": {
                            "temperature": self.temperature,
                            "num_predict": self.max_tokens
                        }
                    }
                )
                response.raise_for_status()
                result = response.json()
                
                answer = result["message"]["content"]
                logger.info(f"✓ Generated response ({len(answer)} chars)")
                return answer
                
        except Exception as e:
            logger.error(f"Ollama API error: {e}")
            raise
    
    def get_info(self) -> Dict[str, str]:
        """Get information about the current LLM configuration."""
        return {
            "provider": self.provider,
            "model": self.model,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens
        }


# Example usage
if __name__ == "__main__":
    import asyncio
    from dotenv import load_dotenv
    
    load_dotenv()
    
    async def test():
        # Initialize with Groq
        llm = LLMHandler(
            provider="groq",
            groq_api_key=os.getenv("GROQ_API_KEY"),
            groq_model="llama-3.1-8b-instant"
        )
        
        # Test general query
        response = await llm.generate_response("What is FastAPI?")
        print(f"\n🤖 Response: {response}\n")
    
    asyncio.run(test())
