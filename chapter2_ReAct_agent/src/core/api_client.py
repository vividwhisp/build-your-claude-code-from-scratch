import os
from openai import OpenAI
from typing import Dict, Any, Optional
from dotenv import load_dotenv


class APIClient:
    _instance = None
    _initialized = False
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    def __init__(self):
        """
        Initialize the API client as a singleton using environment variable configuration.
        """
        if not self._initialized:
            # Load the .env file
            load_dotenv()
            
            # Get configuration from environment variables and raise an error if missing
            self.api_key = self._get_required_env_var("OPENAI_API_KEY")
            self.base_url = self._get_required_env_var("OPENAI_BASE_URL") 
            self.model = self._get_required_env_var("OPENAI_MODEL")
            
            self.client = OpenAI(
                api_key=self.api_key,
                base_url=self.base_url
            )
            self._initialized = True
    
    def _get_required_env_var(self, var_name: str) -> str:
        """
        Get a required environment variable and raise an error if it is missing.
        
        Args:
            var_name: The name of the environment variable
            
        Returns:
            The value of the environment variable
            
        Raises:
            ValueError: If the environment variable is missing or empty
        """
        value = os.getenv(var_name)
        if not value:
            raise ValueError(f"环境变量 {var_name} 未设置或为空。请检查 .env 文件中的配置。")
        return value
    
    def get_completion(self, request_params: Dict[str, Any]):
        """
        Send a chat completion request and return the message.
        
        Args:
            request_params: A dictionary of request parameters, including model, messages, etc.
            
        Returns:
            The AI assistant's response message object.
        """
        request_params["model"] = self.model
        try:
            response = self.client.chat.completions.create(**request_params)
            return response.choices[0].message
        except Exception as e:
            raise Exception(f"API请求失败: {str(e)}")
