from agentverse.registry import Registry

llm_registry = Registry(name="LLMRegistry")
LOCAL_LLMS = [
    "llama-2-7b-chat-hf",
    "llama-2-13b-chat-hf",
    "llama-2-70b-chat-hf",
    "vicuna-7b-v1.5",
    "vicuna-13b-v1.5",
    "DeepSeek-V3",
    "DeepSeek-R1",
    'hunyuan-turbos-latest',
    # 'DeepSeek-V3.1'
]
LOCAL_LLMS_MAPPING = {
    "llama-2-7b-chat-hf": {
        "hf_model_name": "meta-llama/Llama-2-7b-chat-hf",
        "base_url": "http://localhost:5000/v1",
        "api_key": "EMPTY",
    },
    "llama-2-13b-chat-hf": {
        "hf_model_name": "meta-llama/Llama-2-13b-chat-hf",
        "base_url": "http://localhost:5000/v1",
        "api_key": "EMPTY",
    },
    "llama-2-70b-chat-hf": {
        "hf_model_name": "meta-llama/Llama-2-70b-chat-hf",
        "base_url": "http://localhost:5000/v1",
        "api_key": "EMPTY",
    },
    "vicuna-7b-v1.5": {
        "hf_model_name": "lmsys/vicuna-7b-v1.5",
        "base_url": "http://localhost:5000/v1",
        "api_key": "EMPTY",
    },
    "vicuna-13b-v1.5": {
        "hf_model_name": "lmsys/vicuna-13b-v1.5",
        "base_url": "http://localhost:5000/v1",
        "api_key": "EMPTY",
    },
    "DeepSeek-V3": {
        "hf_model_name": "deepseek-ai/DeepSeek-V3",
        "base_url": "http://28.12.131.215:8081/v1",
        "api_key": "EMPTY",
    },
    # "DeepSeek-V3.1": {
    #     "hf_model_name": "deepseek-ai/DeepSeek-V3",
    #     "base_url": "http://29.127.51.15:8081/v1",
    #     "api_key": "EMPTY",
    # },
    
}


from .base import BaseLLM, BaseChatModel, BaseCompletionModel, LLMResult
from .openai import OpenAIChat
