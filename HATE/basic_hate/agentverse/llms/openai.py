import logging
import json
import ast
import os
import numpy as np
from aiohttp import ClientSession
from typing import Dict, List, Optional, Union
from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception_type,
)
from pydantic import Field

from agentverse.llms.base import LLMResult
from agentverse.logging import logger
from agentverse.message import Message

from . import llm_registry, LOCAL_LLMS, LOCAL_LLMS_MAPPING
from .base import BaseChatModel, BaseModelArgs
from .utils.jsonrepair import JsonRepair
from .utils.llm_server_utils import get_llm_server_modelname
from .utils.xx_llms import xx_llms_name
import base64

import hashlib
import hmac, uuid, requests
from datetime import datetime
from colorama import Fore, Back, Style, init
from typing import List, Dict, Tuple, Any

#     return content, prompt_tokens, completion_tokens, total_tokens
MODELS = {
    "o3_pro": "api_openai_o3-pro-2025-06-10",
    "o3": "api_azure_openai_o3",
    "o4_mini": "api_azure_openai_o4-mini",
    "gpt_4o_latest": "api_openai_chatgpt-4o-latest",
    "gpt_4_1": "api_azure_openai_gpt-4.1",
    "gemini2_5_pro": "api_google_gemini-2.5-pro",
    "claude4_opus": "api_aws_anthropic.claude-opus-4-20250514-v1:0",
    "grok4": "api_xai_grok-4-latest",
    "r1": "api_doubao_DeepSeek-R1-250528",
    "v3": "api_doubao_deepseek-v3-250324",
    # Add defaults for previously undefined
    "gemini2_5_flash": "api_google_gemini-2.5-flash",  # Assuming this exists
    "claude4_sonnet": "api_aws_anthropic.claude-sonnet-4-20250514-v1:0",  # Assuming
    "grok3": "api_xai_grok-3-latest"  # Assuming
}

def get_api_result(query: Any, model: str, prompt: str=None) -> Tuple[str, Dict[str, float]]:
    """Anonymization"""
    pass
    # return content, prompt_tokens, completion_tokens, total_tokens, cost

try:
    from openai import OpenAI, AsyncOpenAI
    from openai import OpenAIError
    from openai import AzureOpenAI, AsyncAzureOpenAI
except ImportError:
    is_openai_available = False
    logger.warn(
        "openai package is not installed. Please install it via `pip install openai`"
    )
else:
    api_key = None
    base_url = None
    # model_name = "DeepSeek-V3"
    OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
    OPENAI_BASE_URL = os.environ.get("OPENAI_BASE_URL")
    # debug
    AZURE_API_KEY = os.environ.get("AZURE_OPENAI_API_KEY")
    AZURE_API_BASE = os.environ.get("AZURE_OPENAI_API_BASE")
    # VLLM_BASE_URL = os.environ.get("VLLM_BASE_URL")
    VLLM_API_KEY = os.environ.get("VLLM_API_KEY", "EMPTY")

    if not OPENAI_API_KEY and not AZURE_API_KEY:
        logger.warn(
            "OpenAI API key is not set. Please set an environment variable OPENAI_API_KEY or "
            "AZURE_OPENAI_API_KEY."
        )
    elif OPENAI_API_KEY:
        DEFAULT_CLIENT = OpenAI(api_key=OPENAI_API_KEY, base_url=OPENAI_BASE_URL)
        DEFAULT_CLIENT_ASYNC = AsyncOpenAI(
            api_key=OPENAI_API_KEY, base_url=OPENAI_BASE_URL
        )
        api_key = OPENAI_API_KEY
        base_url = OPENAI_BASE_URL
    elif AZURE_API_KEY:
        DEFAULT_CLIENT = AzureOpenAI(
            api_key=AZURE_API_KEY,
            azure_endpoint=AZURE_API_BASE,
            api_version="2024-02-15-preview",
        )
        DEFAULT_CLIENT_ASYNC = AsyncAzureOpenAI(
            api_key=AZURE_API_KEY,
            azure_endpoint=AZURE_API_BASE,
        )
        api_key = AZURE_API_KEY
        base_url = AZURE_API_BASE
    if VLLM_BASE_URL:
        if model_name := get_llm_server_modelname(VLLM_BASE_URL, VLLM_API_KEY, logger):
            # model_name = /mnt/llama/hf_models/TheBloke_Llama-2-70B-Chat-GPTQ
            # transform to TheBloke/Llama-2-70B-Chat-GPTQ
            # print('Using vLLM model???:', model_name)
            # try:
            #     hf_model_name = LOCAL_LLMS_MAPPING[model_name]["hf_model_name"]
            # except KeyError:
            # hf_model_name = model_name.split("/")[-1].replace("_", "/")
            hf_model_name = LOCAL_LLMS_MAPPING[model_name]["hf_model_name"]
            LOCAL_LLMS.append(model_name)
            LOCAL_LLMS_MAPPING[model_name] = {
                "hf_model_name": hf_model_name,
                "base_url": VLLM_BASE_URL,
                "api_key": VLLM_API_KEY if VLLM_API_KEY else "EMPTY",
            }
            logger.info(f"Using vLLM model: {hf_model_name}")
            # print(f"Using vLLM model: {hf_model_name}")
    if hf_model_name := get_llm_server_modelname(
        "http:/", logger=logger
    ):
        # meta-llama/Llama-2-7b-chat-hf
        # transform to llama-2-7b-chat-hf
        short_model_name = model_name.split("/")[-1].lower()
        LOCAL_LLMS.append(short_model_name)
        LOCAL_LLMS_MAPPING[short_model_name] = {
            "hf_model_name": hf_model_name,
            "base_url": "http:",
            "api_key": "EMPTY",
        }

        logger.info(f"Using FSChat model: {model_name}")
    if hf_model_name := get_llm_server_modelname(
        "http: ", logger=logger
    ):
        # meta-llama/Llama-2-7b-chat-hf
        # transform to llama-2-7b-chat-hf
        short_model_name = model_name.split("/")[-1].lower()
        LOCAL_LLMS.append(short_model_name)
        LOCAL_LLMS_MAPPING[short_model_name] = {
            "hf_model_name": hf_model_name,
            "base_url": " ",
            "api_key": "EMPTY",
        }

        logger.info(f"Using FSChat model: {model_name}")


class OpenAIChatArgs(BaseModelArgs):
    model: str = Field(default="gpt-3.5-turbo")
    max_tokens: int = Field(default=2048)
    temperature: float = Field(default=1.0)
    top_p: int = Field(default=1)
    n: int = Field(default=1)
    stop: Optional[Union[str, List]] = Field(default=None)
    presence_penalty: int = Field(default=0)
    frequency_penalty: int = Field(default=0)
    xx: str = Field(default="")


@llm_registry.register("gpt-35-turbo")
@llm_registry.register("gpt-3.5-turbo")
@llm_registry.register("gpt-4")
@llm_registry.register("vllm")
@llm_registry.register("local")
@llm_registry.register("DeepSeek-V3")
@llm_registry.register("DeepSeek-R1")
@llm_registry.register("hunyuan-turbos-latest")
# @llm_registry.register("api_azure_openai_o3")
# @llm_registry.register("api_openai_chatgpt-4o-latest")
@llm_registry.register("api_google_gemini-2.5-flash")
# @llm_registry.register("api_google_gemini-2.5-pro")
@llm_registry.register("api_openai_o3-pro-2025-06-10")
@llm_registry.register("api_azure_openai_gpt-4.1")
@llm_registry.register("api_aws_anthropic.claude-sonnet-4-20250514-v1:0")
# @llm_registry.register("api_xai_grok-4-latest")
@llm_registry.register("api_anthropic_claude-opus-4-20250514")
@llm_registry.register("api_aws_anthropic.claude-opus-4-20250514-v1:0")
@llm_registry.register("api_google_gemini-2.5-pro")
@llm_registry.register("api_azure_openai_gpt-5")
@llm_registry.register("api_google_claude-opus-4-1@20250805")
@llm_registry.register("api_azure_openai_o3")
@llm_registry.register("api_openai_chatgpt-4o-latest")
@llm_registry.register("api_azure_openai_gpt-4.5-preview")
# @llm_registry.register("api_google_claude-opus-4-1@20250805")
# @llm_registry.register("api_openai_gpt-5-chat-latest-response-async")
@llm_registry.register("api_xai_grok-4-latest")
@llm_registry.register("api_moonshot_kimi-k2-0711-preview")
@llm_registry.register("api_doubao_DeepSeek-V3.1")
@llm_registry.register("api_ali_qwen3-235b-a22b-instruct-2507")
# @llm_registry.register("api_aws_anthropic_claude-opus-4-20250514-v1:0")
@llm_registry.register("DeepSeek-V3.1")
@llm_registry.register("api_google_claude-opus-4@20250514")
@llm_registry.register("api_doubao_DeepSeek-V3.1-250821")
class OpenAIChat(BaseChatModel):
    args: OpenAIChatArgs = Field(default_factory=OpenAIChatArgs)
    client_args: Optional[Dict] = Field(
        default={"api_key": api_key, "base_url": base_url}
    )
    is_azure: bool = Field(default=False)

    total_prompt_tokens: int = 0
    total_completion_tokens: int = 0
    cost: float = 0.0

    def __init__(self, max_retry: int = 3, **kwargs):
        args = OpenAIChatArgs()
        args = args.dict()
        client_args = {"api_key": api_key, "base_url": base_url}
        # check if api_key is an azure key
        is_azure = False
        if AZURE_API_KEY and not OPENAI_API_KEY:
            is_azure = True
        for k, v in args.items():
            args[k] = kwargs.pop(k, v)
        if len(kwargs) > 0:
            logger.warn(f"Unused arguments: {kwargs}")
        if args["model"] in LOCAL_LLMS:
            if args["model"] in LOCAL_LLMS_MAPPING:
                client_args["api_key"] = LOCAL_LLMS_MAPPING[args["model"]]["api_key"]
                client_args["base_url"] = LOCAL_LLMS_MAPPING[args["model"]]["base_url"]
                is_azure = False
            else:
                raise ValueError(
                    f"Model {args['model']} not found in LOCAL_LLMS_MAPPING"
                )
        if 'hunyuan' in args["model"].lower():
            args['presence_penalty'] = 1.0
        # if 'o3' in args["model"].lower() or '4o' in args["model"].lower() or 'gemini' in args["model"].lower() or 'claude' in args["model"].lower() or 'grok' in args["model"].lower() or 'gpt-4.1' in args['model'].lower():
        if args['model'].lower() in xx_llms_name.values() or args['model'] in xx_llms_name.values():
            args['xx'] = 'xx'
        print(args["model"])
        super().__init__(
            args=args, max_retry=max_retry, client_args=client_args, is_azure=is_azure
        )

    @classmethod
    def send_token_limit(self, model: str) -> int:
        send_token_limit_dict = {
            "gpt-3.5-turbo": 4096,
            "gpt-35-turbo": 4096,
            "gpt-3.5-turbo-16k": 16384,
            "gpt-3.5-turbo-0613": 16384,
            "gpt-3.5-turbo-1106": 16384,
            "gpt-3.5-turbo-0125": 16384,
            "gpt-4": 8192,
            "gpt-4-32k": 32768,
            "gpt-4-0613": 32768,
            "gpt-4-1106-preview": 131072,
            "gpt-4-0125-preview": 131072,
            "llama-2-7b-chat-hf": 131072,
        }
        # Default to 4096 tokens if model is not in the dictionary
        return send_token_limit_dict[model] if model in send_token_limit_dict else 4096

    # @retry(
    #     stop=stop_after_attempt(20),
    #     wait=wait_exponential(multiplier=1, min=4, max=10),
    #     reraise=True,
    #     retry=retry_if_exception_type(
    #         exception_types=(OpenAIError, json.decoder.JSONDecodeError, Exception)
    #     ),
    # )
    def generate_response(
        self,
        prepend_prompt: str = "",
        history: List[dict] = [],
        append_prompt: str = "",
        functions: List[dict] = [],
    ) -> LLMResult:
        messages = self.construct_messages(prepend_prompt, history, append_prompt)
        logger.log_prompt(messages)
        if self.args.xx == 'xx':
            content, prompt_tokens, completion_tokens, total_tokens, cost = get_api_result(messages, self.args.model, None)
            self.collect_metrics(None, cost)
            return LLMResult(
                        content=content,
                        send_tokens=prompt_tokens,
                        recv_tokens=completion_tokens,
                        total_tokens=total_tokens,
                    )
        if self.is_azure == True:
            openai_client = AzureOpenAI(
                api_key=self.client_args["api_key"],
                azure_endpoint=self.client_args["base_url"],
                api_version="2024-02-15-preview",
            )
        elif self.is_azure == False:
            openai_client = OpenAI(
                api_key=self.client_args["api_key"],
                base_url=self.client_args["base_url"],
            )
        # try:
            # Execute function call
        if functions != []:
            # args_dict = self.args.dict().remove('')
            response = openai_client.chat.completions.create(
                messages=messages,
                functions=functions,
                **self.args.dict().pop('xx'),
            )

            logger.log_prompt(
                [
                    {
                        "role": "assistant",
                        "content": response.choices[0].message.content,
                    }
                ]
            )
            if response.choices[0].message.function_call is not None:
                self.collect_metrics(response)

                return LLMResult(
                    content=response.choices[0].message.get("content", ""),
                    function_name=response.choices[0].message.function_call.name,
                    function_arguments=ast.literal_eval(
                        response.choices[0].message.function_call.arguments
                    ),
                    send_tokens=response.usage.prompt_tokens,
                    recv_tokens=response.usage.completion_tokens,
                    total_tokens=response.usage.total_tokens,
                )
            else:
                self.collect_metrics(response)
                logger.log_prompt(
                    {
                        "role": "assistant",
                        "content": response.choices[0].message.content,
                    }
                )
                return LLMResult(
                    content=response.choices[0].message.content,
                    send_tokens=response.usage.prompt_tokens,
                    recv_tokens=response.usage.completion_tokens,
                    total_tokens=response.usage.total_tokens,
                )

        else:
            response = openai_client.chat.completions.create(
                messages=messages,
                **self.args.dict().pop('xx'),
            )
            logger.log_prompt(
                [
                    {
                        "role": "assistant",
                        "content": response.choices[0].message.content,
                    }
                ]
            )
            self.collect_metrics(response)
            return LLMResult(
                content=response.choices[0].message.content,
                send_tokens=response.usage.prompt_tokens,
                recv_tokens=response.usage.completion_tokens,
                total_tokens=response.usage.total_tokens,
            )
        # except (OpenAIError, KeyboardInterrupt, json.decoder.JSONDecodeError) as error:
        #     raise

    # @retry(
    #     stop=stop_after_attempt(20),
    #     wait=wait_exponential(multiplier=1, min=4, max=10),
    #     reraise=True,
    #     retry=retry_if_exception_type(
    #         exception_types=(OpenAIError, json.decoder.JSONDecodeError, Exception)
    #     ),
    # )
    async def agenerate_response(
        self,
        prepend_prompt: str = "",
        history: List[dict] = [],
        append_prompt: str = "",
        functions: List[dict] = [],
    ) -> LLMResult:
        messages = self.construct_messages(prepend_prompt, history, append_prompt)
        logger.log_prompt(messages)
        # os._exit(0)
        print('-------------------\n', messages)
        print('self.args: ', self.args)
        if self.args.xx == 'xx':
            content, prompt_tokens, completion_tokens, total_tokens, cost = get_api_result(messages, self.args.model, None)
            self.collect_metrics(None, cost)
            return LLMResult(
                        content=content,
                        send_tokens=prompt_tokens,
                        recv_tokens=completion_tokens,
                        total_tokens=total_tokens,
                    )
        if self.is_azure:
            async_openai_client = AsyncAzureOpenAI(
                api_key=self.client_args["api_key"],
                azure_endpoint=self.client_args["base_url"],
                api_version="2024-02-15-preview",
            )
        else:
            async_openai_client = AsyncOpenAI(
                api_key=self.client_args["api_key"],
                base_url=self.client_args["base_url"],
            )
        # try:
        if functions != []:
            response = await async_openai_client.chat.completions.create(
                messages=messages,
                functions=functions,
                **self.args.dict().pop('xx'),
            )
            logger.log_prompt(
                [
                    {
                        "role": "assistant",
                        "content": response.choices[0].message.content,
                    }
                ]
            )
            if response.choices[0].message.function_call is not None:
                function_name = response.choices[0].message.function_call.name
                valid_function = False
                if function_name.startswith("function."):
                    function_name = function_name.replace("function.", "")
                elif function_name.startswith("functions."):
                    function_name = function_name.replace("functions.", "")
                for function in functions:
                    if function["name"] == function_name:
                        valid_function = True
                        break
                if not valid_function:
                    logger.warn(
                        f"The returned function name {function_name} is not in the list of valid functions. Retrying..."
                    )
                    raise ValueError(
                        f"The returned function name {function_name} is not in the list of valid functions."
                    )
                try:
                    arguments = ast.literal_eval(
                        response.choices[0].message.function_call.arguments
                    )
                except:
                    try:
                        arguments = ast.literal_eval(
                            JsonRepair(
                                response.choices[0].message.function_call.arguments
                            ).repair()
                        )
                    except:
                        logger.warn(
                            "The returned argument in function call is not valid json. Retrying..."
                        )
                        raise ValueError(
                            "The returned argument in function call is not valid json."
                        )
                self.collect_metrics(response)
                logger.log_prompt(
                    {
                        "role": "assistant",
                        "content": response.choices[0].message.content,
                    }
                )
                return LLMResult(
                    function_name=function_name,
                    function_arguments=arguments,
                    send_tokens=response.usage.prompt_tokens,
                    recv_tokens=response.usage.completion_tokens,
                    total_tokens=response.usage.total_tokens,
                )

            else:
                self.collect_metrics(response)
                logger.log_prompt(
                    {
                        "role": "assistant",
                        "content": response.choices[0].message.content,
                    }
                )
                return LLMResult(
                    content=response.choices[0].message.content,
                    send_tokens=response.usage.prompt_tokens,
                    recv_tokens=response.usage.completion_tokens,
                    total_tokens=response.usage.total_tokens,
                )

        else:
            # print('Logging messages: ', messages)
            print(async_openai_client.base_url)
            paras = self.args.dict()
            paras.pop('xx')
            response = await async_openai_client.chat.completions.create(
                messages=messages,
                # **self.args.dict().pop('xx'),
                **paras
            )
            # print(response)
            self.collect_metrics(response)
            logger.log_prompt(
                [
                    {
                        "role": "assistant",
                        "content": response.choices[0].message.content,
                    }
                ]
            )
            return LLMResult(
                content=response.choices[0].message.content,
                send_tokens=response.usage.prompt_tokens,
                recv_tokens=response.usage.completion_tokens,
                total_tokens=response.usage.total_tokens,
            )
        # except (OpenAIError, KeyboardInterrupt, json.decoder.JSONDecodeError) as error:
        #     raise

    def construct_messages(
        self, prepend_prompt: str, history: List[dict], append_prompt: str
    ):
        messages = []
        # print('-------------------\n', history)
        if prepend_prompt != "":
            messages.append({"role": "system", "content": prepend_prompt})
        if append_prompt != "":
            messages.append({"role": "user", "content": append_prompt})
        if len(history) > 0:
            # messages += history
            history_text = '''\n\nDiscussion history are as follows:\n''' + "\n".join(
                [f"{hi['content']}" for hi in history]
            )  + '\n\n'
            messages.insert(1, {"role": "user", "content": history_text})
        elif not append_prompt.startswith('Your task is to evaluate '):
            history_text = '''\n\nDiscussion history are as follows:\n[](Temporarily Empty)\n\n'''
            messages.insert(1, {"role": "user", "content": history_text})
            # history_text = '''\n\nDiscussion history are as follows:\n''' + history_text + '\n\n'
            # # for hi in history:
            # messages[1]['content'] = history_text + messages[1]['content']
        # print('-------------------\n', messages)
        # if len(history) > 2:
        #     os._exit(0)
        return messages

    def collect_metrics(self, response, cost=None):
        if not response:
            self.total_prompt_tokens = -1
            self.total_completion_tokens = -1
            self.cost = cost
        else:
            self.total_prompt_tokens += response.usage.prompt_tokens
            self.total_completion_tokens += response.usage.completion_tokens
            self.cost = -1

    def get_spend(self) -> int:
        input_cost_map = {
            "gpt-3.5-turbo": 0.0015,
            "gpt-3.5-turbo-16k": 0.003,
            "gpt-3.5-turbo-0613": 0.0015,
            "gpt-3.5-turbo-16k-0613": 0.003,
            "gpt-3.5-turbo-1106": 0.0005,
            "gpt-3.5-turbo-0125": 0.0005,
            "gpt-4": 0.03,
            "gpt-4-0613": 0.03,
            "gpt-4-32k": 0.06,
            "gpt-4-1106-preview": 0.01,
            "gpt-4-0125-preview": 0.01,
            "llama-2-7b-chat-hf": 0.0,
            "DeepSeek-V3": 0.0,  # Assuming DeepSeek-V3 has no cost
            'DeepSeek-R1': 0.0,  # Assuming DeepSeek-R1 has no cost
            'hunyuan-turbos-latest': 0.0,  # Assuming
            "DeepSeek-V3.1": 0.0, 
        }

        output_cost_map = {
            "gpt-3.5-turbo": 0.002,
            "gpt-3.5-turbo-16k": 0.004,
            "gpt-3.5-turbo-0613": 0.002,
            "gpt-3.5-turbo-16k-0613": 0.004,
            "gpt-3.5-turbo-1106": 0.0015,
            "gpt-3.5-turbo-0125": 0.0015,
            "gpt-4": 0.06,
            "gpt-4-0613": 0.06,
            "gpt-4-32k": 0.12,
            "gpt-4-1106-preview": 0.03,
            "gpt-4-0125-preview": 0.03,
            "llama-2-7b-chat-hf": 0.0,
            "DeepSeek-V3": 0.0,  # Assuming DeepSeek-V3 has no cost
            'DeepSeek-R1': 0.0,  # Assuming DeepSeek-R1 has no cost
            'hunyuan-turbos-latest': 0.0,  # Assuming
            "DeepSeek-V3.1": 0.0,  
        }

        model = self.args.model
        if model not in input_cost_map or model not in output_cost_map:
            if self.cost == -1:
                raise ValueError(f"Model type {model} not supported")
            return self.cost
        return (
            self.total_prompt_tokens * input_cost_map[model] / 1000.0
            + self.total_completion_tokens * output_cost_map[model] / 1000.0
        )


@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=4, max=10),
    reraise=True,
)
def get_embedding(text: str, attempts=3) -> np.array:
    if AZURE_API_KEY and AZURE_API_BASE:
        client = AzureOpenAI(
            api_key=AZURE_API_KEY,
            azure_endpoint=AZURE_API_BASE,
            api_version="2024-02-15-preview",
        )
    elif OPENAI_API_KEY:
        client = OpenAI(api_key=OPENAI_API_KEY, base_url=OPENAI_BASE_URL)
    try:
        text = text.replace("\n", " ")
        embedding = client.embeddings.create(
            input=text, model="text-embedding-ada-002"
        ).model_dump_json(indent=2)
        return tuple(embedding)
    except Exception as e:
        attempt += 1
        logger.error(f"Error {e} when requesting openai models. Retrying")
        raise
