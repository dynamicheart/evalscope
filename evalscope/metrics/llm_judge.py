import os
import re
import requests
import time
import random
import json
from typing import Any, Dict, List, Optional

from evalscope.api.messages import ChatMessage, ChatMessageSystem, ChatMessageUser
from evalscope.constants import JudgeScoreType
from evalscope.utils.logger import get_logger
import threading

logger = get_logger()

DEFAULT_PROMPT_TEMPLATE = """Your job is to look at a question, a gold target, and a predicted answer, and return a letter "A" or "B" to indicate whether the predicted answer is correct or incorrect.

[Question]
{question}

[Reference Answer]
{gold}

[Predicted Answer]
{pred}

Evaluate the model's answer based on correctness compared to the reference answer.
Grade the predicted answer of this new question as one of:
A: CORRECT
B: INCORRECT

Just return the letters "A" or "B", with no text around it.
"""  # noqa: E501


DEFAULT_NUMERIC_SCORE_TEMPLATE = """Please act as an impartial judge and evaluate the quality of the response provided by an AI assistant to the user question displayed below. Your evaluation should consider factors such as the helpfulness, relevance, accuracy, depth, creativity, and level of detail of the response.
Begin your evaluation by providing a short explanation. Be as objective as possible.
After providing your explanation, you must rate the response on a scale of 0 (worst) to 1 (best) by strictly following this format: \"[[rating]]\", for example: \"Rating: [[0.5]]\"

[Question]
{question}

[Response]
{pred}
"""  # noqa: E501

DEFAULT_JUDGE_MODEL = 'Qwen/Qwen3-235B-A22B'
DEFAULT_API_URL = 'https://api-inference.modelscope.cn/v1/'


class RateLimiter:
    """Global rate limiter for multi-thread environment."""
    def __init__(self, rate_per_sec):
        """
        rate_per_sec: allowed requests per second (global)
        """
        self.rate = rate_per_sec
        self.allowance = rate_per_sec  # 当前可用额度
        self.last_check = time.time()
        self.lock = threading.Lock()

    def acquire(self):
        """Block until allowed to send a request"""
        while True:
            with self.lock:
                current = time.time()
                elapsed = current - self.last_check
                self.last_check = current
                self.allowance += elapsed * self.rate
                if self.allowance > self.rate:
                    self.allowance = self.rate

                if self.allowance >= 1.0:
                    self.allowance -= 1.0
                    return  # allowed
            time.sleep(0.01)  # sleep 10ms and retry

class LLMJudge:
    """
    A metric that uses LLM to judge the quality of model predictions by comparing them with reference answers.
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        api_url: Optional[str] = None,
        model_id: Optional[str] = None,
        system_prompt: Optional[str] = None,
        prompt_template: Optional[str] = None,
        generation_config: Optional[Dict[str, Any]] = None,
        score_pattern: Optional[str] = None,
        score_mapping: Optional[Dict[str, float]] = None,
        score_type: str = JudgeScoreType.PATTERN,  # 'pattern', 'numeric'
        **kwargs
    ):
        """
        Initialize LLMJudge metric.

        Args:
            api_key (str, optional): API key for OpenAI or compatible service
            api_base (str, optional): API base URL
            model_id (str, optional): Model ID for LLM
            system_prompt (str, optional): System prompt for the judge
            prompt_template (str, optional): Prompt template for the judge
            generation_config (dict, optional): Generation configuration for the judge
            score_pattern (str, optional): Regex pattern to extract score from LLM response
            score_mapping (dict, optional): Mapping from extracted score to float value
            score_type (str, optional): Type of score extraction strategy ('pattern', 'numeric') defaults to 'pattern'.
                - 'pattern': Use score_pattern and score_mapping to extract categorical scores
                - 'numeric': Treat the extracted value as a direct numerical score
        """
        self.api_key = api_key or os.environ.get('MODELSCOPE_SDK_TOKEN', 'EMPTY')
        self.api_key_list = self.api_key.split(',')
        self.api_key = self.api_key_list[0]
        self.rate_limiter = RateLimiter(rate_per_sec=5)
        self.key_lock = threading.Lock()
        self.key_cooldown = {key: 0 for key in self.api_key_list}  # 冷却时间戳

        self.api_url = api_url or os.environ.get('MODELSCOPE_API_BASE', DEFAULT_API_URL)
        self.model_id = model_id or os.environ.get('MODELSCOPE_JUDGE_LLM', DEFAULT_JUDGE_MODEL)
        self.system_prompt = system_prompt or os.environ.get('JUDGE_SYSTEM_PROMPT', None)
        self.generation_config = generation_config or {'temperature': 0.001, 'max_tokens': 1024}

        # Default score mapping for A/B pattern
        self.score_type = score_type
        if self.score_type == JudgeScoreType.NUMERIC:
            self.score_pattern = score_pattern or r'\[\[(\d+(?:\.\d+)?)\]\]'
            self.prompt_template = prompt_template or os.environ.get(
                'JUDGE_PROMPT_TEMPLATE', DEFAULT_NUMERIC_SCORE_TEMPLATE
            )
        elif self.score_type == JudgeScoreType.PATTERN:
            self.score_pattern = score_pattern or r'(A|B)'
            self.prompt_template = prompt_template or os.environ.get('JUDGE_PROMPT_TEMPLATE', DEFAULT_PROMPT_TEMPLATE)
        else:
            raise ValueError(f"Invalid score_type: {self.score_type}. Must be 'pattern' or 'numeric'.")
        self.score_mapping = score_mapping or {'A': 1.0, 'B': 0.0}

        self._init_server_adapter()

    def _init_server_adapter(self):
        from evalscope.api.model import GenerateConfig, get_model

        self.model = get_model(
            model=self.model_id,
            eval_type='openai_api',
            base_url=self.api_url,
            api_key=self.api_key,
            config=GenerateConfig(**self.generation_config),
        )

    def _get_available_key(self):
        """Select an available API key (not in cooldown)."""
        while True:
            with self.key_lock:
                now = time.time()
                valid_keys = [k for k in self.api_key_list if now >= self.key_cooldown[k]]
                if valid_keys:
                    return random.choice(valid_keys)
            sleep_time = random.uniform(5, 10)
            logger.warning(f"All API keys cooling down, wait {sleep_time:.1f}s...")
            time.sleep(sleep_time)

    def ernie_judge(
        self,
        prompt: str = '',
        system_prompt: Optional[str] = None,
        messages: Optional[List[ChatMessage]] = None
    ) -> str:
        """
        Generate a response from the LLM based on the provided prompt and context.
        If messages is provided, it will be used as the input context.

        Args:
            prompt (str): The prompt to evaluate
            system_prompt (str, optional): The system prompt to use for the evaluation
            messages (List[ChatMessage], optional): A list of chat messages to include in the evaluation
        Returns:
            str: The response from the LLM
        """
        # 构造 messages
        if messages is not None:
            input_messages = messages
            assert False, "ChatMessage not implemented yet for ernie judge"
        else:
            system_content = system_prompt or self.system_prompt
            input_messages = [{"role": "user", "content": prompt}]
            if system_content:
                input_messages.insert(0, {"role": "system", "content": system_content})

        # 请求体
        payload = {
            "model": self.model_id,
            "temperature": self.generation_config.get("temperature", 0.001),
            "max_tokens": self.generation_config.get("max_tokens", 1024),
            "stream": False,
            "safety_level": "none",
            "messages": input_messages
        }

        # 随机选择一个api_key
        max_retries = 5
        for attempt in range(max_retries):
            self.rate_limiter.acquire()  # global rate limit
            api_key = self._get_available_key()
            try:
                headers = {"Content-Type": "application/json"}
                url = f"{self.api_url}?access_token={api_key}"
                resp = requests.post(url, headers=headers, data=json.dumps(payload), timeout=600)

                if resp.status_code != 200:
                    logger.error(f"[{attempt+1}/{max_retries}] HTTP {resp.status_code}: {resp.text}")
                    continue

                result = resp.json()

                # Check if rate limit reached
                if result.get("error_code") == 336502:
                    logger.warning(f"API key {api_key[:6]}... hit rate limit, cooling down for 30 seconds.")
                    self.key_cooldown[api_key] = time.time() + random.uniform(30, 40)
                    time.sleep(2 + attempt)  # exponential backoff
                    continue

                llm_response = result.get("result") or result.get("result_text") or result.get("output") or ""
                if not llm_response:
                    logger.warning(f"Empty response: {result}")
                return llm_response

            except Exception as e:
                logger.error(f"Error during {self.model_id}@{self.api_url}: {e}")
                time.sleep(2 ** attempt + random.uniform(0, 1))

        logger.error(f"judge() failed after {max_retries} retries")
        return ""

    def judge(
        self,
        prompt: str = '',
        system_prompt: Optional[str] = None,
        messages: Optional[List[ChatMessage]] = None
    ) -> str:
        """
        Generate a response from the LLM based on the provided prompt and context.
        If messages is provided, it will be used as the input context.

        Args:
            prompt (str): The prompt to evaluate
            system_prompt (str, optional): The system prompt to use for the evaluation
            messages (List[ChatMessage], optional): A list of chat messages to include in the evaluation
        Returns:
            str: The response from the LLM
        """
        if os.environ.get('MODELSCOPE_USE_ERNIE_JUDGE', '1') == '1':
            return self.ernie_judge(prompt, system_prompt, messages)

        # parse messages
        if messages is not None:
            input_messages = messages
        else:
            system_content = system_prompt or self.system_prompt
            input_messages = [ChatMessageUser(content=prompt)]
            if system_content:
                input_messages.insert(0, ChatMessageSystem(content=system_content))
        try:
            # Send request using ServerModelAdapter
            response = self.model.generate(input_messages)

            # Extract content from response
            llm_response = response.completion
            return llm_response
        except Exception as e:
            logger.error(f'Error occurred during {self.model_id}@{self.api_url} LLM judge evaluation: {e}')
            return ''

    def build_prompt(self, pred: str, gold: str, question: Optional[str] = None):
        if question is None:
            question = 'Not provided'

        # check variables in prompt_template
        prompt = self.prompt_template
        if '{question}' in self.prompt_template:
            prompt = prompt.replace('{question}', question)
        if '{pred}' in self.prompt_template:
            prompt = prompt.replace('{pred}', pred)
        if '{gold}' in self.prompt_template:
            prompt = prompt.replace('{gold}', gold)
        return prompt

    def get_score(self, response: str) -> float:
        """
        Extract score from LLM response using the configured pattern and mapping.

        Args:
            response (str): The response from the LLM

        Returns:
            float: The numeric score extracted from the response
        """
        if response is None:
            return 0.0

        # choose extraction method based on score_type
        if self.score_type == JudgeScoreType.NUMERIC:
            return self._extract_numeric_score(response)
        elif self.score_type == JudgeScoreType.PATTERN:
            return self._extract_pattern_score(response)

    def _extract_numeric_score(self, response: str) -> Optional[float]:
        """extract numeric score from the response using the score_pattern"""
        match = re.search(self.score_pattern, response)

        if match:
            # try to convert each captured group to float
            for group in match.groups():
                if group is not None:
                    try:
                        return float(group)
                    except (ValueError, TypeError):
                        continue

            # if not found in groups, try the whole match
            try:
                return float(match.group(0))
            except (ValueError, TypeError):
                logger.warning(f'Failed to convert any extracted value to float from: {match.group(0)}')

        return None

    def _extract_pattern_score(self, response: str) -> float:
        """use the score_pattern to extract categorical scores"""
        match = re.search(self.score_pattern, response)
        if match:
            answer = match.group(0)
            return self.score_mapping.get(answer, 0.0)
        else:
            logger.warning(f"No match found for pattern '{self.score_pattern}' in response: {response}")
            return 0.0
