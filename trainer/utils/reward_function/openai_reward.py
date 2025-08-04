"""OpenAI API based reward function for GRPO training."""

import openai
import json
from typing import List, Optional, Any, Dict
import logging
import time
from concurrent.futures import ThreadPoolExecutor


class OpenAIRewardFunction:
    """OpenAI API를 사용하여 completion의 품질을 평가하는 reward function 클래스."""

    def __init__(
        self,
        api_key: str,
        model: str = "gpt-4o-mini",
        max_score: float = 10.0,
        base_url: Optional[str] = None,
        evaluation_prompt_template: Optional[str] = None,
        max_workers: int = 5,
        request_timeout: int = 30,
        max_retries: int = 3,
        retry_delay: float = 1.0
    ):
        """
        Args:
            api_key: OpenAI API key
            model: OpenAI model to use for evaluation (default: gpt-4o-mini)
            max_score: Maximum score to be assigned (default: 10.0)
            base_url: Custom OpenAI API base URL (optional)
            evaluation_prompt_template: Custom evaluation prompt template
            max_workers: Maximum number of concurrent API requests
            request_timeout: Request timeout in seconds
            max_retries: Maximum number of retries for failed requests
            retry_delay: Delay between retries in seconds
        """
        self.client = openai.OpenAI(
            api_key=api_key,
            base_url=base_url
        )
        self.model = model
        self.max_score = max_score
        self.max_workers = max_workers
        self.request_timeout = request_timeout
        self.max_retries = max_retries
        self.retry_delay = retry_delay

        # Default evaluation prompt template with reference answer comparison
        self.evaluation_prompt = evaluation_prompt_template or """You are an AI assistant tasked with evaluating the quality of text completions by comparing them with a reference answer.

Given a prompt, a completion to evaluate, and a reference answer (ground truth), please evaluate the completion based on the following criteria:
1. Relevance: How well does the completion address the prompt compared to the reference?
2. Accuracy: How correct is the completion compared to the reference answer?
3. Completeness: How complete is the completion compared to the reference?
4. Quality: Overall quality of the completion compared to the reference standard?
5. Coherence: Is the completion as well-structured and clear as the reference?

Compare the completion against the reference answer to determine quality.

Please provide a score from 0 to {max_score}, where:
- 0-2: Much worse than reference (very poor quality)
- 3-4: Worse than reference (poor quality)
- 5-6: Similar to reference (average quality)
- 7-8: Better than reference (good quality)
- 9-{max_score}: Much better than reference (excellent quality)

Prompt: {prompt}
Reference Answer (Ground Truth): {reference_answer}
Completion to Evaluate: {completion}

Please respond with ONLY a single number (integer or decimal) between 0 and {max_score}. No explanation, no additional text, just the number."""

        self.logger = logging.getLogger(__name__)

    def _evaluate_single_completion(self, prompt: str, completion: str, reference_answer: Optional[str] = None) -> float:
        """단일 completion을 평가합니다."""
        if reference_answer is not None:
            # With reference answer comparison
            evaluation_text = self.evaluation_prompt.format(
                prompt=prompt,
                completion=completion,
                reference_answer=reference_answer,
                max_score=self.max_score
            )
        else:
            # Fallback to original evaluation without reference
            fallback_prompt = """You are an AI assistant tasked with evaluating the quality of text completions.

Given a prompt and its completion, please evaluate the completion based on the following criteria:
1. Relevance: How well does the completion address the prompt?
2. Coherence: Is the completion logically structured and easy to follow?
3. Accuracy: Is the information provided correct and factual?
4. Helpfulness: How useful is the completion to someone asking the question?
5. Completeness: Does the completion adequately address all aspects of the prompt?

Please provide a score from 0 to {max_score}, where:
- 0-2: Very poor quality
- 3-4: Poor quality
- 5-6: Average quality
- 7-8: Good quality
- 9-{max_score}: Excellent quality

Prompt: {prompt}
Completion: {completion}

Please respond with ONLY a single number (integer or decimal) between 0 and {max_score}. No explanation, no additional text, just the number."""

            evaluation_text = fallback_prompt.format(
                prompt=prompt,
                completion=completion,
                max_score=self.max_score
            )

        print(f"[DEBUG] Evaluation text: '{evaluation_text}'")

        for attempt in range(self.max_retries):
            try:
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {"role": "user", "content": evaluation_text}
                    ],
                    max_completion_tokens=2048,    # Increased to ensure complete response
                    timeout=self.request_timeout
                )

                score_text = response.choices[0].message.content.strip()

                # Extract numeric score
                try:
                    score = float(score_text)
                    # Clamp score to valid range
                    score = max(0.0, min(score, self.max_score))
                    print(f"[DEBUG] Parsed score: {score}")
                    return score
                except ValueError:
                    print(f"[DEBUG] Failed to parse as float: '{score_text}'")
                    # Try to extract first number from the response
                    import re
                    # More comprehensive regex to catch decimals and integers
                    numbers = re.findall(r'\d+(?:\.\d+)?', score_text)
                    print(f"[DEBUG] Found numbers: {numbers}")
                    if numbers:
                        score = float(numbers[0])
                        score = max(0.0, min(score, self.max_score))
                        print(f"[DEBUG] Extracted score: {score}")
                        return score
                    else:
                        # Try to find any number-like pattern
                        import re
                        all_numbers = re.findall(r'\d+', score_text)
                        print(f"[DEBUG] Found digits: {all_numbers}")
                        if all_numbers:
                            score = float(all_numbers[0])
                            score = max(0.0, min(score, self.max_score))
                            print(f"[DEBUG] Fallback extracted score: {score}")
                            return score
                        else:
                            print(f"[ERROR] No numeric content found in: '{score_text}'")
                            raise ValueError(f"No numeric score found in response: '{score_text}'")

            except Exception as e:
                self.logger.warning("Attempt %d failed: %s", attempt + 1, str(e))
                if attempt < self.max_retries - 1:
                    time.sleep(self.retry_delay * (2 ** attempt))  # Exponential backoff
                else:
                    self.logger.error("All attempts failed for completion evaluation: %s", str(e))
                    # Return middle score as fallback
                    return self.max_score / 2.0

    def evaluate_completions(self, prompts: List[str], completions: List[str], reference_answers: Optional[List[str]] = None) -> List[float]:
        """여러 completion을 병렬로 평가합니다."""
        if len(prompts) != len(completions):
            raise ValueError("prompts and completions must have the same length")

        if reference_answers is not None and len(reference_answers) != len(prompts):
            raise ValueError("reference_answers length must match prompts and completions length")

        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            futures = []
            for i, (prompt, completion) in enumerate(zip(prompts, completions)):
                reference = reference_answers[i] if reference_answers else None
                futures.append(
                    executor.submit(self._evaluate_single_completion, prompt, completion, reference)
                )

            scores = []
            for future in futures:
                try:
                    score = future.result(timeout=self.request_timeout + 10)
                    scores.append(score)
                except Exception as e:
                    self.logger.error("Failed to get score: %s", str(e))
                    scores.append(self.max_score / 2.0)  # Fallback score

        return scores


def openai_quality_reward_func(
    prompts: List[str],
    completions: List[str],
    api_key: str,
    model: str = "gpt-4o-mini",
    max_score: float = 10.0,
    normalize_to_range: Optional[tuple] = None,
    ground_truth: Optional[List[str]] = None,  # Changed from 'output' to 'ground_truth'
    **kwargs
) -> List[float]:
    """
    OpenAI API를 사용하여 completion의 품질을 평가하는 reward function.

    Args:
        prompts: 입력 프롬프트 리스트
        completions: 평가할 completion 리스트
        api_key: OpenAI API key
        model: 사용할 OpenAI 모델 (default: gpt-4o-mini)
        max_score: 최대 점수 (default: 10.0)
        normalize_to_range: 점수를 특정 범위로 정규화 (예: (-1, 1))
        ground_truth: Ground truth 답변 리스트 (reference answers)
        **kwargs: 기타 인자들 (trainer_state 등)

    Returns:
        각 completion에 대한 점수 리스트
    """
    evaluator = OpenAIRewardFunction(
        api_key=api_key,
        model=model,
        max_score=max_score
    )

    scores = evaluator.evaluate_completions(prompts, completions, ground_truth)

    # 점수 정규화 (옵션)
    if normalize_to_range:
        min_target, max_target = normalize_to_range
        # 0 ~ max_score 범위를 min_target ~ max_target 범위로 변환
        scores = [
            min_target + (score / max_score) * (max_target - min_target)
            for score in scores
        ]

    return scores


def create_openai_reward_func(
    api_key: str,
    model: str = "gpt-4o-mini",
    max_score: float = 10.0,
    normalize_to_range: Optional[tuple] = None,
    evaluation_prompt_template: Optional[str] = None,
    **evaluator_kwargs
):
    """
    사전 설정된 OpenAI reward function을 생성하는 팩토리 함수.

    Args:
        api_key: OpenAI API key
        model: 사용할 OpenAI 모델
        max_score: 최대 점수
        normalize_to_range: 점수 정규화 범위
        evaluation_prompt_template: 커스텀 평가 프롬프트 템플릿
        **evaluator_kwargs: OpenAIRewardFunction에 전달될 추가 인자들

    Returns:
        GRPOTrainer에서 사용할 수 있는 reward function
    """
    evaluator = OpenAIRewardFunction(
        api_key=api_key,
        model=model,
        max_score=max_score,
        evaluation_prompt_template=evaluation_prompt_template,
        **evaluator_kwargs
    )

    def reward_func(prompts: List[str], completions: List[str], **kwargs) -> List[float]:
        # Extract ground_truth from kwargs if available
        ground_truth = kwargs.get('ground_truth', None)

        print(f"[DEBUG] Reward function called with:")
        print(f"[DEBUG] - Number of prompts: {len(prompts)}")
        print(f"[DEBUG] - Number of completions: {len(completions)}")
        print(f"[DEBUG] - Available kwargs keys: {list(kwargs.keys())}")
        print(f"[DEBUG] - Ground truth found: {ground_truth is not None}")

        if ground_truth is None:
            raise ValueError("[ERROR] Ground truth is required but not found in dataset. Please ensure your dataset has a 'ground_truth' column.")

        print(f"[DEBUG] - Number of ground_truth entries: {len(ground_truth)}")
        print(f"[DEBUG] - First ground_truth sample: {ground_truth[0][:100] if len(ground_truth) > 0 else 'None'}...")

        scores = evaluator.evaluate_completions(prompts, completions, ground_truth)

        if normalize_to_range:
            min_target, max_target = normalize_to_range
            scores = [
                min_target + (score / max_score) * (max_target - min_target)
                for score in scores
            ]

        print(f"[DEBUG] Final scores: {scores}")
        return scores

    return reward_func


def create_ground_truth_comparison_reward_func(
    api_key: str,
    model: str = "gpt-4o-mini",
    max_score: float = 10.0,
    normalize_to_range: Optional[tuple] = None,
    **evaluator_kwargs
):
    """
    Ground truth 기반 비교 평가를 위한 OpenAI reward function을 생성하는 팩토리 함수.

    이 함수는 dataset의 'output' column (ground truth)을 참조 답변으로 사용하여
    생성된 completion의 품질을 평가합니다.

    Args:
        api_key: OpenAI API key
        model: 사용할 OpenAI 모델
        max_score: 최대 점수
        normalize_to_range: 점수 정규화 범위
        **evaluator_kwargs: OpenAIRewardFunction에 전달될 추가 인자들

    Returns:
        GRPOTrainer에서 사용할 수 있는 ground truth 비교 reward function
    """
    # 법률 조항 및 핵심 요소 검증에 특화된 prompt template
    ground_truth_prompt = """You are an AI assistant specialized in evaluating legal text completions by comparing them with reference answers.

Given a prompt, a completion to evaluate, and a reference answer, please evaluate the completion based on legal accuracy and core elements.

Evaluation criteria (특히 법률 조항 언급 시):
1. Legal Article Citation Accuracy: If legal articles/clauses (조항) are mentioned, are they correctly cited and relevant?
2. Core Legal Elements: Does the completion include the essential legal elements present in the reference?
3. Legal Reasoning: Is the legal reasoning sound and appropriate, even if different from the reference?
4. Factual Accuracy: Are the legal facts and interpretations correct?
5. Completeness: Are the key legal points adequately addressed?

Important: The completion does NOT need to be identical to the reference. Different approaches and reasoning can be equally valid as long as they are legally sound and include core elements.

Please provide a score from 0 to {max_score}, where:
- 0-2: Incorrect legal citations or missing critical legal elements
- 3-4: Some legal inaccuracies or incomplete coverage of core elements
- 5-6: Generally accurate with minor issues in legal citations or elements
- 7-8: Accurate legal citations and good coverage of core elements
- 9-{max_score}: Excellent legal accuracy with proper citations and comprehensive core elements

Prompt: {prompt}
Reference Answer (Legal Reference): {reference_answer}
Completion to Evaluate: {completion}

Please respond with ONLY a single number (integer or decimal) between 0 and {max_score}. No explanation, no additional text, just the number."""

    evaluator = OpenAIRewardFunction(
        api_key=api_key,
        model=model,
        max_score=max_score,
        evaluation_prompt_template=ground_truth_prompt,
        **evaluator_kwargs
    )

    def reward_func(prompts: List[str], completions: List[str], ground_truth, **kwargs) -> List[float]:

        print(f"[DEBUG] Ground truth comparison reward function called with:")
        print(f"[DEBUG] - Number of prompts: {len(prompts)}")
        print(f"[DEBUG] - Number of completions: {len(completions)}")
        print(f"[DEBUG] - Available kwargs keys: {list(kwargs.keys())}")
        print(f"[DEBUG] - Ground truth found: {ground_truth is not None}")

        if ground_truth is None:
            raise ValueError("[ERROR] Ground truth is required but not found in dataset. Please ensure your dataset has a 'ground_truth' column.")

        print(f"[DEBUG] - Number of ground_truth entries: {len(ground_truth)}")
        print(f"[DEBUG] - First ground_truth sample: {ground_truth[0][:100] if len(ground_truth) > 0 else 'None'}...")

        print("[INFO] Using ground truth for comparison evaluation.")
        scores = evaluator.evaluate_completions(prompts, completions, ground_truth)

        if normalize_to_range:
            min_target, max_target = normalize_to_range
            scores = [
                min_target + (score / max_score) * (max_target - min_target)
                for score in scores
            ]

        print(f"[DEBUG] Final scores: {scores}")
        return scores

    return reward_func
