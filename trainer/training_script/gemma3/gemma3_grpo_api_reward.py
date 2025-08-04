import os
import torch
import torch.distributed as dist
import logging
os.environ["TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC"] = "3600"
from datetime import datetime
from typing import Any
from dotenv import load_dotenv
from trl import GRPOTrainer, GRPOConfig

from peft import get_peft_model, prepare_model_for_kbit_training
from transformers import (
    DataCollatorForLanguageModeling,
    PreTrainedTokenizerBase,
    TrainingArguments,
)
from datasets import Dataset
from trainer.utils.tools import is_quantized_model
from trainer.utils.peft_config_loader import get_peft_config
from trainer.utils.optimizer_loader import get_optimizer_cls_and_kwargs
from trainer.utils.env_config import get_openai_api_key
from trainer.utils.trainer_toolkit import filter_unuse_parms, set_additional_parms

from trainer.arguments import (
    BaseArguments,
    ModelArguments,
    DataArguments,
    AdditionalTrainerArguments,
    SentenceTransformerArguments,
    DeepSpeedArguments,
    PeftConfigArguments
)
from trainer.training_script.gemma3.dataprocessor.gemma3_grpo_api_reward import processor
from trainer.utils.reward_function.openai_reward import create_openai_reward_func

load_dotenv()
logger = logging.getLogger(__name__)
current_time = datetime.now().strftime("%Y%m%d_%H%M%S")

def trainer_instance(
    model: torch.nn.Module,
    reference_model: torch.nn.Module,
    tokenizer: PreTrainedTokenizerBase,
    train_dataset: Dataset,
    eval_dataset: Dataset,
    base_args: BaseArguments,
    model_args: ModelArguments,
    data_args: DataArguments,
    add_trainer_args: AdditionalTrainerArguments,
    st_args: SentenceTransformerArguments,
    deepspeed_args: DeepSpeedArguments,
    peftconfig_args: PeftConfigArguments,
    training_args: TrainingArguments,
) -> GRPOTrainer:
    """
    모델, 토크나이저, 데이터셋 및 설정 인자들을 받아 분산 학습을 수행하는 trainer 함수.
    상위 train.py에서 모델 및 기타 필요한 객체를 로드한 뒤 호출한다고 가정.
    """

    ## 데이터셋 프로세싱

    if is_quantized_model(model):
        print("[INFO] Quantinization Model is detected. Use prepare_model_for_kbit_training")
        model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=training_args.gradient_checkpointing)

    if training_args.gradient_checkpointing:
        model.config.use_cache = False
        model.enable_input_require_grads()
        model.gradient_checkpointing_enable()

    if peftconfig_args.use_peft:
        peft_config = get_peft_config(peftconfig_args, model_args.model_train_method)
        model = get_peft_model(model, peft_config)
        model.add_adapter("reference", peft_config)
        if training_args.bf16 and peftconfig_args.peft_type != "ia3":
            model = model.to(torch.bfloat16)

        model.print_trainable_parameters()

    else:
        peft_config = None

    args_dict = filter_unuse_parms(training_args)

    # mmlm = Multi-Modal LM에서는 Autoprocessor를 기대하니, tokenizer를 명시.
    # remove_unused_columns 중복 전달 방지를 위해 args_dict에서 제거
    args_dict.pop('remove_unused_columns', None)

    training_args = GRPOConfig(
        max_prompt_length=4096,
        max_completion_length=4096,
        use_vllm=True,
        vllm_mode="colocate",
        vllm_tensor_parallel_size=8,
        vllm_gpu_memory_utilization=0.1,
        remove_unused_columns=False,
        **args_dict)
    # 기본값이 DataCollatorForLanguageModeling임
    # 아래 내용을 제거해도 괜찮은 것인지 파악.

    training_args = set_additional_parms(training_args, base_args, data_args, peftconfig_args, add_trainer_args)
    training_args.ddp_find_unused_parameters = True
    optimizer_cls_and_kwargs = (
        get_optimizer_cls_and_kwargs(model, base_args, training_args)
        if base_args.use_stableadamw else None
    )

    train_dataset, eval_dataset = processor(
        train_dataset=train_dataset,
        test_dataset=eval_dataset,
        tokenizer=tokenizer.tokenizer,
        tokenizer_max_len=data_args.tokenizer_max_len,
        data_args=data_args,
    )

    # 이거 프로세싱 클래스라 그냥 AutoProcess 넣었는데 왜 AutoProcess.tokenizer 넣어야 작동하는지 모르겠음
    #TODO 아마 멀티모달 모델이라도 멀티모달 아닌 경우에는 이렇게 넣어야 하는듯. 추후 Method 살펴보고 수정할 필요가 있음.
    common_args = {
        "model": model,
        "args": training_args,
        "reward_processing_classes": tokenizer.tokenizer,
        "train_dataset": train_dataset,
        "eval_dataset": eval_dataset,
    }

    openai_api_reward_func = create_openai_reward_func(
        api_key=get_openai_api_key(),
        model="o4-mini-2025-04-16",
        normalize_to_range=(-1, 1),  # Add normalization for GRPO
    )

    if openai_api_reward_func is not None:
        # GRPO expects a list of reward functions
        common_args["reward_funcs"] = [openai_api_reward_func]

    if optimizer_cls_and_kwargs is not None:
        common_args["optimizer_cls_and_kwargs"] = optimizer_cls_and_kwargs

    print("[INFO] Dataset Sample: ", train_dataset[0])

    trainer_instance = GRPOTrainer(**common_args)

    if hasattr(trainer_instance.model, "_set_static_graph"):
        trainer_instance.model._set_static_graph()

    return trainer_instance


if __name__ == "__main__":
    # OpenAI reward function 테스트
    print("[INFO] Testing OpenAI Reward Function...")

    # reward function 생성
    openai_api_reward_func = create_openai_reward_func(
        api_key=get_openai_api_key(),
        model="gpt-4.1-nano-2025-04-14"
    )

    # 테스트용 데이터
    test_prompts = [
        "What is the capital of France?",
        "Explain the concept of machine learning in simple terms.",
        "Write a short poem about spring."
    ]

    test_completions = [
        "The capital of France is Paris.",
        "Machine learning is a type of artificial intelligence where computers learn from data.",
        "Spring brings flowers and sunshine, nature awakens from winter's rest."
    ]

    try:
        # reward function 실행
        print(f"[INFO] Testing with {len(test_prompts)} prompt-completion pairs...")
        scores = openai_api_reward_func(test_prompts, test_completions)

        print("[INFO] Results:")
        for i, (prompt, completion, score) in enumerate(zip(test_prompts, test_completions, scores)):
            print(f"Test {i+1}:")
            print(f"  Prompt: {prompt}")
            print(f"  Completion: {completion}")
            print(f"  Score: {score}")
            print()

        print(f"[INFO] Average Score: {sum(scores) / len(scores):.2f}")
        print("[INFO] Test completed successfully!")

    except Exception as e:
        print(f"[ERROR] Test failed: {str(e)}")
        import traceback
        traceback.print_exc()
