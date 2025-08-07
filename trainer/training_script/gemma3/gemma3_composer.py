import os
import torch
os.environ["TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC"] = "3600"
from transformers import (
    PreTrainedTokenizerBase,
    TrainingArguments,
)
from datasets import Dataset
from trainer.arguments import (
    BaseArguments,
    ModelArguments,
    DataArguments,
    AdditionalTrainerArguments,
    SentenceTransformerArguments,
    DeepSpeedArguments,
    PeftConfigArguments
)

from trainer.training_script.gemma3.gemma3_clm import trainer_instance as gemma3_clm_trainer_instance
from trainer.training_script.gemma3.gemma3_sft import trainer_instance as gemma3_sft_trainer_instance
from trainer.training_script.gemma3.gemma3_grpo_api_reward import trainer_instance as gemma3_grpo_api_reward_trainer_instance

def gemma3_trainer_composer(
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
):
    """
    모델, 토크나이저, 데이터셋 및 설정 인자들을 받아 분산 학습을 수행하는 trainer 함수.
    상위 train.py에서 모델 및 기타 필요한 객체를 로드한 뒤 호출한다고 가정.
    """
    if model_args.language_model_class == "gemma3":
        if base_args.use_sfttrainer:
            return gemma3_sft_trainer_instance(
                model = model,
                tokenizer = tokenizer,
                train_dataset = train_dataset,
                eval_dataset = eval_dataset,
                base_args = base_args,
                model_args = model_args,
                data_args = data_args,
                add_trainer_args = add_trainer_args,
                st_args = st_args,
                deepspeed_args = deepspeed_args,
                peftconfig_args = peftconfig_args,
                training_args = training_args
            )
        elif base_args.use_grpotrainer:
            print("[INFO] Using Gemma3 GRPO API Reward Trainer.")
            if model_args.model_train_method == "grpo_api_reward":
                trainer_instance = gemma3_grpo_api_reward_trainer_instance(
                    model = model,
                    reference_model = reference_model,
                    tokenizer = tokenizer,
                    train_dataset = train_dataset,
                    eval_dataset = eval_dataset,
                    base_args = base_args,
                    model_args = model_args,
                    data_args = data_args,
                    add_trainer_args = add_trainer_args,
                    st_args = st_args,
                    deepspeed_args = deepspeed_args,
                    peftconfig_args = peftconfig_args,
                    training_args = training_args
                )
                return trainer_instance
        else:
            return gemma3_clm_trainer_instance(
                model = model,
                tokenizer = tokenizer,
                train_dataset = train_dataset,
                eval_dataset = eval_dataset,
                base_args = base_args,
                model_args = model_args,
                data_args = data_args,
                add_trainer_args = add_trainer_args,
                st_args = st_args,
                deepspeed_args = deepspeed_args,
                peftconfig_args = peftconfig_args,
                training_args = training_args
            )
    else:
        raise ValueError(f"Incorrect language model class: {model_args.language_model_class}")
