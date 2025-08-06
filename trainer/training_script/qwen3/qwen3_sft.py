import os
import torch
import torch.distributed as dist
import logging
os.environ["TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC"] = "3600"
from datetime import datetime
from typing import Any
from dotenv import load_dotenv
from trl import SFTTrainer, SFTConfig

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

from trainer.training_script.qwen3.dataprocessor.qwen3_sft_processor import processor

load_dotenv()
logger = logging.getLogger(__name__)
current_time = datetime.now().strftime("%Y%m%d_%H%M%S")

def trainer_instance(
    model: torch.nn.Module,
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
) -> None:
    """
    모델, 토크나이저, 데이터셋 및 설정 인자들을 받아 분산 학습을 수행하는 trainer 함수.
    상위 train.py에서 모델 및 기타 필요한 객체를 로드한 뒤 호출한다고 가정.
    """

    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer.tokenizer,
        mlm=False,
    )

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
        if base_args.use_dpotrainer:
            model.add_adapter("reference", peft_config)
        if training_args.bf16 and peftconfig_args.peft_type != "ia3":
            model = model.to(torch.bfloat16)

        model.print_trainable_parameters()

    else:
        peft_config = None

    args_dict = filter_unuse_parms(training_args)

    train_dataset, eval_dataset = processor(
        train_dataset=train_dataset,
        test_dataset=eval_dataset,
        tokenizer=tokenizer,
        tokenizer_max_len=data_args.tokenizer_max_len,
        data_args=data_args,
    )

    TrainerClass = SFTTrainer
    training_args = SFTConfig(**args_dict)
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    training_args = set_additional_parms(training_args, base_args, data_args, peftconfig_args, add_trainer_args)
    training_args.ddp_find_unused_parameters = True
    training_args.max_length = data_args.tokenizer_max_len
    optimizer_cls_and_kwargs = (
        get_optimizer_cls_and_kwargs(model, base_args, training_args)
        if base_args.use_stableadamw else None
    )

    common_args = {
        "model": model,
        "args": training_args,
        "processing_class": tokenizer,
        "train_dataset": train_dataset,
        "eval_dataset": eval_dataset,
        "data_collator": data_collator,
    }

    if  optimizer_cls_and_kwargs is not None:
        common_args["optimizer_cls_and_kwargs"] = optimizer_cls_and_kwargs

    trainer_instance = TrainerClass(**common_args)

    if hasattr(trainer_instance.model, "_set_static_graph"):
        trainer_instance.model._set_static_graph()

    return trainer_instance
