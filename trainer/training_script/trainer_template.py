import os
import torch
import torch.distributed as dist
import logging
os.environ["TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC"] = "3600"
from datetime import datetime
from dotenv import load_dotenv

from transformers import (
    set_seed,
    PreTrainedTokenizerBase,
    TrainingArguments,
)
from datasets import Dataset
from trainer.utils.mlflow_tool import init_mlflow

from trainer.arguments import (
    BaseArguments,
    ModelArguments,
    DataArguments,
    AdditionalTrainerArguments,
    SentenceTransformerArguments,
    DeepSpeedArguments,
    PeftConfigArguments
)

from trainer.training_script.gemma3.gemma3_composer import gemma3_trainer_composer
from trainer.utils.constants_loader import get_constant_list

TEXT_EMBEDDING_TASK = get_constant_list("TEXT_EMBEDDING")
SENTENCE_TRANSFORMER_TASK = get_constant_list("SENTENCE_TRANSFORMER")
CROSS_ENCODER_TASK = get_constant_list("CROSS_ENCODER")
MULTIMODAL_LANGUAGE_MODEL = get_constant_list("MULTIMODAL_LANGUAGE_MODEL")
CLASSIFICATION_TASK = get_constant_list("CLASSIFICATION")

load_dotenv()
logger = logging.getLogger(__name__)
current_time = datetime.now().strftime("%Y%m%d_%H%M%S")

def trainer(
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
) -> None:
    """
    모델, 토크나이저, 데이터셋 및 설정 인자들을 받아 분산 학습을 수행하는 trainer 함수.
    상위 train.py에서 모델 및 기타 필요한 객체를 로드한 뒤 호출한다고 가정.
    """
    HUGGING_FACE_TOKEN = data_args.hugging_face_token
    MLFLOW_URL = data_args.mlflow_url
    push_to_hub_last = training_args.push_to_hub

    if deepspeed_args.use_deepspeed:
        rank = int(os.environ.get("LOCAL_RANK", -1))
    else:
        if dist.is_initialized():
            rank = dist.get_rank()
        else:
            rank = 0

    set_seed(training_args.seed)

    if (training_args.report_to == 'mlflow') and (rank == 0):
        init_mlflow(
            mlflow_url=MLFLOW_URL,
            experiment_name=model_args.model_name,
            run_name=data_args.mlflow_run_id,
            mlflow_force_run=False,
            rank=rank,
        )

    #### MAKE TRAINER INSTANCE ####
    if model_args.language_model_class == "gemma3":
        trainer_instance = gemma3_trainer_composer(
            model, reference_model, tokenizer, train_dataset, eval_dataset,
            base_args, model_args, data_args, add_trainer_args,
            st_args, deepspeed_args, peftconfig_args, training_args
        )
    else:
        raise ValueError(f"Unsupported language model class: {model_args.language_model_class}")

    trainer_instance.train(resume_from_checkpoint=model_args.is_resume)

    if dist.is_initialized():
        if (rank == 0) and push_to_hub_last:
            retry_num = 0
            while retry_num < 5:
                try:
                    trainer_instance.push_to_hub(commit_message=f"{model_args.model_commit_msg} Done", token=HUGGING_FACE_TOKEN)
                    logger.info("Model pushed to hub.")
                    break
                except Exception as e:
                    retry_num += 1
                    logger.warning(f"Model Push Failed. Retry ... {retry_num} / 5")
                    logger.warning(f"{e}")

        torch.distributed.barrier()
        dist.destroy_process_group()
    else:
        if push_to_hub_last:
            retry_num = 0
            while retry_num < 5:
                try:
                    trainer_instance.push_to_hub(commit_message=f"{model_args.model_commit_msg} Done", token=HUGGING_FACE_TOKEN)
                    logger.info("Model pushed to hub.")
                    break
                except Exception as e:
                    retry_num += 1
                    logger.warning(f"Model Push Failed. Retry ... {retry_num} / 5")
                    logger.warning(f"{e}")
