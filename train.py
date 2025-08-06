import os

import json
import logging
import torch
import torch.distributed as dist
import time
import shutil

from datetime import datetime
from huggingface_hub import HfApi, login
from transformers import (
    AutoTokenizer,
    AutoProcessor,
    TrainingArguments,
    HfArgumentParser,
    set_seed,
)

from trainer.arguments import BaseArguments, ModelArguments, DataArguments, AdditionalTrainerArguments, SentenceTransformerArguments, DeepSpeedArguments, PeftConfigArguments
from trainer.utils.dataloader_toolkit import DataLoader
from trainer.utils.deepspeed_config_loader import get_deepspeed_config
from trainer.utils.model_loader import huggingface_model_load
from trainer.utils.tokenizer_loader import huggingface_tokenizer_load
from trainer.utils.tools import select_best_checkpoint_folder, parse_csv_string, selective_freeze, print_trainable_parameters
from trainer.training_script.trainer_template import trainer
from trainer.dataloader import serve_dataset
from trainer.utils.constants_loader import get_constant_list

os.environ["TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC"] = "3600"
# os.environ["NCCL_DEBUG"] = "INFO"

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(message)s",
    level=logging.INFO
)
logger = logging.getLogger(__name__)

num_gpus = torch.cuda.device_count()
current_time = datetime.now().strftime("%Y%m%d_%H%M%S")

TEXT_EMBEDDING_TASK = get_constant_list("TEXT_EMBEDDING")
SENTENCE_TRANSFORMER_TASK = get_constant_list("SENTENCE_TRANSFORMER")
CROSS_ENCODER_TASK = get_constant_list("CROSS_ENCODER")
MULTIMODAL_LANGUAGE_MODEL = get_constant_list("MULTIMODAL_LANGUAGE_MODEL")
CLASSIFICATION_TASK = get_constant_list("CLASSIFICATION")

def get_task_name(task):
    if task in SENTENCE_TRANSFORMER_TASK:
        return "Sentence-Transformer"
    elif task in TEXT_EMBEDDING_TASK:
        return "Text Embedding"
    elif task in CROSS_ENCODER_TASK:
        return "Cross-Encoder"
    elif task in MULTIMODAL_LANGUAGE_MODEL:
        return "Multimodal Language Model"
    elif task == "mlm":
        return "Masked Language Modeling"
    elif task == "clm":
        return "Causal Language Modeling"
    elif task == CLASSIFICATION_TASK:
        return "Sequence Classification"
    else:
        return "unknown"

def main():
    parser = HfArgumentParser((BaseArguments, ModelArguments, DataArguments, AdditionalTrainerArguments, SentenceTransformerArguments, DeepSpeedArguments, PeftConfigArguments, TrainingArguments))
    base_args, model_args, data_args, add_trainer_args, st_args, deepspeed_args, peftconfig_args, training_args = parser.parse_args_into_dataclasses()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("[INFO] Device: %s (GPUs: %s)", device, num_gpus)

    HUGGING_FACE_USER_ID = data_args.hugging_face_user_id
    HUGGING_FACE_TOKEN = data_args.hugging_face_token
    MLFLOW_URL = data_args.mlflow_url

    if deepspeed_args.use_deepspeed:
        rank = int(os.environ.get("LOCAL_RANK", -1))
        torch.cuda.set_device(rank)

        if dist.is_initialized():
            logger.info("[INFO] Distributed process group already initialized. Rank: %s", dist.get_rank())
            dist.barrier(device_ids=[rank])
        else:
            dist.init_process_group(backend="nccl", init_method="env://", timeout=datetime.timedelta(seconds=7200))
            logger.info("[INFO] Initialized distributed process group. Rank: %s", dist.get_rank())
            rank = dist.get_rank()

        try:
            deepspeed_config = get_deepspeed_config(training_args=training_args, deepspeed_args=deepspeed_args)
        except:
            raise ValueError("Fail to load deep speed config!")

        training_args.deepspeed = deepspeed_config

    else:
        if dist.is_initialized():
            rank = dist.get_rank()
            logger.info("[INFO] Distributed process group already initialized. Rank: %s", dist.get_rank())
            dist.barrier(device_ids=[rank])
        else:
            try:
                local_rank = int(os.environ.get("LOCAL_RANK", "0"))
                torch.cuda.set_device(local_rank)
                dist.init_process_group(backend="nccl", init_method="env://", timeout=datetime.timedelta(seconds=7200))
                logger.info("[INFO] Initialized distributed process group. Rank: %s", dist.get_rank())
                rank = dist.get_rank()
            except:
                logger.warning("[WARNING] Please Check Local Rank Error.")
                rank = 0

        training_args.deepspeed = None

    print("="*50)
    print("="*50)
    print(f"============== [INFO] local_rank: {rank} ==============")
    print(f"============== [INFO] local_rank: {rank} ==============")

    if len(model_args.model_name) < 1:
        logger.warning(f"[WARNING] Unvalid Project Name. Set 'default_{current_time}'")
        model_args.model_name = f'default_{current_time}'

    if len(HUGGING_FACE_USER_ID) < 1:
        project_default_path = f"result__default__{model_args.model_name}"
        hub_default_path = None
        if training_args.push_to_hub:
            logger.warning("[WARNING] Unset Hugging Face User ID. Disabling 'Push to hub'.")
            training_args.push_to_hub = False
    else:
        project_default_path = f"result__{HUGGING_FACE_USER_ID}__{model_args.model_name}"
        hub_default_path = f"{HUGGING_FACE_USER_ID}/{model_args.model_name}"

    if (training_args.output_dir is None) or (len(training_args.output_dir) < 1):
        training_args.output_dir = project_default_path

    else:
        training_args.output_dir = os.path.join(training_args.output_dir, model_args.model_name)

    os.makedirs(training_args.output_dir, exist_ok=True)

    if (training_args.hub_model_id is None) or (len(training_args.hub_model_id) < 1):
        training_args.hub_model_id = hub_default_path

    trainers = [
        ("SFT Trainer", base_args.use_sfttrainer),
        ("DPO Trainer", base_args.use_dpotrainer),
        ("PPO Trainer", base_args.use_ppotrainer),
        ("GRPO Trainer", base_args.use_grpotrainer),
        ("KL-SFT Trainer", base_args.use_custom_kl_sfttrainer),
    ]
    selected_trainer = [name for name, value in trainers if value]
    if len(selected_trainer) > 1:
        raise ValueError("[FATAL ERROR] Only one trainer can be selected. Please check the trainer selection.")
    elif len(selected_trainer) == 1:
        if (rank == 0):
            print(f"[INFO] Selected trainer: {selected_trainer[0]}")
        if selected_trainer[0] != "SFT Trainer":
            is_trl = True
        else:
            is_trl = False
    else:
        if (rank == 0):
            print("[INFO] No TRL Trainer selected. Proceeding with Hugging Face Default Trainer.")
        is_trl = False

    if (rank == 0):
        print(f'[INFO] IS TRL USING? = {is_trl}')

    if peftconfig_args.use_peft:
        try:
            peftconfig_args.lora_target_modules = parse_csv_string(peftconfig_args.lora_target_modules)
            peftconfig_args.lora_modules_to_save = parse_csv_string(peftconfig_args.lora_modules_to_save)
        except:
            raise ValueError("[FATAL ERROR] Use Lora, but 'lora_target_modules' is not valid. It should be comma seperated 'str'.")

    if model_args.selected_parameter and model_args.selected_parameter != "none" and len(model_args.selected_parameter) > 0:
        try:
            parsed = parse_csv_string(model_args.selected_parameter)
            if isinstance(parsed, list) and len(parsed) > 0:
                model_args.selected_parameter = parsed
                if (rank == 0):
                    print(f"[INFO] Selected parameter: {model_args.selected_parameter}")
            else:
                raise ValueError("[FATAL ERROR] 'selected_parameter' is not valid. It should be comma separated 'str' with at least one value.")
        except Exception:
            raise ValueError("[FATAL ERROR] 'selected_parameter' is not valid. It should be comma separated 'str'.")

    if (rank == 0):
        if (os.path.exists(training_args.output_dir)
            and os.listdir(training_args.output_dir)
            and training_args.do_train
            and not training_args.overwrite_output_dir):
            raise ValueError(
                f"[ERROR] Output directory ({training_args.output_dir}) already exists and is not empty. Use --overwrite_output_dir to overcome."
            )
        else:
            os.makedirs(training_args.output_dir, exist_ok=True)
            training_method = get_task_name(model_args.model_train_method)
            metadata = {
                "base_model": model_args.model_name_or_path,
                "training_method": training_method,
                "commit_msg": model_args.model_commit_msg,
                "user_name": HUGGING_FACE_USER_ID,
                "use_deepspeed": deepspeed_args.use_deepspeed,
                "use_peft": peftconfig_args.use_peft,
                "use_sfttrainer": base_args.use_sfttrainer,
                "use_dpotrainer": base_args.use_dpotrainer,
                "use_ppotrainer": base_args.use_ppotrainer,
                "use_grpotrainer": base_args.use_grpotrainer,
                "use_kl_sfttrainer": base_args.use_custom_kl_sfttrainer,
                "use_stableadamw": base_args.use_stableadamw,
                "use_attn_implementation": base_args.use_attn_implementation,
            }
            metadata_file_path = os.path.join(training_args.output_dir, "metadata.json")
            with open(metadata_file_path, "w", encoding="utf-8") as f:
                json.dump(metadata, f, indent=4)
        logger.warning(
            "Process rank: %s, device: %s, n_gpu: %s, distributed training: %s, 16-bits training: %s",
            training_args.local_rank,
            training_args.device,
            training_args.n_gpu,
            bool(training_args.local_rank != -1),
            training_args.fp16,
        )
        logger.info("Training/evaluation parameters: %s", training_args)
        logger.info("Model parameters: %s", model_args)
        logger.info("Data parameters: %s", data_args)

        if (HUGGING_FACE_TOKEN is not None) and (len(HUGGING_FACE_TOKEN) > 0):
            login(token=HUGGING_FACE_TOKEN)

    set_seed(training_args.seed)

    if (training_args.push_to_hub) and (rank == 0):
        api = HfApi()
        api.create_repo(repo_id=training_args.hub_model_id, exist_ok=True)
        logger.info(f"Repository {training_args.hub_model_id} created or exists already.")

        try:
            api.upload_file(
                path_or_fileobj=metadata_file_path,
                path_in_repo="metadata.json",
                repo_id=training_args.hub_model_id,
                repo_type="model",
                token=HUGGING_FACE_TOKEN,
            )
            logger.info(f"Metadata file uploaded to {training_args.hub_model_id}.")
        except Exception as e:
            logger.error(f"Failed to upload metadata file: {e}")

    else:
        time.sleep(3)

    if MLFLOW_URL and (len(MLFLOW_URL) > 0) and (rank == 0):
        training_args.report_to = "mlflow"
        logger.info(f"Reporting to %s. Logging with MLFlow.{MLFLOW_URL}", training_args.report_to)
    else:
        training_args.report_to = None

    if base_args.use_attn_implementation:
        attn_implementation = base_args.attn_implementation
        if training_args.fp16:
            torch_dtype = torch.float16
        elif training_args.bf16:
            torch_dtype = torch.bfloat16
        else:
            torch_dtype = "auto"

        st_model_arg = {
            "attn_implementation": base_args.attn_implementation,
            "torch_dtype": torch.float16,
        }
    else:
        attn_implementation = None
        if training_args.fp16:
            torch_dtype = torch.float16
        elif training_args.bf16:
            torch_dtype = torch.bfloat16
        else:
            torch_dtype = "auto"
        st_model_arg = {}

    try:
        tokenizer = huggingface_tokenizer_load(
            model_path=model_args.model_name_or_path,
            tokenizer_name=model_args.tokenizer_name,
            max_seq_length=data_args.tokenizer_max_len,
            model_subfolder=model_args.model_subfolder,
            language_model_class=model_args.language_model_class,
        )
        logger.info("[INFO] Tokenizer loaded successfully.")

    except Exception as e:
        logger.error("[FATAL ERROR] Fail to pull tokenizer: %s", e)
        raise RuntimeError("[FATAL ERROR] Fail to pull tokenizer")

    if data_args.tokenizer_max_len == 0:
        data_args.tokenizer_max_len = None

    # 향후 확인: 실제로 huggingface나 local은 따로 분리할 필요가 없으나 .. 혹시 다른 경우가 존재할 수도 있어 일단 나눠서 작성함.
    # 만약 따로 문제되는 것이 없다면 하나로 통합할 것. 혹은 local method 자체를 삭제해도 괜찮을듯.
    model = None

    if model_args.model_load_method == 'huggingface':
        try:
            model = huggingface_model_load(
                model_args.model_name_or_path,
                model_args.model_train_method,
                num_labels=add_trainer_args.num_labels,
                st_model_arg=st_model_arg,
                max_seq_length=data_args.tokenizer_max_len,
                pooling_mode=st_args.st_pooling_mode,
                dense_feature=st_args.st_dense_feature,
                subfolder=model_args.model_subfolder,
                token=data_args.hugging_face_token,
                language_model_class=model_args.language_model_class,
                device=device,
                attn_implementation=attn_implementation,
                torch_dtype=torch_dtype,
            )
        except Exception as e:
            logger.error("[FATAL ERROR] Fail to pull model: %s", e)
            raise RuntimeError("[FATAL ERROR] Fail to pull model")

    model.to(device)
    if isinstance(model_args.selected_parameter, list) and len(model_args.selected_parameter) > 0:
        selective_freeze(model, model_args.selected_parameter)
        if (rank == 0):
            for name, param in model.named_parameters():
                if param.requires_grad:
                    print(f"{name}: requires_grad={param.requires_grad}")
            print_trainable_parameters(model)

    reference_model = None

    try:
        train_dataset, test_dataset = serve_dataset(
            train_data=data_args.train_data,
            train_data_dir=data_args.train_data_dir,
            train_data_split=data_args.train_data_split,
            test_data=data_args.test_data,
            test_data_dir=data_args.test_data_dir,
            test_data_split=data_args.test_data_split,
            train_test_split_ratio=data_args.train_test_split_ratio,
            dataset_main_colunm=data_args.dataset_main_colunm,
            dataset_sub_colunm=data_args.dataset_sub_colunm,
            dataset_minor_colunm=data_args.dataset_minor_colunm,
            dataset_last_colunm=data_args.dataset_last_colunm,
            data_filtering=data_args.data_filtering,
            data_args=data_args
        )
    except Exception as e:
        logger.error("[FATAL ERROR] Dataset Load Error: %s", e)
        raise RuntimeError("[FATAL ERROR] Dataset Load Error")

    try:
        trainer(model, reference_model, tokenizer, train_dataset, test_dataset, base_args, model_args, data_args, add_trainer_args, st_args, deepspeed_args, peftconfig_args, training_args)

    except Exception as e:
        logger.error("[FATAL ERROR] Train Error: %s", e)
        if dist.is_initialized():
            dist.destroy_process_group()
        exit(1)

    else:
        if dist.is_initialized():
            dist.destroy_process_group()

    if dist.is_initialized():
        dist.destroy_process_group()

if __name__ == "__main__":
    main()
