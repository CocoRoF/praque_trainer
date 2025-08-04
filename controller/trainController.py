from fastapi import APIRouter, BackgroundTasks, HTTPException
from pydantic import BaseModel, Field
from typing import Optional, Dict, Any, Union, List
import threading
import logging
import os
import signal
import subprocess
import json
from datetime import datetime
import glob
from pathlib import Path
from trainer.utils.mlflow_tool import get_mlflow_info
from trainer.utils.env_config import get_huggingface_token, get_huggingface_user_id, get_minio_config, get_mlflow_url

# from zoneinfo import ZoneInfo

# 로깅 설정
logger = logging.getLogger("polar-trainer")

# 작업 데이터를 저장할 디렉터리
JOB_DATA_DIR = "job_data"

# 디렉터리가 없으면 생성
os.makedirs(JOB_DATA_DIR, exist_ok=True)

# 라우터 생성
router = APIRouter(
    prefix="/train",
    tags=["training"],
    responses={404: {"description": "Not found"}},
)

class MLFlowParams(BaseModel):
    mlflow_url: str = Field("https://polar-mlflow-git.x2bee.com/", description="MLFlow URL")
    mlflow_exp_id: str = Field("test", description="MLFlow Experiment ID")
    mlflow_run_id: str = Field("test", description="MLFlow Run ID")

# 훈련 파라미터 모델 정의
class TrainingParams(BaseModel):
    # Common settings
    number_gpu: int = Field(1, description="Number of GPUs to use")
    project_name: str = Field("test-project", description="Name of the current project")
    training_method: str = Field("cls", description="Training method to perform")
    model_load_method: str = Field("huggingface", description="Where to load the model from")
    dataset_load_method: str = Field("huggingface", description="Where to load the dataset from")
    hugging_face_user_id: str = Field(default_factory=get_huggingface_user_id, description="HuggingFace ID")
    hugging_face_token: str = Field(default_factory=get_huggingface_token, description="HuggingFace Token")
    mlflow_url: str = Field(default_factory=get_mlflow_url, description="MLFlow URL")
    mlflow_run_id: str = Field("test", description="MLFlow Run ID")
    minio_url: str = Field(default_factory=lambda: get_minio_config()["url"], description="MinIO URL")
    minio_access_key: str = Field(default_factory=lambda: get_minio_config()["access_key"], description="MinIO Access Key")
    minio_secret_key: str = Field(default_factory=lambda: get_minio_config()["secret_key"], description="MinIO Secret Key")

    # DeepSpeed settings
    use_deepspeed: bool = Field(False, description="Whether to use DeepSpeed")
    ds_jsonpath: Optional[str] = Field("", description="JSON path for DeepSpeed configuration")
    ds_preset: str = Field("zero-2", description="DeepSpeed preset configuration")
    ds_stage2_bucket_size: float = Field(5e8, description="DeepSpeed stage2 bucket size")
    ds_stage3_sub_group_size: float = Field(1e9, description="DeepSpeed stage3 sub group size")
    ds_stage3_max_live_parameters: float = Field(1e6, description="DeepSpeed stage3 max live parameters")
    ds_stage3_max_reuse_distance: float = Field(1e6, description="DeepSpeed stage3 max reuse distance")

    # Model settings
    model_name_or_path: str = Field(..., description="Path or name to load the model from")
    language_model_class: str = Field("none", description="Language model class for data pre-processing")
    ref_model_path: Optional[str] = Field("", description="Reference model path for DPO/PPO/GRPO trainers")
    model_subfolder: Optional[str] = Field("", description="Subfolder for Huggingface model")
    config_name: Optional[str] = Field("", description="Config name for Huggingface model")
    tokenizer_name: Optional[str] = Field("", description="Tokenizer name for Huggingface model")
    cache_dir: Optional[str] = Field("", description="Cache directory for downloaded models")

    # Data settings
    train_data: str = Field(..., description="Training data source")
    train_data_dir: Optional[str] = Field("", description="Directory for training data")
    train_data_split: str = Field("train", description="Split for training data")
    test_data: Optional[str] = Field("", description="Test data source")
    test_data_dir: Optional[str] = Field("", description="Directory for test data")
    test_data_split: str = Field("test", description="Split for test data")

    # Additional column settings with corrected spelling
    dataset_main_column: Optional[str] = Field("instruction", description="Main column in dataset (corrected spelling)")
    dataset_sub_column: Optional[str] = Field("output", description="Secondary column in dataset (corrected spelling)")
    dataset_minor_column: Optional[str] = Field("", description="Tertiary column in dataset (corrected spelling)")
    dataset_last_column: Optional[str] = Field("", description="Last column in dataset")

    # Push settings
    push_to_hub: bool = Field(True, description="Whether to push to Huggingface Hub")
    push_to_minio: bool = Field(True, description="Whether to push to MinIO")
    minio_model_load_bucket: str = Field("models", description="MinIO bucket to load models from")
    minio_model_save_bucket: str = Field("models", description="MinIO bucket to save models to")
    minio_data_load_bucket: str = Field("data", description="MinIO bucket to load data from")

    # Training specific settings
    use_sfttrainer: bool = Field(False, description="Whether to use TRL's SFTTrainer")
    use_dpotrainer: bool = Field(False, description="Whether to use TRL's DPOTrainer")
    use_ppotrainer: bool = Field(False, description="Whether to use TRL's PPOTrainer")
    use_grpotrainer: bool = Field(False, description="Whether to use TRL's GRPTrainer")
    use_custom_kl_sfttrainer: bool = Field(False, description="Whether to use custom KL-SFTTrainer")
    mlm_probability: float = Field(0.2, description="Probability for masking in MLM")
    num_labels: int = Field(17, description="Number of labels for classification")

    # DPO Setting
    dpo_loss_type: str = Field("sigmoid", description="Loss type for DPO trainer")
    dpo_beta: float = Field(0.1, description="DPO Beta. Level of preference")
    dpo_label_smoothing: float = Field(0.0, description="DPO Label smoothin value. Binary Epsilon")

    # Sentence transformer settings
    st_pooling_mode: str = Field("mean", description="Pooling mode for sentence transformer")
    st_dense_feature: int = Field(0, description="Dense feature size for sentence transformer")
    st_loss_func: str = Field("CosineSimilarityLoss", description="Loss function for sentence transformer")
    st_evaluation: Optional[str] = Field("", description="Evaluation method for sentence transformer")
    st_guide_model: str = Field("nlpai-lab/KURE-v1", description="Guide model for sentence transformer")
    st_cache_minibatch: int = Field(16, description="Cache minibatch size for sentence transformer")
    st_triplet_margin: int = Field(5, description="Triplet margin for sentence transformer")
    st_cache_gist_temperature: float = Field(0.01, description="Temperature for CachedGISTEmbedLoss in sentence transformer")
    st_use_adaptivelayerloss: bool = Field(False, description="Whether to use AdaptiveLayerLoss in sentence transformer")
    st_adaptivelayerloss_n_layer: int = Field(4, description="Number of layers for AdaptiveLayerLoss in sentence transformer")

    # Other settings
    use_attn_implementation: bool = Field(True, description="Whether to use attention implementation")
    attn_implementation: str = Field("eager", description="Attention implementation type")
    is_resume: bool = Field(False, description="Whether to resume training")
    model_commit_msg: str = Field("large-try", description="Commit message for model push")
    train_test_split_ratio: float = Field(0.05, description="Ratio for train-test split")
    data_filtering: bool = Field(True, description="Whether to filter data")
    tokenizer_max_len: int = Field(256, description="Maximum length for tokenizer")
    output_dir: Optional[str] = Field("", description="Directory to save output")
    overwrite_output_dir: bool = Field(True, description="Whether to overwrite output directory")

    # Optimizer settings
    use_stableadamw: bool = Field(True, description="Whether to use StableAdamW optimizer")
    optim: str = Field("adamw_torch", description="Optimizer to use")
    adam_beta1: float = Field(0.900, description="Beta1 for Adam optimizer")
    adam_beta2: float = Field(0.990, description="Beta2 for Adam optimizer")
    adam_epsilon: float = Field(1e-7, description="Epsilon for Adam optimizer")

    # Saving and evaluation settings
    save_strategy: str = Field("steps", description="Strategy for saving model")
    save_steps: int = Field(1000, description="Steps between saves")
    eval_strategy: str = Field("steps", description="Strategy for evaluation")
    eval_steps: int = Field(1000, description="Steps between evaluations")
    save_total_limit: int = Field(1, description="Maximum number of checkpoints to save")
    hub_model_id: Optional[str] = Field("", description="Model ID for Huggingface Hub")
    hub_strategy: str = Field("checkpoint", description="Strategy for pushing to Huggingface Hub")

    # Logging and training settings
    logging_steps: int = Field(5, description="Steps between logging")
    max_grad_norm: float = Field(1, description="Maximum gradient norm")
    per_device_train_batch_size: int = Field(4, description="Batch size per device for training")
    per_device_eval_batch_size: int = Field(4, description="Batch size per device for evaluation")
    gradient_accumulation_steps: int = Field(16, description="Steps for gradient accumulation")
    ddp_find_unused_parameters: bool = Field(True, description="Whether to find unused parameters in DDP")
    learning_rate: float = Field(2e-5, description="Learning rate")
    gradient_checkpointing: bool = Field(True, description="Whether to use Gradient Checkpointing")
    num_train_epochs: int = Field(1, description="Number of training epochs")
    warmup_ratio: float = Field(0.1, description="Ratio for warmup")
    weight_decay: float = Field(0.01, description="Weight decay")
    do_train: bool = Field(True, description="Whether to train")
    do_eval: bool = Field(True, description="Whether to evaluate")
    bf16: bool = Field(True, description="Whether to use bf16 precision")
    fp16: bool = Field(False, description="Whether to use fp16 precision")

    # PEFT settings
    use_peft: bool = Field(False, description="Whether to use PEFT")
    peft_type: str = Field("lora", description="Type of PEFT to use")

    # For LoRA
    lora_target_modules: str = Field("", description="A comma-separated STR that refers to LORA's target layer")
    lora_r: int = Field(8, description="The dimension (rank) of the low-rank decomposition matrix")
    lora_alpha: int = Field(16, description="The scaling factor, which controls the size of low-rank updates")
    lora_dropout: float = Field(0.05, description="Dropout probability applied to LoRA layers")
    lora_modules_to_save: str = Field("", description="A comma-separated STR that refers to the layers to unfreeze and train")

    # For AdaLoRA
    adalora_init_r: int = Field(12, description="Initial AdaLoRA rank")
    adalora_target_r: int = Field(4, description="Target AdaLoRA rank")
    adalora_tinit: int = Field(50, description="Number of warmup steps for AdaLoRA")
    adalora_tfinal: int = Field(100, description="Fix the resulting budget distribution steps for AdaLoRA")
    adalora_delta_t: int = Field(10, description="Interval of steps for AdaLoRA to update rank")
    adalora_orth_reg_weight: float = Field(0.5, description="Orthogonal regularization weight for AdaLoRA")

    # For IA3
    ia3_target_modules: Optional[str] = Field("", description="Target modules for the IA3 method")
    feedforward_modules: Optional[str] = Field("", description="Target feedforward modules for the IA3 method")

    # For LlamaAdapter
    adapter_layers: int = Field(30, description="Number of adapter layers (from the top) in llama-adapter")
    adapter_len: int = Field(16, description="Number of adapter tokens to insert in llama-adapter")

    # For Vera
    vera_target_modules: Optional[str] = Field("", description="Target modules for the vera method")

    # For LayerNorm Tuning
    ln_target_modules: Optional[str] = Field("", description="Target modules for the ln method")

class TrainingResponse(BaseModel):
    status: str
    message: str
    job_id: str

# 훈련 작업 상태 저장을 위한 딕셔너리
training_jobs = {}
# 동시성 제어를 위한 락
jobs_lock = threading.Lock()
# 실행 중인 작업 추적을 위한 집합
running_jobs = set()
# running_jobs 집합을 위한 락
running_lock = threading.Lock()
# 각 작업별 로그를 저장할 딕셔너리
job_logs = {}
# 로그 저장을 위한 락
logs_lock = threading.Lock()

def save_job_to_json(job_id: str, job_data: Dict[str, Any]) -> None:
    """
    작업 데이터를 JSON 파일로 저장합니다.
    """
    file_path = os.path.join(JOB_DATA_DIR, f"{job_id}.json")
    with open(file_path, 'w', encoding='utf-8') as f:
        json.dump(job_data, f, ensure_ascii=False, indent=2)
    logger.info(f"Job data saved to {file_path}")

def load_job_from_json(job_id: str) -> Dict[str, Any]:
    """
    JSON 파일에서 작업 데이터를 로드합니다.
    """
    file_path = os.path.join(JOB_DATA_DIR, f"{job_id}.json")
    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail="Job not found")

    with open(file_path, 'r', encoding='utf-8') as f:
        return json.load(f)

def load_all_jobs_from_json() -> Dict[str, Dict[str, Any]]:
    """
    모든 작업 데이터를 JSON 파일에서 로드합니다.
    """
    jobs = {}
    job_files = glob.glob(os.path.join(JOB_DATA_DIR, "*.json"))

    for file_path in job_files:
        try:
            job_id = Path(file_path).stem  # 파일명에서 확장자를 제외한 부분 추출
            with open(file_path, 'r', encoding='utf-8') as f:
                jobs[job_id] = json.load(f)
        except Exception as e:
            logger.error(f"Error loading job data from {file_path}: {str(e)}")

    return jobs

def append_job_logs(job_id: str, log_entry: Dict[str, Any]) -> None:
    """
    작업 로그를 JSON 파일에 추가합니다.
    """
    try:
        job_data = load_job_from_json(job_id)

        if "recent_logs" not in job_data:
            job_data["recent_logs"] = []

        job_data["recent_logs"].append(log_entry)

        # 최대 로그 수 제한
        if len(job_data["recent_logs"]) > 500:
            job_data["recent_logs"] = job_data["recent_logs"][-500:]

        save_job_to_json(job_id, job_data)
    except Exception as e:
        logger.error(f"Error appending log to job {job_id}: {str(e)}")

# LogCaptureHandler 수정
class LogCaptureHandler(logging.Handler):
    """
    로그를 캡처하여 작업별로 저장하는 핸들러
    """
    def __init__(self, job_id):
        super().__init__()
        self.job_id = job_id

    def emit(self, record):
        log_entry = self.format(record)
        log_data = {
            "timestamp": datetime.now().isoformat(),
            "level": record.levelname,
            "message": log_entry
        }

        # 메모리에 로그 저장 (실시간 로그용)
        with logs_lock:
            if self.job_id not in job_logs:
                job_logs[self.job_id] = []
            job_logs[self.job_id].append(log_data)
            # 최대 500개의 로그만 유지
            if len(job_logs[self.job_id]) > 500:
                job_logs[self.job_id] = job_logs[self.job_id][-500:]

        # JSON 파일에는 로그를 저장하지 않음 (작업 완료 시에만 저장)

def run_training_thread(params: Dict[str, Any], job_id: str):
    """
    훈련 프로세스를 실행하는 스레드 함수.
    별도의 스레드에서 훈련을 실행하여 메인 스레드를 차단하지 않도록 합니다.
    """
    # 이 작업을 위한 로그 핸들러 설정
    log_handler = LogCaptureHandler(job_id)
    log_handler.setFormatter(logging.Formatter('%(message)s'))
    logger.addHandler(log_handler)

    try:
        with running_lock:
            if job_id in running_jobs:
                logger.warning(f"Job {job_id} is already running. Skipping.")
                return
            running_jobs.add(job_id)

        # 작업 상세 정보 저장
        job_data = {
            "status": "running",
            "start_time": datetime.now().isoformat(),
            "model_info": {
                "name": params.get("model_name_or_path", ""),
                "type": params.get("training_method", "cls"),
                "load_method": params.get("model_load_method", "huggingface"),
            },
            "dataset_info": {
                "train_data": params.get("train_data", ""),
                "test_data": params.get("test_data", ""),
                "main_column": params.get("dataset_main_colunm", "goods_nm"),
                "sub_column": params.get("dataset_sub_colunm", "label"),
            },
            "training_config": {
                "project_name": params.get("project_name", "test-project"),
                "batch_size": params.get("per_device_train_batch_size", 4),
                "learning_rate": params.get("learning_rate", 2e-5),
                "epochs": params.get("num_train_epochs", 1),
                "gpu_count": params.get("number_gpu", 1),
                "use_deepspeed": params.get("use_deepspeed", False),
            },
            "progress": {
                "current_epoch": 0,
                "total_epochs": params.get("num_train_epochs", 1),
                "current_step": 0,
                "estimated_total_steps": 0,
                "completion_percentage": 0
            },
            "parameters": params,  # 전체 파라미터 저장
            "recent_logs": []
        }

        with jobs_lock:
            training_jobs[job_id] = job_data

        # JSON 파일로 저장
        save_job_to_json(job_id, job_data)

        logger.info(f"Starting training job {job_id}")
        logger.info(f"Parameters: {params}")

        # 직접 명령어 구성
        command = [
            "deepspeed" if params.get("use_deepspeed", False) else "torchrun",
        ]

        if params.get("use_deepspeed", False):
            command += [f"--num_gpus={params.get('number_gpu', 1)}"]
        else:
            command += [f"--nproc_per_node={params.get('number_gpu', 1)}"]

        command += [
            "/app/polar-trainer/train.py",
            f"--hugging_face_user_id={params.get('hugging_face_user_id', get_huggingface_user_id())}",
            f"--hugging_face_token={params.get('hugging_face_token', get_huggingface_token())}",
            f"--mlflow_url={params.get('mlflow_url', 'https://polar-mlflow-git.x2bee.com/')}",
            f"--mlflow_run_id={params.get('mlflow_run_id', 'test')}",
            f"--minio_url={params.get('minio_url', get_minio_config()['url'])}",
            f"--minio_access_key={params.get('minio_access_key', get_minio_config()['access_key'])}",
            f"--minio_secret_key={params.get('minio_secret_key', get_minio_config()['secret_key'])}",
            f"--model_name={params.get('project_name', 'test-project')}",
            f"--model_train_method={params.get('training_method', 'cls')}",
            f"--model_load_method={params.get('model_load_method', 'huggingface')}",
            f"--dataset_load_method={params.get('dataset_load_method', 'huggingface')}",
            f"--model_name_or_path={params.get('model_name_or_path', '')}",
            f"--language_model_class={params.get('language_model_class', 'none')}",
            f"--ref_model_path={params.get('ref_model_path', '')}",
            f"--cache_dir={params.get('cache_dir', '')}",
            f"--model_subfolder={params.get('model_subfolder', '')}",
            f"--config_name={params.get('config_name', '')}",
            f"--tokenizer_name={params.get('tokenizer_name', '')}",
            f"--train_data={params.get('train_data', '')}",
            f"--train_data_dir={params.get('train_data_dir', '')}",
            f"--train_data_split={params.get('train_data_split', 'train')}",
            f"--test_data={params.get('test_data', '')}",
            f"--test_data_dir={params.get('test_data_dir', '')}",
            f"--test_data_split={params.get('test_data_split', 'test')}",
            f"--dataset_main_column={params.get('dataset_main_column', 'instruction')}",
            f"--dataset_sub_column={params.get('dataset_sub_column', 'output')}",
            f"--dataset_minor_column={params.get('dataset_minor_column', '')}",
            f"--dataset_last_column={params.get('dataset_last_column', '')}",
            f"--push_to_hub={params.get('push_to_hub', True)}",
            f"--push_to_minio={params.get('push_to_minio', True)}",
            f"--minio_model_load_bucket={params.get('minio_model_load_bucket', 'models')}",
            f"--minio_model_save_bucket={params.get('minio_model_save_bucket', 'models')}",
            f"--minio_data_load_bucket={params.get('minio_data_load_bucket', 'data')}",
            f"--mlm_probability={params.get('mlm_probability', 0.2)}",
            f"--num_labels={params.get('num_labels', 17)}",
            f"--st_pooling_mode={params.get('st_pooling_mode', 'mean')}",
            f"--st_dense_feature={params.get('st_dense_feature', 0)}",
            f"--st_loss_func={params.get('st_loss_func', 'CosineSimilarityLoss')}",
            f"--st_evaluation={params.get('st_evaluation', '')}",
            f"--st_guide_model={params.get('st_guide_model', 'nlpai-lab/KURE-v1')}",
            f"--st_cache_minibatch={params.get('st_cache_minibatch', 16)}",
            f"--st_triplet_margin={params.get('st_triplet_margin', 5)}",
            f"--use_sfttrainer={params.get('use_sfttrainer', False)}",
            f"--use_dpotrainer={params.get('use_dpotrainer', False)}",
            f"--use_ppotrainer={params.get('use_ppotrainer', False)}",
            f"--use_grpotrainer={params.get('use_grpotrainer', False)}",
            f"--use_custom_kl_sfttrainer={params.get('use_custom_kl_sfttrainer', False)}",
            f"--use_deepspeed={params.get('use_deepspeed', False)}",
            f"--ds_jsonpath={params.get('ds_jsonpath', '')}",
            f"--ds_preset={params.get('ds_preset', 'zero-2')}",
            f"--ds_stage2_bucket_size={params.get('ds_stage2_bucket_size', 5e8)}",
            f"--ds_stage3_sub_group_size={params.get('ds_stage3_sub_group_size', 1e9)}",
            f"--ds_stage3_max_live_parameters={params.get('ds_stage3_max_live_parameters', 1e6)}",
            f"--ds_stage3_max_reuse_distance={params.get('ds_stage3_max_reuse_distance', 1e6)}",
            f"--use_attn_implementation={params.get('use_attn_implementation', True)}",
            f"--attn_implementation={params.get('attn_implementation', 'eager')}",
            f"--is_resume={params.get('is_resume', False)}",
            f"--model_commit_msg={params.get('model_commit_msg', 'large-try')}",
            f"--train_test_split_ratio={params.get('train_test_split_ratio', 0.05)}",
            f"--data_filtering={params.get('data_filtering', True)}",
            f"--tokenizer_max_len={params.get('tokenizer_max_len', 256)}",
            f"--output_dir={params.get('output_dir', '')}",
            f"--overwrite_output_dir={params.get('overwrite_output_dir', True)}",  # 기본값을 True로 변경
            f"--use_stableadamw={params.get('use_stableadamw', True)}",
            f"--optim={params.get('optim', 'adamw_torch')}",
            f"--adam_beta1={params.get('adam_beta1', 0.9)}",
            f"--adam_beta2={params.get('adam_beta2', 0.99)}",
            f"--adam_epsilon={params.get('adam_epsilon', 1e-7)}",
            f"--save_strategy={params.get('save_strategy', 'steps')}",
            f"--save_steps={params.get('save_steps', 1000)}",
            f"--eval_strategy={params.get('eval_strategy', 'steps')}",
            f"--eval_steps={params.get('eval_steps', 1000)}",
            f"--save_total_limit={params.get('save_total_limit', 1)}",
            f"--hub_model_id={params.get('hub_model_id', '')}",
            f"--hub_strategy={params.get('hub_strategy', 'checkpoint')}",
            f"--logging_steps={params.get('logging_steps', 5)}",
            f"--max_grad_norm={params.get('max_grad_norm', 1)}",
            f"--per_device_train_batch_size={params.get('per_device_train_batch_size', 4)}",
            f"--per_device_eval_batch_size={params.get('per_device_eval_batch_size', 4)}",
            f"--gradient_accumulation_steps={params.get('gradient_accumulation_steps', 16)}",
            f"--ddp_find_unused_parameters={params.get('ddp_find_unused_parameters', True)}",
            f"--learning_rate={params.get('learning_rate', 2e-5)}",
            f"--gradient_checkpointing={params.get('gradient_checkpointing', True)}",
            f"--num_train_epochs={params.get('num_train_epochs', 1)}",
            f"--warmup_ratio={params.get('warmup_ratio', 0.1)}",
            f"--weight_decay={params.get('weight_decay', 0.01)}",
            f"--do_train={params.get('do_train', True)}",
            f"--do_eval={params.get('do_eval', True)}",
            f"--bf16={params.get('bf16', True)}",
            f"--fp16={params.get('fp16', False)}",
            f"--use_peft={params.get('use_peft', False)}",
            f"--peft_type={params.get('peft_type', 'lora')}",
            f"--lora_target_modules={params.get('lora_target_modules', 'q_proj,k_proj')}",
            f"--lora_r={params.get('lora_r', 16)}",
            f"--lora_alpha={params.get('lora_alpha', 32)}",
            f"--lora_dropout={params.get('lora_dropout', 0.05)}",
            f"--lora_modules_to_save={params.get('lora_modules_to_save', '')}",
            f"--adalora_init_r={params.get('adalora_init_r', 16)}",
            f"--adalora_target_r={params.get('adalora_target_r', 16)}",
            f"--adalora_tinit={params.get('adalora_tinit', 0)}",
            f"--adalora_tfinal={params.get('adalora_tfinal', 1000)}",
            f"--adalora_delta_t={params.get('adalora_delta_t', 100)}",
            f"--adalora_orth_reg_weight={params.get('adalora_orth_reg_weight', 0.01)}",
            f"--ia3_target_modules={params.get('ia3_target_modules', 'q_proj,k_proj')}",
            f"--feedforward_modules={params.get('feedforward_modules', 'ffn')}",
            f"--adapter_layers={params.get('adapter_layers', '0,1,2,3')}",
            f"--adapter_len={params.get('adapter_len', 16)}",
            f"--vera_target_modules={params.get('vera_target_modules', 'q_proj,k_proj')}",
            f"--ln_target_modules={params.get('ln_target_modules', 'layer_norm')}",
            f"--dpo_loss_type={params.get('dpo_loss_type', 'sigmoid')}",
            f"--dpo_beta={params.get('dpo_beta', 0.1)}",
            f"--dpo_label_smoothing={params.get('dpo_label_smoothing', 0.0)}",
            f"--st_cache_gist_temperature={params.get('st_cache_gist_temperature', 0.01)}",
            f"--st_use_adaptivelayerloss={params.get('st_use_adaptivelayerloss', False)}",
            f"--st_adaptivelayerloss_n_layer={params.get('st_adaptivelayerloss_n_layer', 4)}"
        ]

        # 로그에 명령어 출력
        logger.info(f"Executing command: {' '.join(command)}")

        # 명령어 실행 (stdout과 stderr를 캡처)
        process = subprocess.Popen(
            command,
            shell=False,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1,  # 라인 버퍼링
            universal_newlines=True
        )

        with jobs_lock:
            training_jobs[job_id]["process_id"] = process.pid
            # Save updated job data to JSON file
            save_job_to_json(job_id, training_jobs[job_id])

        logger.info(f"Process started with PID: {process.pid}")

        # 비동기적으로 stdout 읽기
        def read_output(pipe, log_level):
            for line in iter(pipe.readline, ''):
                if not line:
                    break

                # 실제 내용만 추출
                stripped_line = line.strip()

                # 빈 줄은 INFO로 처리
                if not stripped_line:
                    logger.info(" ")
                    continue

                # stderr에서 오는 출력 중 실제 에러가 아닌 정보성 메시지를 식별하기 위한 패턴들
                info_patterns = [
                    # 진행률 표시줄
                    lambda x: "%" in x and "|" in x and "[" not in x[:5],
                    # PyTorch 메시지
                    lambda x: any(term in x for term in ["Root Cause", "exitcode", "traceback", "find_unused_parameters", "DeepSpeed"]),
                    # 업로드 메시지
                    lambda x: "Upload" in x,
                    # 파라미터 출력 (최적화기 설정 등)
                    lambda x: any(param in x for param in ["amsgrad:", "betas:", "lr:", "eps:", "weight_decay:", "Parameter Group"]),
                    # 학습 진행 정보
                    lambda x: x.startswith("{") and any(metric in x for metric in ["'loss':", "'grad_norm':", "'learning_rate':", "'epoch':"]),
                    # 괄호나 중괄호로 끝나는 라인 (파라미터 설정의 끝)
                    lambda x: x.strip() in [")", "}", ")"]
                ]

                # 패턴 확인
                is_info = any(pattern(stripped_line) for pattern in info_patterns)

                # stderr에서 왔지만 실제로는 정보 메시지인 경우
                if log_level == "ERROR" and is_info:
                    logger.info(stripped_line)
                else:
                    # 원래 로그 레벨대로 출력
                    if log_level == "INFO":
                        logger.info(stripped_line)
                    else:
                        logger.error(stripped_line)

                # 진행 상황 업데이트
                if "epoch" in stripped_line.lower() and "step" in stripped_line.lower():
                    try:
                        # 로그에서 에포크와 스텝 정보 추출 시도
                        parts = stripped_line.lower().split()
                        update_needed = False

                        for i, part in enumerate(parts):
                            if part.startswith("epoch"):
                                epoch_info = parts[i+1].split("/")
                                if len(epoch_info) == 2:
                                    current_epoch = float(epoch_info[0])
                                    total_epochs = float(epoch_info[1])
                                    with jobs_lock:
                                        training_jobs[job_id]["progress"]["current_epoch"] = current_epoch
                                        training_jobs[job_id]["progress"]["total_epochs"] = total_epochs
                                        update_needed = True
                            if part.startswith("step"):
                                step_info = parts[i+1].split("/")
                                if len(step_info) == 2:
                                    current_step = int(step_info[0])
                                    total_steps = int(step_info[1])
                                    with jobs_lock:
                                        training_jobs[job_id]["progress"]["current_step"] = current_step
                                        training_jobs[job_id]["progress"]["estimated_total_steps"] = total_steps
                                        training_jobs[job_id]["progress"]["completion_percentage"] = round(
                                            (current_epoch - 1 + current_step/total_steps) / total_epochs * 100, 2
                                        )
                                        update_needed = True

                        # If progress was updated, save to JSON file
                        if update_needed:
                            with jobs_lock:
                                save_job_to_json(job_id, training_jobs[job_id])

                    except Exception as e:
                        logger.warning(f"Error parsing progress info: {str(e)}")

        # 출력 읽기 스레드 시작
        stdout_thread = threading.Thread(target=read_output, args=(process.stdout, "INFO"), daemon=True)
        stderr_thread = threading.Thread(target=read_output, args=(process.stderr, "ERROR"), daemon=True)
        stdout_thread.start()
        stderr_thread.start()

        # 프로세스 완료 대기
        return_code = process.wait()

        # 출력 읽기 스레드가 종료될 때까지 대기
        stdout_thread.join()
        stderr_thread.join()

        logger.info(f"Process completed with return code: {return_code}")

        with jobs_lock:
            if return_code == 0:
                training_jobs[job_id]["status"] = "completed"
                training_jobs[job_id]["end_time"] = datetime.now().isoformat()
                training_jobs[job_id]["progress"]["completion_percentage"] = 100

                # 작업 완료 시 로그를 JSON 파일에 포함하여 저장
                with logs_lock:
                    if job_id in job_logs:
                        training_jobs[job_id]["recent_logs"] = job_logs[job_id]

                # Save updated job status to JSON file
                save_job_to_json(job_id, training_jobs[job_id])
                logger.info(f"Training job {job_id} completed successfully")
            else:
                training_jobs[job_id]["status"] = "failed"
                training_jobs[job_id]["error"] = f"Process exited with code {return_code}"
                training_jobs[job_id]["end_time"] = datetime.now().isoformat()

                # 작업 실패 시에도 로그를 JSON 파일에 포함하여 저장
                with logs_lock:
                    if job_id in job_logs:
                        training_jobs[job_id]["recent_logs"] = job_logs[job_id]

                # Save updated job status to JSON file
                save_job_to_json(job_id, training_jobs[job_id])
                logger.error(f"Training job {job_id} failed with return code {return_code}")

    except Exception as e:
        with jobs_lock:
            training_jobs[job_id]["status"] = "failed"
            training_jobs[job_id]["error"] = str(e)
            training_jobs[job_id]["end_time"] = datetime.now().isoformat()

            # 예외 발생 시에도 로그를 JSON 파일에 포함하여 저장
            with logs_lock:
                if job_id in job_logs:
                    training_jobs[job_id]["recent_logs"] = job_logs[job_id]

            # Save updated job status to JSON file
            save_job_to_json(job_id, training_jobs[job_id])
        logger.error(f"Training job {job_id} failed: {str(e)}")

    finally:
        # 로그 핸들러 제거
        logger.removeHandler(log_handler)
        # Always remove the job from running_jobs, even if an exception occurs
        with running_lock:
            if job_id in running_jobs:
                running_jobs.remove(job_id)
                logger.info(f"Removed job {job_id} from running jobs")

def run_training(params: Dict[str, Any], job_id: str):
    """
    훈련 프로세스를 실행하는 새 스레드를 시작합니다.
    이 함수는 FastAPI 백그라운드 태스크에 의해 호출됩니다.
    """
    # 훈련 프로세스를 위한 새 스레드 생성
    thread = threading.Thread(
        target=run_training_thread,
        args=(params, job_id),
        daemon=True  # 애플리케이션 종료를 차단하지 않도록 데몬 스레드로 설정
    )
    thread.start()
    logger.info(f"Started training thread for job {job_id}")

@router.post("", response_model=TrainingResponse)
async def start_training(params: TrainingParams, background_tasks: BackgroundTasks):
    """
    새 훈련 작업을 시작합니다.
    """
    current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
    job_id = f"job_{current_time}"

    # Pydantic 모델을 딕셔너리로 변환
    params_dict = params.model_dump()
    params_dict["mlflow_run_id"] = job_id
    # 백그라운드에서 훈련 시작
    background_tasks.add_task(run_training, params_dict, job_id)

    return {
        "status": "accepted",
        "message": "Training job started",
        "job_id": job_id
    }

@router.post("/mlflow", response_model=Dict[str, Any])
async def get_mlflow(params: MLFlowParams):
    try:
        experiment_id, run_id = get_mlflow_info(params.mlflow_url, params.mlflow_exp_id, params.mlflow_run_id)

        return {
            "status": "success",
            "experiment_id": experiment_id,
            "run_id": run_id,
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/{job_id}", response_model=Dict[str, Any])
async def get_job_status(job_id: str):
    """
    특정 훈련 작업의 상태를 조회합니다.
    """
    try:
        # JSON 파일에서 작업 데이터 로드
        job_info = load_job_from_json(job_id)

        # 아직 실행 중인 경우, 일부 정보를 메모리에서 업데이트
        running_job = False
        with jobs_lock:
            if job_id in training_jobs and training_jobs[job_id]["status"] == "running":
                running_job = True
                # 메모리에 있는 최신 정보로 업데이트
                for key in ["status", "progress"]:
                    if key in training_jobs[job_id]:
                        job_info[key] = training_jobs[job_id][key]

        # 실행 중인 작업의 경우 메모리에서 최신 로그를 가져옴
        if running_job:
            with logs_lock:
                if job_id in job_logs:
                    job_info["recent_logs"] = job_logs[job_id][-50:]  # 최근 50개 로그만 반환
                else:
                    job_info["recent_logs"] = []
        # 이미 완료된 작업의 경우 JSON 파일에 저장된 로그를 사용하되 최근 50개만 반환
        elif "recent_logs" in job_info:
            job_info["recent_logs"] = job_info["recent_logs"][-50:]
        else:
            job_info["recent_logs"] = []

        # 실행 시간 계산
        if "start_time" in job_info:
            start_time = datetime.fromisoformat(job_info["start_time"])
            if "end_time" in job_info and job_info["end_time"]:
                end_time = datetime.fromisoformat(job_info["end_time"])
                duration = (end_time - start_time).total_seconds()
            else:
                duration = (datetime.now() - start_time).total_seconds()

            # 실행 시간을 시, 분, 초 형식으로 변환
            hours, remainder = divmod(duration, 3600)
            minutes, seconds = divmod(remainder, 60)
            job_info["duration"] = {
                "hours": int(hours),
                "minutes": int(minutes),
                "seconds": int(seconds),
                "total_seconds": duration
            }

        # 작업 요약 추가
        job_info["summary"] = {
            "status": job_info.get("status", "unknown"),
            "model": job_info.get("model_info", {}).get("name", "Unknown"),
            "dataset": job_info.get("dataset_info", {}).get("train_data", "Unknown"),
            "progress": job_info.get("progress", {}).get("completion_percentage", 0)
        }

        return job_info
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting job status for {job_id}: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error getting job status: {str(e)}")

@router.get("", response_model=Dict[str, Dict[str, Any]])
async def get_all_jobs():
    """
    모든 훈련 작업의 상태를 조회합니다.
    """
    try:
        # JSON 파일에서 모든 작업 데이터 로드
        jobs = load_all_jobs_from_json()

        # 실행 중인 작업의 경우 메모리에서 최신 상태 업데이트
        with jobs_lock:
            for job_id, job_data in training_jobs.items():
                if job_id in jobs and job_data["status"] == "running":
                    # 주요 정보만 업데이트
                    jobs[job_id]["status"] = job_data["status"]
                    jobs[job_id]["progress"] = job_data["progress"]

        return jobs
    except Exception as e:
        logger.error(f"Error getting all jobs: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error getting all jobs: {str(e)}")

@router.delete("/{job_id}", response_model=Dict[str, Any])
async def cancel_job(job_id: str):
    """
    실행 중인 훈련 작업을 취소합니다.
    """
    try:
        # JSON 파일에서 작업 데이터 로드
        job = load_job_from_json(job_id)

        if job["status"] != "running":
            return {
                "status": "error",
                "message": f"Job is not running. Current status: {job['status']}"
            }

        # 메모리에 있는 정보 확인
        process_id = None
        with jobs_lock:
            if job_id in training_jobs and "process_id" in training_jobs[job_id]:
                process_id = training_jobs[job_id]["process_id"]

        if not process_id:
            # JSON 파일에서 process_id 확인
            process_id = job.get("process_id")

        if process_id:
            try:
                os.kill(process_id, signal.SIGTERM)
                logger.info(f"Sent SIGTERM to process {process_id}")

                # 프로세스가 정상적으로 종료될 시간을 줌
                import time
                time.sleep(2)

                # 프로세스가 여전히 실행 중인지 확인
                try:
                    os.kill(process_id, 0)  # Signal 0은 프로세스가 존재하는지 확인하는 데 사용됨
                    # 여기에 도달하면 프로세스가 여전히 실행 중이므로 SIGKILL 시도
                    logger.warning(f"Process {process_id} did not terminate with SIGTERM, sending SIGKILL")
                    os.kill(process_id, signal.SIGKILL)
                except OSError:
                    # 프로세스가 더 이상 실행되지 않음
                    pass

                # 상태 업데이트
                job["status"] = "cancelled"
                job["end_time"] = datetime.now().isoformat()

                # 메모리 및 JSON 파일 업데이트
                with jobs_lock:
                    if job_id in training_jobs:
                        training_jobs[job_id]["status"] = "cancelled"
                        training_jobs[job_id]["end_time"] = job["end_time"]

                save_job_to_json(job_id, job)

                logger.info(f"Training job {job_id} cancelled")

                return {
                    "status": "success",
                    "message": f"Job {job_id} cancelled successfully"
                }
            except ProcessLookupError:
                logger.warning(f"Process {process_id} not found, it may have already terminated")
                job["status"] = "cancelled"
                job["end_time"] = datetime.now().isoformat()
                save_job_to_json(job_id, job)
                return {
                    "status": "success",
                    "message": f"Job {job_id} process already terminated, marked as cancelled"
                }
        else:
            return {
                "status": "error",
                "message": "Process ID not found for this job"
            }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error cancelling job {job_id}: {str(e)}")
        return {
            "status": "error",
            "message": f"Error cancelling job: {str(e)}"
        }
