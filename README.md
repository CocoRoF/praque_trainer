# 🚀 POLAR Trainer 실행 가이드

본 프로젝트는 다양한 학습 방식(Classification, Masked LM, Causal LM, DPO 등)을 지원하는 **멀티 트레이닝 프레임워크**입니다.  모든 학습은 하나의 통일된 파라미터 스크립트로 실행되며, Trainer는 내부 로직에 따라 필요한 인자만 사용합니다.

---

## 📌 학습 방식 (Training Method)

| 학습 방식 | 설명 |
|-----------|------|
| `trainer_ce` | Cross Encoder 학습용. Sequence Classification Task로 구현. (테스트 및 수정 필요) |
| `trainer_clm`       | Gemma 등 Causal Language Modeling을 학습하는 용도 |
| `trainer_cls`       | Classification 학습 용도.  |
| `trainer_mlm`            | ModernBERT와 같은 Masked LM 학습 용도 |
| `trainer_mmlm`  | 이미지 + 텍스트 기반의 멀티모달 언어모델 (Ex: Gemma3) |
| `trainer_st`            | Sentence-Transformer 학습용도 (Loss 세분화 필요) |

---

## ⚙️ 공통 실행 방식

핵심 실행 파일은 아래와 같음.
해당 파일에 존재하는 요소들이 API controller에서 사용되어야 함.

```bash
python train_api.py
```

---

## ⚙️ 공통 Argument 구조

모든 학습 방식은 동일한 Argument 스크립트를 사용하며, Trainer 내부에서 필요한 인자만 사용합니다.

```python
# 예시. gemma 트레이닝 Method는 MMLM Trainer를 실행시키는 인자임.
# 관련된 설정은 .env 파일에서 수정.

project_name = "polar-gemma-ece-27b"
training_method = "gemma"
use_sfttrainer = True
use_dpotrainer = False
use_ppotrainer = False
use_grpotrainer = False
use_deepspeed = True
model_name_or_path = "google/gemma-3-27b-it"
train_data = "CocoRoF/e-commerce_polar_dataset"
...
```

**모든 인자는 명시적으로 Trainer 내부로 넘기도록 하며**, 이후 개별 Trainer에서 필요한 인자를 사용합니다.

---

# 🚀 POLAR Trainer 실행 가이드

본 프로젝트는 다양한 학습 방식(Classification, Masked LM, Causal LM, DPO 등)을 지원하는 **멀티 트레이닝 프레임워크**입니다.  모든 학습은 하나의 통일된 파라미터 스크립트로 실행되며, Trainer는 내부 로직에 따라 필요한 인자만 사용합니다.

---

## 📌 학습 방식 (Training Method)

| 학습 방식 | 설명 |
|-----------|------|
| `classification` | 텍스트 분류 모델 학습용 |
| `maskedlm`       | BERT 기반 Masked Language Modeling |
| `causallm`       | GPT, Gemma 등 Causal Language Modeling |
| `multimodal-lm`  | 이미지 + 텍스트 기반의 멀티모달 언어모델 |
| `dpo`            | Direct Preference Optimization (RLHF 기반) |
| `sft`            | Supervised Fine-tuning |

---

## ⚙️ Argument 설명표 (전체)

| 파라미터 | 설명 |
|----------|------|
| `project_name` | 프로젝트의 이름. 이것을 기준으로 Huggingface 및 MLFlow, Minio등의 이름이 설정됨. |
| `hugging_face_user_id` | HuggingFace 사용자 ID |
| `hugging_face_token` | HuggingFace 토큰 (모델 push 시 필요) |
| `mlflow_url` | MLflow 서버 주소 |
| `minio_url` | MinIO 서버 주소 |
| `minio_access_key` / `minio_secret_key` | MinIO 접속용 인증 정보 |
| `number_gpu` | 사용할 GPU 수 |
| `training_method` | 학습 방식 지정 (`cls`, `mlm`, `clm`, `image-text`, `cross-encoder`, `sentence-transformer` 등) |
| `model_load_method` / `dataset_load_method` | 모델/데이터 로드 방식 (`huggingface` or `minio`) |
| `use_sfttrainer` / `use_dpotrainer` / `use_ppotrainer` / `use_grpotrainer` | 사용할 TRL Trainer 선택 플래그 (한 가지만 `True` 혹은 모두 `False`로 설정, 모두 `False`인 경우 기본 Trainer 사용)|
| `use_deepspeed` | DeepSpeed 사용 여부 |
| `ds_preset` | 사용할 DeepSpeed preset (`zero-1`, `zero-2`, `zero-3`) |
| `ds_jsonpath` | 직접 제공하는 deepspeed config json 경로 (직접 제공하는 경우 파일을 만들어서 넣어두어야 함) |
| `ds_stage2_bucket_size` / `ds_stage3_sub_group_size` 등 | ZeRO-2/3 최적화 파라미터 |
| `model_name_or_path` | HuggingFace의 모델 경로 또는 로컬 경로 (모델 다운로드 경로) |
| `model_subfolder` | 모델 서브폴더 지정 시 사용(모델 다운로드시 서브폴더가 있는 경우) |
| `config_name` | 커스텀 config 로딩용 (모델 다운로드시 config를 통해 로드하는 경우) |
| `tokenizer_name` | Tokenizer를 모델에 존재하는 것과 다른 것을 로드하는 경우. 빈 문자열 기입시 model의 경로 사용. |
| `train_data`, `train_data_dir`, `train_data_split` | 학습 데이터 경로 및 Split 설정. train_data_dir가 빈 문자열이면 기본값 로드. |
| `test_data`, `test_data_dir`, `test_data_split` | 평가용 데이터 설정. Train 데이터 설정과 마찬가지로 작동. |
| `dataset_main_column`, `dataset_sub_column`, `dataset_minor_column` | 사용할 주요 컬럼 선택. 총 3개까지 컬럼을 필터링. |
| `push_to_hub`, `push_to_minio` | 모델 훈련 결과 Push 여부 |
| `minio_model_load_bucket`, `minio_model_save_bucket`, `minio_data_load_bucket` | MinIO 버킷 설정. |
| `mlm_probability` | MLM 학습 시 마스킹 확률 |
| `num_labels` | Classification 문제에서 라벨 개수 |
| `st_pooling_mode`, `st_dense_feature`, `st_loss_func` | Sentence-Transformer 학습 구성용 (추후 고도화 필요) |
| `st_evaluation`, `st_guide_model` | ST 평가 및 Guide 모델 지정 (추후 고도화 필요) |
| `st_cache_minibatch`, `st_triplet_margin` | ST Loss 구성 관련 파라미터 (추후 고도화 필요) |
| `flash_attn` | Flash Attention 2.0 사용 여부 (현재 Gemma 학습을 위해 Eager Attention을 사용, 추후 세분화 Method 구현 필요) |
| `is_resume` | 학습 재개 여부 (checkpoint 기반) |
| `model_commit_msg` | 모델 push 시 커밋 메시지 |
| `train_test_split_ratio` | Train 데이터 분할 비율 (평가용) |
| `data_filtering` | 결측치 자동 필터링 여부 |
| `tokenizer_max_len` | 토크나이저 max length |
| `output_dir` | 결과 저장 경로 (미지정 시 자동 설정) |
| `overwrite_output_dir` | 기존 결과 덮어쓰기 여부 |
| `use_stableadamw` | StableAdamW Optimizer 사용 여부 (이를 기본 지원하기 시작하면 삭제 예정) |
| `optim` | Optimizer 종류 (`adamw_torch` 등 use_stableadamw를 사용시 비활성화됨) |
| `adam_beta1`, `adam_beta2`, `adam_epsilon` | Adam Optimizer 하이퍼파라미터 |
| `save_strategy`, `save_steps`, `save_total_limit` | 모델 저장 조건 및 주기 설정 |
| `eval_strategy`, `eval_steps` | 평가 전략 및 주기 |
| `hub_model_id`, `hub_strategy` | 모델 Hub push 시 repo/방식 지정 |
| `logging_steps` | 로그 출력 주기 |
| `max_grad_norm` | gradient clipping 최대 norm |
| `per_device_train_batch_size`, `per_device_eval_batch_size` | GPU당 학습/평가 배치 사이즈 |
| `gradient_accumulation_steps` | Gradient 누적 스텝 수 (실제 배치 증가 효과) |
| `learning_rate`, `warmup_ratio`, `weight_decay` | 학습률, 워밍업, 가중치 감소 설정 |
| `gradient_checkpointing` | 메모리 최적화를 위한 gradient checkpointing 여부 |
| `num_train_epochs` | 학습할 에폭 수 |
| `do_train`, `do_eval` | 학습/평가 실행 여부 |
| `bf16` | BF16 mixed precision 학습 여부 |
| `use_peft` | PEFT 적용 여부 |
| `peft_type` | PEFT 유형 (lora, ia3 등) |
| `lora_target_modules` | LoRA가 적용될 대상 모듈들 (e.g., q_proj,k_proj...) ( ','로 구분한 STR 형태로 입력해야 함) |
| `lora_r`, `lora_alpha`, `lora_dropout` | LoRA 관련 설정 값 |
| `lora_modules_to_save` | LoRA 이외에 같이 저장할 layer (lm_head 등) ( ','로 구분한 STR 형태로 입력해야 함) |
| `dpo_beta`, `dpo_label_smoothing` | DPO Trainer 전용 파라미터 |

---

## 🔄 Hugging Face Repository Pull & Push

이 프로젝트에는 Hugging Face Hub에서 모델이나 데이터셋을 다운로드하여 다른 repository로 복사하는 기능이 포함되어 있습니다.

### 기본 사용법

```python
from hugging_face_pull_and_push import pull_and_push_repo

# 모델 복사
url = pull_and_push_repo(
    source_repo_id="microsoft/DialoGPT-medium",    # 소스 모델
    target_repo_id="your-username/my-dialogpt",    # 대상 repository
    token="hf_your_token_here",                    # Hugging Face 토큰
    private=True,                                  # private repo로 생성
    commit_message="Mirror DialoGPT model"
)

# 데이터셋 복사
url = pull_and_push_repo(
    source_repo_id="squad",
    target_repo_id="your-username/my-squad",
    source_repo_type="dataset",
    target_repo_type="dataset",
    token="hf_your_token_here"
)
```

### 커맨드라인 사용법

```bash
python hugging_face_pull_and_push.py \
    --source microsoft/DialoGPT-medium \
    --target your-username/my-model \
    --token hf_your_token_here \
    --private \
    --message "Mirror model"
```

### 주요 기능

- **전체 Repository 복사**: 모든 파일(모델 가중치, 설정 파일, 토크나이저 등)을 완전히 복사
- **타입별 지원**: 모델, 데이터셋, Space 모두 지원
- **선택적 파일 무시**: 특정 패턴의 파일들을 제외하고 복사 가능
- **프라이빗 Repository**: 대상을 private repository로 설정 가능
- **커스텀 커밋 메시지**: 의미있는 커밋 메시지 추가
- **안전한 임시 파일 관리**: 다운로드된 파일들의 자동 정리

자세한 사용 예시는 `example_usage.py` 파일을 참고하세요.

---
