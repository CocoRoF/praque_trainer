from dataclasses import dataclass, field
from typing import Optional, List


@dataclass
class BaseArguments:
    use_attn_implementation: bool = field(
        default=True,
        metadata={
            "help": "다른 Attention Function을 사용할지 여부를 결정합니다."
        }
    )
    attn_implementation: str = field(
        default="flash_attention_2",
        metadata={
            "help": "attn_implementation의 종류를 결정합니다. Ex) 'flash_attention_2', 'eager' 등"
        }
    )
    use_stableadamw: bool = field(
        default=False,
        metadata={
            "help": "Stable AdamW 옵티마이저를 사용할지 여부를 지정합니다. True로 설정하면 'optim' 인자는 무시되고 Stable AdamW가 사용됩니다."
        }
    )
    use_sfttrainer: bool = field(
        default=False,
        metadata={
            "help": "TRL의 SFTTrainer를 사용할지의 여부를 지정합니다. False인 경우, HuggingFace의 Trainer를 사용합니다."
        }
    )
    use_dpotrainer: bool = field(
        default=False,
        metadata={
            "help": "TRL의 DPOTrainer를 사용할지의 여부를 지정합니다. False인 경우, HuggingFace의 Trainer를 사용합니다."
        }
    )
    use_ppotrainer: bool = field(
        default=False,
        metadata={
            "help": "TRL의 PPOTrainer를 사용할지의 여부를 지정합니다. False인 경우, HuggingFace의 Trainer를 사용합니다."
        }
    )
    use_grpotrainer: bool = field(
        default=False,
        metadata={
            "help": "TRL의 GRPOTrainer를 사용할지의 여부를 지정합니다. False인 경우, HuggingFace의 Trainer를 사용합니다."
        }
    )
    use_custom_kl_sfttrainer: bool = field(
        default=False,
        metadata={
            "help": "TRL의 SFTTrainer를 커스텀한 KL-SFTTrainer를 사용할지의 여부를 지정합니다. False인 경우, HuggingFace의 Trainer를 사용합니다."
        }
    )

@dataclass
class ModelArguments:
    """
    미세 조정(fine-tuning)에 사용할 모델, 구성(config) 및 토크나이저 관련 인자들을 정의합니다.
    """
    model_name: str = field(
        metadata={
            "help": "프로젝트에서 사용할 모델의 이름을 지정합니다. 이 이름은 학습 및 저장 시 식별자로 사용됩니다."
        }
    )
    model_train_method: str = field(
        metadata={
            "help": "언어 모델을 학습할 방식을 선택합니다. 사용 가능한 옵션은 ['mlm'(Masked Language Modeling), 'clm'(Causal Language Modeling), 'sts'(Sentence-Transformer), 'nli'(Natural Language Inference), 'cls'(Classification), 'instruction'(Instruction-based)] 입니다."
        }
    )
    model_name_or_path: str = field(
        metadata={
            "help": "사전 학습된 모델의 경로 또는 huggingface.co/models에 등록된 모델 식별자를 입력합니다."
        }
    )
    language_model_class: str = field(
        default='none',
        metadata={
            "help": "Data Processor를 위한 Language Model Class를 지정합니다. 사용 가능한 옵션은 ['none', 'gemma3']입니다."
        }
    )
    ref_model_path: str = field(
        default=None,
        metadata={
            "help": "reference model의 경로를 지정합니다. DPO, PPO, GRPOTrainer에서 사용됩니다."
        }
    )
    model_load_method: str = field(
        default='huggingface',
        metadata={
            "help": "모델을 불러올 때 사용할 방식을 지정합니다. 사용 가능한 옵션은 ['local'(로컬 경로), 'huggingface'(허깅페이스 허브), 'minio'(MinIO 저장소)]이며, 기본값은 'huggingface'입니다."
        }
    )
    model_subfolder: Optional[str] = field(
        default=None,
        metadata={
            "help": "Huggingface 모델이 여러 폴더로 구성된 경우, 실제 모델 파일이 위치한 하위 폴더 경로를 지정합니다."
        }
    )
    push_to_minio: Optional[bool] = field(
        default=False,
        metadata={
            "help": "학습 완료 후 모델을 MinIO 저장소에 업로드할지 여부를 지정합니다. True이면 업로드를 시도합니다."
        }
    )
    minio_model_load_bucket: Optional[str] = field(
        default='models',
        metadata={
            "help": "MinIO에서 모델을 불러올 때 사용할 버킷(bucket) 이름을 지정합니다. 기본값은 'models'입니다."
        }
    )
    minio_model_save_bucket: Optional[str] = field(
        default='models',
        metadata={
            "help": "MinIO에 모델을 저장할 때 사용할 버킷(bucket) 이름을 지정합니다. 기본값은 'models'입니다."
        }
    )
    config_name: Optional[str] = field(
        default=None,
        metadata={
            "help": "모델 구성 파일의 이름 또는 경로를 지정합니다. 모델 이름과 다를 경우 별도로 지정하여 사용합니다."
        }
    )
    tokenizer_name: Optional[str] = field(
        default=None,
        metadata={
            "help": "사전 학습된 토크나이저의 이름 또는 경로를 지정합니다. 모델 이름과 다를 경우 별도로 지정할 수 있습니다."
        }
    )
    selected_parameter: Optional[str] = field(
        default="",
        metadata={
            "help": "이는 모델의 Parameter의 Name을 지정해야 합니다. 입력된 문자열에서 공백과 '\n'은 모두 제거되고 ,을 기준으로 list로 나눕니다."
        }
    )
    cache_dir: Optional[str] = field(
        default=None,
        metadata={
            "help": "사전 학습된 모델을 저장할 로컬 캐시 디렉토리 경로를 지정합니다. S3에서 다운로드 받은 모델 파일을 저장하는 데 사용됩니다."
        }
    )
    is_resume: bool = field(
        default=False,
        metadata={
            "help": "이전 학습의 체크포인트가 존재할 경우, 해당 체크포인트에서 이어서 학습할지 여부를 지정합니다. True이면 재개합니다."
        }
    )
    model_commit_msg: Optional[str] = field(
        default=None,
        metadata={
            "help": "허깅페이스(Hugging Face) 허브에 모델을 업로드할 때 사용할 커밋(commit) 메시지를 지정합니다."
        }
    )


@dataclass
class DataArguments:
    hugging_face_user_id: str = field(
        default=None,
        metadata={
            "help": "허깅페이스 유저 아이디, 만약 입력되지 않으면 Huggingface관련 기능이 작동하지 않습니다."
        }
    )
    hugging_face_token: str = field(
        default=None,
        metadata={
            "help": "허깅페이스 토큰, 만약 입력되지 않으면 Huggingface관련 기능이 작동하지 않습니다."
        }
    )
    mlflow_url: str = field(
        default=None,
        metadata={
            "help": "MLFlow의 URL. 만약 입력되지 않으면 MLFlow에 로깅이 제대로 처리되지 않습니다."
        }
    )
    mlflow_run_id: str = field(
        default=None,
        metadata={
            "help": "MLFlow의 Run ID. 만약 입력되지 않으면 MLFlow에 로깅이 제대로 처리되지 않습니다."
        }
    )
    minio_url: str = field(
        default=None,
        metadata={
            "help": "MinIO의 URL. 만약 입력되지 않으면 MinIO 관련 기능이 작동하지 않습니다."
        }
    )
    minio_access_key: str = field(
        default=None,
        metadata={
            "help": "MinIO의 minio_access_key. 만약 입력되지 않으면 MinIO 관련 기능이 작동하지 않습니다."
        }
    )
    minio_secret_key: str = field(
        default=None,
        metadata={
            "help": "MinIO의 minio_secret_key. 만약 입력되지 않으면 MinIO 관련 기능이 작동하지 않습니다."
        }
    )
    dataset_load_method: str = field(
        default=None,
        metadata={
            "help": "모델을 불러올 때 사용할 방식을 지정합니다. 사용 가능한 옵션은 ['huggingface'(허깅페이스 허브), 'minio'(MinIO 저장소)]이며, 기본값은 'huggingface'입니다."
        }
    )
    train_data: str = field(
        default=None,
        metadata={
            "help": "학습 데이터셋의 파일 경로 또는 Hugging Face 데이터셋의 이름을 지정합니다."
        }
    )
    train_data_dir: Optional[str] = field(
        default=None,
        metadata={
            "help": "학습 데이터셋이 저장된 서브 폴더 경로를 지정합니다. 데이터셋이 여러 폴더로 구성된 경우 사용합니다."
        }
    )
    train_data_split: str = field(
        default='train',
        metadata={
            "help": "학습 데이터셋에서 사용할 분할(split) 이름을 지정합니다. 기본값은 'train'입니다."
        }
    )
    test_data: Optional[str] = field(
        default=None,
        metadata={
            "help": "테스트 데이터셋의 파일 경로 또는 Hugging Face 데이터셋의 이름을 지정합니다. 선택 사항입니다."
        }
    )
    test_data_dir: Optional[str] = field(
        default=None,
        metadata={
            "help": "테스트 데이터셋이 저장된 서브 폴더 경로를 지정합니다. 데이터셋이 여러 폴더로 구성된 경우 사용합니다."
        }
    )
    test_data_split: str = field(
        default='test',
        metadata={
            "help": "테스트 데이터셋에서 사용할 분할(split) 이름을 지정합니다. 기본값은 'test'입니다."
        }
    )
    train_test_split_ratio: Optional[float] = field(
        default=0.05,
        metadata={
            "help": "테스트 데이터셋이 별도로 제공되지 않을 경우, 학습 데이터셋을 분할하여 테스트 데이터로 사용할 비율을 지정합니다. 기본값은 0.05(5%)입니다."
        }
    )
    dataset_main_column: str = field(
        default="text",
        metadata={
            "help": "토크나이징할 주 데이터 컬럼의 이름을 지정합니다. 일반적으로 텍스트 데이터가 포함된 컬럼입니다. 기본값은 'text'입니다."
        }
    )
    dataset_sub_column: str = field(
        default="label",
        metadata={
            "help": "분류나 기타 작업에 사용될 부가 정보(예: 라벨, 요약 등)가 포함된 컬럼의 이름을 지정합니다. 기본값은 'label'입니다."
        }
    )
    dataset_minor_column: str = field(
        default="negative",
        metadata={
            "help": "데이터셋에 존재할 수 있는 추가(마이너) 컬럼의 이름을 지정합니다. 예를 들어 부정적인 문장 등이 포함될 수 있습니다. 기본값은 'negative'입니다."
        }
    )
    dataset_last_column: str = field(
        default="",
        metadata={
            "help": "데이터 셋이 4가지 컬럼이 필요한 경우 사용합니다. 기본값은 ''입니다."
        }
    )
    data_filtering: bool = field(
        default=True,
        metadata={
            "help": "입력 데이터가 정상적인 문자열(STR) 형식인지 확인하는 옵션입니다. True이면 데이터 형식을 검증하여 필터링합니다."
        }
    )
    tokenizer_max_len: int = field(
        default=2048,
        metadata={
            "help": "토크나이저가 한 번에 처리할 수 있는 최대 시퀀스 길이를 지정합니다. 이 값을 초과하는 경우 토큰이 잘리거나 에러가 발생할 수 있으므로 데이터에 맞게 조정해야 합니다."
        }
    )
    minio_data_load_bucket: Optional[str] = field(
        default='data',
        metadata={
            "help": "MinIO에서 Dataset을 불러올 때 사용할 버킷(bucket) 이름을 지정합니다. 기본값은 'data'입니다."
        }
    )

@dataclass
class AdditionalTrainerArguments:
    mlm_probability: float = field(
        default=0.3,
        metadata={
            "help": "Masked Language Modeling(MLM) 적용 시 단어를 마스킹할 확률을 지정합니다. 기본값 0.3은 ModernBERT의 설정과 유사합니다."
        }
    )
    num_labels: Optional[int] = field(
        default=None,
        metadata={
            "help": "텍스트 분류와 같은 분류 작업을 수행할 때 사용될 라벨(범주)의 총 개수를 지정합니다."
        }
    )
    dpo_loss_type: Optional[str] = field(
        default='sigmoid',
        metadata={
            "help": "DPO에서 사용할 loss 종류. 기본값은 sigmoid 이며, hinge 로도 설정 가능. sigmoid 는 soft한 확률 기반, hinge 는 더 결정적인 margin 기반 접근.",
            "choices": ["sigmoid", "hinge", "ipo", "exo_pair", "nca_pair", "robust", "bco_pair", "sppo_hard", "aot_pair", "apo_down", 'discopop'],
        },
    )
    dpo_beta: Optional[float] = field(
        default=0.1,
        metadata={
            "help": "dpo_beta는 DPO(Direct Preference Optimization)에서 사용되는 하이퍼파라미터로, 모델의 선호도를 조정하는 데 사용됩니다. 기본값은 0.1입니다."
        }
    )
    dpo_label_smoothing: Optional[float] = field(
        default=0.0,
        metadata={
            "help": " DPO의 binary classification적 측면에서 부드러운 라벨을 사용할지 조정. 일반적으로 0.0 (비활성화)이나 0.1 사용 가능."
        }
    )


@dataclass
class SentenceTransformerArguments:
    st_pooling_mode: Optional[str] = field(
        default='mean',
        metadata={
            "help": "Sentence-Transformer 학습 시 사용할 풀링(pooling) 방식을 지정합니다. 옵션에는 'cls', 'lasttoken', 'max', 'mean', 'mean_sqrt_len_tokens', 'weightedmean' 등이 있으며, 기본값은 'mean'입니다."
        }
    )
    st_dense_feature: Optional[int] = field(
        default=0,
        metadata={
            "help": "Sentence-Transformer 모델의 dense layer에 포함될 피처(feature) 수를 지정합니다. 0으로 설정하면 dense layer 없이 구성됩니다. 기본값은 0입니다."
        }
    )
    st_loss_func: Optional[str] = field(
        default="CosineSimilarityLoss",
        metadata={
            "help": "Sentence-Transformer 학습에 사용할 손실 함수(loss function)를 지정합니다. 기본값은 'CosineSimilarityLoss'입니다."
        }
    )
    st_evaluation: Optional[str] = field(
        default=None,
        metadata={
            "help": "Sentence-Transformer 모델 평가 방식을 지정합니다. 별도의 평가 지표나 방법을 적용할 경우 사용합니다."
        }
    )
    st_guide_model: Optional[str] = field(
        default="nlpai-lab/KURE-v1",
        metadata={
            "help": "CachedGISTEmbedLoss와 같은 손실 함수를 사용할 때 참조할 Sentence-Transformer 가이드 모델을 지정합니다. 기본값은 'nlpai-lab/KURE-v1'입니다."
        }
    )
    st_cache_minibatch: Optional[int] = field(
        default=16,
        metadata={
            "help": "Sentence-Transformer 손실 함수 계산 시 사용할 미니 배치의 크기를 지정합니다. 기본값은 16입니다."
        }
    )
    st_cache_gist_temperature: Optional[float] = field(
        default=0.01,
        metadata={
            "help": "Sentence-Transformer CachedGISTEmbedLoss에서 사용할 온도(temperature) 값을 지정합니다. 이 값은 손실 함수의 민감도를 조절하며, 기본값은 0.01입니다."
        }
    )
    st_triplet_margin: Optional[int] = field(
        default=5,
        metadata={
            "help": "트리플렛 손실(triplet loss) 계산 시 적용할 마진(margin) 값을 지정합니다. 기본값은 5입니다."
        }
    )

    st_use_adaptivelayerloss: Optional[bool] = field(
        default=False,
    )
    st_adaptivelayerloss_n_layer: Optional[int] = field(
        default=4
    )

@dataclass
class DeepSpeedArguments:
    use_deepspeed: bool = field(
        default=False,
        metadata={
            "help": "DeepSpeed 인자를 만들어 Trainer에 삽입할지 결정합니다."
        }
    )
    ds_jsonpath: Optional[str] = field(
        default=None,
        metadata={
            "help": "DeepSpeed 사용시, 세부 설정을 담은 Json파일의 경로."
        }
    )
    ds_preset: Optional[str] = field(
        default='zero-2',
        metadata={
            "help": "학습 데이터셋이 저장된 서브 폴더 경로를 지정합니다. 데이터셋이 여러 폴더로 구성된 경우 사용합니다."
        }
    )
    ds_stage2_bucket_size: Optional[float] = field(
        default=5e8,
        metadata={
            "help": "allgather_bucket_size와 reduce_bucket_size는 사용 가능한 GPU 메모리와 통신 속도를 절충합니다. 값이 작을수록 통신 속도가 느려지고 더 많은 GPU 메모리를 사용할 수 있습니다. 예를 들어, 배치 크기가 큰 것이 약간 느린 훈련 시간보다 더 중요한지 균형을 맞출 수 있습니다."
        }
    )
    ds_stage3_sub_group_size: Optional[float] = field(
        default=1e7,
        metadata={
            "help": "allgather_bucket_size와 reduce_bucket_size는 사용 가능한 GPU 메모리와 통신 속도를 절충합니다. 값이 작을수록 통신 속도가 느려지고 더 많은 GPU 메모리를 사용할 수 있습니다. 예를 들어, 배치 크기가 큰 것이 약간 느린 훈련 시간보다 더 중요한지 균형을 맞출 수 있습니다."
        }
    )
    ds_stage3_max_live_parameters: Optional[float] = field(
        default=1e7,
        metadata={
            "help": "allgather_bucket_size와 reduce_bucket_size는 사용 가능한 GPU 메모리와 통신 속도를 절충합니다. 값이 작을수록 통신 속도가 느려지고 더 많은 GPU 메모리를 사용할 수 있습니다. 예를 들어, 배치 크기가 큰 것이 약간 느린 훈련 시간보다 더 중요한지 균형을 맞출 수 있습니다."
        }
    )
    ds_stage3_max_reuse_distance: Optional[float] = field(
        default=1e7,
        metadata={
            "help": "allgather_bucket_size와 reduce_bucket_size는 사용 가능한 GPU 메모리와 통신 속도를 절충합니다. 값이 작을수록 통신 속도가 느려지고 더 많은 GPU 메모리를 사용할 수 있습니다. 예를 들어, 배치 크기가 큰 것이 약간 느린 훈련 시간보다 더 중요한지 균형을 맞출 수 있습니다."
        }
    )


@dataclass
class PeftConfigArguments:
    use_peft: bool = field(
        default=False,
        metadata={
            "help": "Peft 인자를 만들어 Trainer에 삽입할지 결정합니다."
        }
    )
    peft_type: str = field(
        default="lora",
        metadata={
            "help": ("The PEFT type to use."),
            "choices": ["none", "lora", "ia3", "adalora", "llama-adapter", "vera", "ln_tuning"],
        },
    )

    # LoRA를 사용하는 경우
    lora_target_modules: str = field(
        default="q_proj,k_proj",
        metadata={
            "help": "peft의 Target이 될 Module을 선택합니다. 이는 모델의 Parameter의 Name을 지정해야 합니다. 정확하게 일치하는 것이 없다면, Trainer는 자동으로 해당 단어를 Endword로 사용하는 Layer를 선택하려고 시도합니다. 값은 ,로 구분되는 문자열이 입력되어야 합니다. 입력된 문자열에서 공백과 '\n'은 모두 제거되고 ,을 기준으로 list로 나눕니다."
        }
    )
    lora_r: Optional[int] = field(
        default=16,
        metadata={
            "help": "low-rank 분해 행렬의 차원(rank). 기본값은 16입니다."
        }
    )
    lora_alpha: Optional[int] = field(
        default=32,
        metadata={
            "help": "스케일링 팩터로, low-rank 업데이트의 크기를 조절합니다. 기본값은 32입니다."
        }
    )
    lora_dropout: Optional[float] = field(
        default=0.05,
        metadata={
            "help": "peft 레이어에 적용되는 dropout 확률. 기본값은 0.05입니다."
        }
    )
    lora_modules_to_save: Optional[str] = field(
        default="",
        metadata={
            "help": "모델에서 Unfreeze & Train할 Layer를 선택합니다. 만약 Special token을 Chat Template에 포함하고 있는 모델을 학습하는 경우, Embedding Layer 및 LMHead가 반드시 Trainable한 Parameter에 포함되어야 합니다. 이 때, 해당 Layer를 이 Arguments에 입력하여야 합니다. 값은 ,로 구분되는 문자열이 입력되어야 합니다. 입력된 문자열에서 공백과 '\n'은 모두 제거되고 ,을 기준으로 list로 나눕니다."
        }
    )

    # AdaLoRA를 사용하는 경우
    adalora_init_r: Optional[int] = field(
        default=12,
        metadata={"help": "Initial AdaLoRA rank"},
    )
    adalora_target_r: Optional[int] = field(
        default=4,
        metadata={"help": "Target AdaLoRA rank"},
    )
    adalora_tinit: Optional[int] = field(
        default=50,
        metadata={"help": "Number of warmup steps for AdaLoRA wherein no pruning is performed"},
    )
    adalora_tfinal: Optional[int] = field(
        default=100,
        metadata={
            "help": "Fix the resulting budget distribution and fine-tune the model for tfinal steps when using AdaLoRA"
        },
    )
    adalora_delta_t: Optional[int] = field(
        default=10,
        metadata={"help": "Interval of steps for AdaLoRA to update rank"},
    )
    adalora_orth_reg_weight: float = field(
        default=0.5,
        metadata={"help": "Orthogonal regularization weight for AdaLoRA"},
    )

    # 기타 다른 Peft Method
    ia3_target_modules: Optional[str] = field(
        default_factory=lambda: None,
        metadata={"help": "Target modules for the IA3 method."},
    )
    feedforward_modules: Optional[str] = field(
        default_factory=lambda: None,
        metadata={"help": "Target feedforward modules for the IA3 method."},
    )
    adapter_layers: Optional[int] = field(
        default=30,
        metadata={"help": "Number of adapter layers (from the top) in llama-adapter"},
    )
    adapter_len: Optional[int] = field(
        default=16,
        metadata={"help": "Number of adapter tokens to insert in llama-adapter"},
    )
    vera_target_modules: Optional[str] = field(
        default_factory=lambda: None,
        metadata={"help": "Target modules for the vera method."},
    )
    ln_target_modules: Optional[str] = field(
        default_factory=lambda: None,
        metadata={"help": "Target modules for the ln method."},
    )
