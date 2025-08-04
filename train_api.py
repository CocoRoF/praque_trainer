import subprocess
import os
from trainer.utils.env_config import get_huggingface_token, get_huggingface_user_id, get_minio_config

## Arguments
hugging_face_user_id = get_huggingface_user_id()
hugging_face_token = get_huggingface_token()
mlflow_url = "https://polar-mlflow-git.x2bee.com/"
minio_config = get_minio_config()
minio_url = minio_config["url"]
minio_access_key = minio_config["access_key"]
minio_secret_key = minio_config["secret_key"]
number_gpu = 8 ## [중요/필수] 공통설정. GPU 숫자.
project_name = "polar-gemma-ece-27b" ## [중요/필수] 공통설정. 현재 프로젝트의 이름.
training_method = "gemma" ## ## [중요/필수] 공통설정. 수행할 Training의 Method.
model_load_method = "huggingface" ## [중요/필수] 공통설정. 모델을 어디서 로드할 것인지. (['huggingface', 'minio'])
dataset_load_method = "huggingface" ## [중요/필수] 공통설정. 데이터셋을 어디서 로드할 것인지. (['huggingface', 'minio'])
use_sfttrainer = True ## [중요/필수] TRL의 SFTTrainer 를 사용할 것인지. 고급사용자옵션. 기본은 False.
use_dpotrainer = False ## [중요/필수] TRL의 DPOTrainer 를 사용할 것인지. 고급사용자옵션. 기본은 False.
use_ppotrainer = False ## [중요/필수] TRL의 PPOTrainer 를 사용할 것인지. 고급사용자옵션. 기본은 False.
use_grpotrainer = False ## [중요/필수] TRL의 GRPOTRainer 를 사용할 것인지. 고급사용자옵션. 기본은 False.
use_deepspeed = True ## [중요/필수] DeepSpeed를 사용할 것인지. 고급사용자옵션. 기본은 False.
ds_jsonpath = "" ## [옵션] 만약 DeepSpeed를 사용하는 경우 Json 경로.
ds_preset = "zero-2" ## [옵션] DeepSpeed를 사용하는 경우 설정의 Preset. 기본은 zero-2. [zero-1, zero-2, zero-3]가 가능.
ds_stage2_bucket_size = 5e8 ## [옵션] DeepSpeed를 사용하는 경우, allgather_bucket_size와 reduce_bucket_size는 사용 가능한 GPU 메모리와 통신 속도를 절충합니다. 값이 작을수록 통신 속도가 느려지고 더 많은 GPU 메모리를 사용할 수 있습니다. 예를 들어, 배치 크기가 큰 것이 약간 느린 훈련 시간보다 더 중요한지 균형을 맞출 수 있습니다.
ds_stage3_sub_group_size = 1e7
ds_stage3_max_live_parameters = 1e7
ds_stage3_max_reuse_distance = 1e7
model_name_or_path = "google/gemma-3-27b-it" ## [중요/필수] 공통설정. 모델을 로드할 경로.
                                                             ##Huggingface의 경우 'id/repo' 형태이며, minio의 경우 model이 들어있는 bucket 아래의 folder명을 지칭하여야 함.
model_subfolder = "" ## [옵션] Huggingface에서 모델을 로드하는 경우 해당 모델의 폴더명
config_name = "" ## [옵션] Huggingface의 AutoModel + Config를 통해 모델을 초기화 시키는 경우에 사용. (현재 미구현)
tokenizer_name = "" ## [옵션] 보통 모델과 토크나이저는 함께 저장되는데, 만약 토크나이저만 다른 곳에서 로드하고 싶은 경우, Huggingface 경로 설정.
train_data = "CocoRoF/e-commerce_polar_dataset" ## [중요/필수] 공통설정. GPU 숫자.
train_data_dir = "" ## [옵션] Huggingface에서 로드시 기본적으로 default dataset의 경우 해당 데이터셋을 가져오나, 특정 폴더를 지칭하고 싶은 경우에 사용.
train_data_split = "train" ## [옵션] datasets.Dataset의 Split 설정. 기본값은 train임. (미설정시에도 Train으로 작동하도록 하는 것이 좋음.)
test_data = "" ## [옵션] Test 데이터가 따로 있다면 Train과 같은 원리로 지정. 만약 None or ""으로 설정한다면, Split_ratio에 따라, Train 데이터를 이용하여 Test 데이터로 사용함.
test_data_dir = "" ## [옵션] 위와 같음.
test_data_split = "" ## [옵션] 위와 같음.

# Dataset의 경우 training_method에 따라 기대하는 형태가 정해져있음.
# 예를 들어 mlm, clm의 경우 단일 text를 column으로 가지는 데이터셋을 기대함.
# cls의 경우 [text, labels]의 형태를 기대함.
# ce의 경우 [sentence1, sentence2, labels]의 형태를 기대함
# sentence-transformer의 경우 목적에 따라 다른데, [anchor, positive, negative] 혹은 [sentence1, sentence2, labels]의 형태의 데이터를 기대함.
# 기본적으로 Trainer는 이러한 형태의 데이터가 제시되었을 것이라 기대하며(컬럼 이름은 상관하지 않음), 자동으로 컬럼 이름을 바꾸어 사용함.
# 여러 컬럼이 존재하는 경우, 아래 파라미터를 조정하여 특정 컬럼만 제시하여야 함.
dataset_main_column = "text" ## [옵션] Dataset의 특정 컬럼(Feature)을 선택해야 하는 경우 사용함.
dataset_sub_column = "" ## [옵션] 두 번째 컬럼을 지칭하는 경우.
dataset_minor_column = "" ## [옵션] 세 번째 컬럼을 지칭하는 경우

push_to_hub = True ## [필수] Huggingface에 푸쉬하는 경우 사용. 만약 hub_model_id 인자가 제공되면 해당 repo에 푸쉬를 진행하며, 존재하지 않는다면
                   ## .env의 {HUGGING_FACE_USER_ID}/{project_name} 의 repo로 푸쉬를 진행함.
push_to_minio = False ## [필수] Huggingface에 푸쉬하는 경우 사용. minio_model_load_bucket에 지정된 bucket의 project_name폴더 아래로 푸쉬함. 이는 Trainer에 의해 local에 저장된 자료를 모두 푸쉬하는 형태로 작동함.
minio_model_load_bucket = "models" ## [필수] MinIO에서 모델을 불러오는 Bucket의 이름. 기본값은 models
minio_model_save_bucket = "models" ## [필수] MinIO에서 모델을 저장하는는 Bucket의 이름. 기본값은 models
minio_data_load_bucket = "data" ## [필수] MinIO에서 데이터를 로드하는 Bucket의 이름. 기본값은 data
mlm_probability = 0.2 ## [옵션] mlm형태의 학습을 진행하는 경우 필수. 주어진 텍스트를 어느 비율로 Masking하는지를 결정함. 보통은 0.1~0.3을 사용.
num_labels = 17 ## [옵션] cls형태의 학습을 진행하는 경우 필수. 분류해야하는 라벨의 수.
st_pooling_mode = "mean" ## [옵션] Sentence-Transformer의 Pooling 옵션. 가능한 것은 'cls', 'lasttoken', 'max', 'mean', 'mean_sqrt_len_tokens', 'weightedmean'
st_dense_feature = 0 ## [옵션] Sentence-Transformer의 최종 Dense레이어의 Hidden Size. 보통은 Model의 Hidden Size와 동일하게 설정함. 0으로 설정하는 경우 Dence Layer를 사용하지 않음.
st_loss_func = "CosineSimilarityLoss" ## [옵션] Sentence-Transformer사용시 필수 옵션. 어떠한 Loss Function을 사용하여 학습할지 결정. Loss Function에 따라 요구하는 데이터의 형태가 달라짐.
st_evaluation = "" ## [옵션] Sentence-Transformer에서 Evaluation 진행시 필수 옵션. 해당 Evaluator를 사용하여 학습 후 평가 진행.
st_guide_model = "nlpai-lab/KURE-v1" ## [옵션] Sentence-Transformer에서 GISTEmbedLoss를 사용하는 경우 설정. 이를 보조하는 Guide 모델을 지정해야 함.
st_cache_minibatch = 16 ## [옵션] Sentence-Transformer 에서 Cache~~류의 Loss사용하는 경우 설정. 해당 로스의 미니배치 사이즈(일반적으로 생각하는 배치)이며, 기존에 지정하는 bache의 경우 cache되는 총 배치를 지정해야 함. 이 경우 Memory는 오직 minibatch에만 영향을 받음.
st_triplet_margin = 5 ## [옵션] Sentence-Transformer에서 Margin을 사용하는 Triplet Loss류를 사용하는 경우 설정. 해당 Loss의 성능의 경우 Margin에 심각하게 영향을 받음. 이를 조정하여 학습의 성능을 검증해야 함.
flash_attn = True ## [옵션] Flash-Attention2를 사용할지를 결정. 만약 GPU가 이를 지원하는 경우 사용하는 것을 권장함.
is_resume = False ## [옵션] 만약 model_name_or_path에 Training 정보를 제공하며, 중간에 학습이 중단된 경우, 이를 True로 설정하여 다시 시작할 수 있음.
model_commit_msg = "gemma-try" ## [옵션] 모델에 대한 설명. Commit 메세지. Metadata에 저장됨.
train_test_split_ratio = 0.0 ## [옵션] 만약 Test데이터를 제공하지 않고, Train 데이터를 분할하여 사용하는 경우, 그 비율. 0이면 작동하지 않음.
data_filtering = False ## [옵션] 데이터가 None값과 같은 결측치로 인해 오류가 발생하는 경우 사용. 데이터의 Column을 돌며 Str, Float, Int 형태의 데이터인지 검사함. 이외의 데이터는 모두 오류로 간주.
tokenizer_max_len = 0 ## [필수] Tokenizer의 Max-Length. 이것이 처리되는 데이터의 최종 길이가 됨.
output_dir = "" ## [옵션] 모델의 저장 경로를 임의로 지정하고 싶은 경우 사용. 만약 ""으로 둔다면 "result__{HUGGING_FACE_USER_ID}__{project_name}"를 사용함.
overwrite_output_dir = False ## [옵션] Output_directory를 덮어씌울지 결정. 기본적으로 덮어쓰기를 함. False의 경우, 폴더가 겹치는 경우 진행되지 않음.
use_stableadamw = False ## [필수] StableAdamW Optimizer를 사용할지 결정. 미사용한다면 기본적으로 AdamW Optimizer를 사용함.
optim = "adamw_torch" ## [필수] 만약 StableAdamW가 False인 경우 해당 인자를 통해 Optimizer를 설정할 수 있음.
adam_beta1 = 0.900 ## [필수] Adam류 Optimizer를 사용하는 경우 Beta1 ~ Beta2 범위로 Beta를 결정하여야 함.
adam_beta2 = 0.990 ## [필수] Adam류 Optimizer를 사용하는 경우 Beta1 ~ Beta2 범위로 Beta를 결정하여야 함.
adam_epsilon = 1e-7 ## [필수] Adam류 Optimizer의 Epsilon.
save_strategy = "steps" ## [필수] 저장 방법. "no", "epoch", "steps" 중 하나를 사용할 것. no면 저장하지 않음.
save_steps = 1000 ## [옵션] 만약 save_strategy Steps인 경우, 어떤 스텝마다 저장을 수행할지 결정.
eval_strategy = "steps" ## [필수] 평가를 수행하는 방법. "no", "epoch", "steps" 중 하나를 사용할 것. No거나 Test data가 없다면 진행하지 않음.
eval_steps = 1000 ## [옵션] 만약 eval_strategy Steps인 경우, 어떤 스텝마다 저장을 수행할지 결정.
save_total_limit = 1 ## [옵션] 만약 저장을 진행한다면, 해당 저장을 통해 최대 몇 개의 Checkpoint를 저장할지 결정.
hub_model_id = "" ## [옵션] 만약 Huggingface에 푸쉬한다면, 어디로 푸쉬할지 임의로 결정. 설정하지 않으면 .env의 {HUGGING_FACE_USER_ID}/{project_name} 의 repo로 푸쉬를 진행함.
hub_strategy = "every_save" ## [옵션] Huggingface에 푸쉬하는 경우, 어떤 방식으로 푸쉬할지 결정. 기본값은 checkpoint. ['end', 'every_save', 'checkpoint', 'all_checkpoints']가 가능. checkpoint를 권장.
logging_steps = 5 ## [필수] 학습 시 학습 현황에 대해 몇 step마다 log를 찍을지 결정.
max_grad_norm = 1 ## [필수] 기울기 클리핑의 값. 기본값은 1로 하며, gradient norm이 크다면 이를 줄이는 방식으로 조정.

# 배치사이즈는 매우 중요한 요소인데
# 최종적인 배치사이즈는 [number_gpu * per_device_train_batch_size * gradient_accumulation_steps] 로 계산됨.
per_device_train_batch_size = 1 ## [중요/필수] 학습시 GPU마다 batch의 크기. Memory 점유에 직접적 영향을 끼침.
per_device_eval_batch_size = 1 ## [중요/필수] 평가시 GPU마다 batch의 크기. Memory 점유에 직접적 영향을 끼침.
gradient_accumulation_steps = 32 ## [중요/필수] 기울기 업데이트를 해당 값만큼 누적한 뒤 진행. Batch를 증가시킨 효과가 있음.
learning_rate = 1e-4 ## [중요/필수] 학습률. 너무 크면 기울기 폭발 가능성, 너무 작으면 오버피팅 가능성.
gradient_checkpointing = True
num_train_epochs = 1 ## [중요/필수] Epoch의 수.
warmup_ratio = 0.1 ## [중요/필수] 기울기 Warmup의 비율.
weight_decay = 0.01 ## [필수] weight_decay의 값.
do_train = True ## [필수]
do_eval = True ## [필수]
bf16 = True ## [필수] bf16 정밀도를 사용할지 결정.
use_peft = True ##[필수] Lora 인자를 만들어 Trainer에 삽입할지 결정
peft_type = 'lora' ##[필수] 어떤 PEFT 알고리듬을 사용할지
lora_target_modules = "q_proj,k_proj,v_proj,o_proj" ##LoRA의 Target이 될 Module을 선택합니다. 이는 모델의 Parameter의 Name을 지정해야 합니다. 정확하게 일치하는 것이 없다면, Trainer는 자동으로 해당 단어를 Endword로 사용하는 Layer를 선택하려고 시도합니다.
lora_r = 16 ## low-rank 분해 행렬의 차원(rank). 기본값은 16입니다.
lora_alpha = 32 ## 스케일링 팩터로, low-rank 업데이트의 크기를 조절합니다. 기본값은 32입니다.
lora_dropout = 0.05 ## LoRA 레이어에 적용되는 dropout 확률. 기본값은 0.05입니다.
lora_modules_to_save = "embed_tokens,lm_head" ## 모델에서 Unfreeze & Train할 Layer를 선택합니다. 만약 Special token을 Chat Template에 포함하고 있는 모델을 학습하는 경우, Embedding Layer 및 LMHead가 반드시 Trainable한 Parameter에 포함되어야 합니다. 이 때, 해당 Layer를 이 Arguments에 입력하여야 합니다. 값은 ,로 구분되는 문자열이 입력되어야 합니다. 입력된 문자열에서 공백과 '\n'은 모두 제거되고 ,을 기준으로 list로 나눕니다.
dpo_beta = 0.25
dpo_label_smoothing = 0.1
# script_dir = os.path.dirname(os.path.abspath(__file__))
# train_script_path = os.path.join(script_dir, "train.py")

command = [
    "deepspeed" if use_deepspeed else "torchrun",
]

if use_deepspeed:
    command += [f"--num_gpus={number_gpu}"]
else:
    command += [f"--nproc_per_node={number_gpu}"]

command += [
    "train.py",
    f"--hugging_face_user_id={hugging_face_user_id}",
    f"--hugging_face_token={hugging_face_token}",
    f"--mlflow_url={mlflow_url}",
    f"--minio_url={minio_url}",
    f"--minio_access_key={minio_access_key}",
    f"--minio_secret_key={minio_secret_key}",
    f"--model_name={project_name}",
    f"--model_train_method={training_method}",
    f"--model_load_method={model_load_method}",
    f"--dataset_load_method={dataset_load_method}",
    f"--model_name_or_path={model_name_or_path}",
    f"--model_subfolder={model_subfolder}",
    f"--config_name={config_name}",
    f"--tokenizer_name={tokenizer_name}",
    f"--train_data={train_data}",
    f"--train_data_dir={train_data_dir}",
    f"--train_data_split={train_data_split}",
    f"--test_data={test_data}",
    f"--test_data_dir={test_data_dir}",
    f"--test_data_split={test_data_split}",
    f"--dataset_main_colunm={dataset_main_column}",
    f"--dataset_sub_colunm={dataset_sub_column}",
    f"--dataset_minor_colunm={dataset_minor_column}",
    f"--push_to_hub={push_to_hub}",
    f"--push_to_minio={push_to_minio}",
    f"--minio_model_load_bucket={minio_model_load_bucket}",
    f"--minio_model_save_bucket={minio_model_save_bucket}",
    f"--minio_data_load_bucket={minio_data_load_bucket}",
    f"--mlm_probability={mlm_probability}",
    f"--num_labels={num_labels}",
    f"--st_pooling_mode={st_pooling_mode}",
    f"--st_dense_feature={st_dense_feature}",
    f"--st_loss_func={st_loss_func}",
    f"--st_evaluation={st_evaluation}",
    f"--st_guide_model={st_guide_model}",
    f"--st_cache_minibatch={st_cache_minibatch}",
    f"--st_triplet_margin={st_triplet_margin}",
    f"--use_sfttrainer={use_sfttrainer}",
    f"--use_dpotrainer={use_dpotrainer}",
    f"--use_ppotrainer={use_ppotrainer}",
    f"--use_grpotrainer={use_grpotrainer}",
    f"--use_deepspeed={use_deepspeed}",
    f"--ds_jsonpath={ds_jsonpath}",
    f"--ds_preset={ds_preset}",
    f"--ds_stage2_bucket_size={ds_stage2_bucket_size}",
    f"--ds_stage3_sub_group_size={ds_stage3_sub_group_size}",
    f"--ds_stage3_max_live_parameters={ds_stage3_max_live_parameters}",
    f"--ds_stage3_max_reuse_distance={ds_stage3_max_reuse_distance}",
    f"--flash_attn={flash_attn}",
    f"--is_resume={is_resume}",
    f"--model_commit_msg={model_commit_msg}",
    f"--train_test_split_ratio={train_test_split_ratio}",
    f"--data_filtering={data_filtering}",
    f"--tokenizer_max_len={tokenizer_max_len}",
    f"--output_dir={output_dir}",
    f"--overwrite_output_dir={overwrite_output_dir}",
    f"--use_stableadamw={use_stableadamw}",
    f"--optim={optim}",
    f"--adam_beta1={adam_beta1}",
    f"--adam_beta2={adam_beta2}",
    f"--adam_epsilon={adam_epsilon}",
    f"--save_strategy={save_strategy}",
    f"--save_steps={save_steps}",
    f"--eval_strategy={eval_strategy}",
    f"--eval_steps={eval_steps}",
    f"--save_total_limit={save_total_limit}",
    f"--hub_model_id={hub_model_id}",
    f"--hub_strategy={hub_strategy}",
    f"--logging_steps={logging_steps}",
    f"--max_grad_norm={max_grad_norm}",
    f"--per_device_train_batch_size={per_device_train_batch_size}",
    f"--per_device_eval_batch_size={per_device_eval_batch_size}",
    f"--gradient_accumulation_steps={gradient_accumulation_steps}",
    f"--learning_rate={learning_rate}",
    f"--gradient_checkpointing={gradient_checkpointing}",
    f"--num_train_epochs={num_train_epochs}",
    f"--warmup_ratio={warmup_ratio}",
    f"--weight_decay={weight_decay}",
    f"--do_train={do_train}",
    f"--do_eval={do_eval}",
    f"--bf16={bf16}",
    f"--use_peft={use_peft}",
    f"--peft_type={peft_type}",
    f"--lora_target_modules={lora_target_modules}",
    f"--lora_r={lora_r}",
    f"--lora_alpha={lora_alpha}",
    f"--lora_dropout={lora_dropout}",
    f"--lora_modules_to_save={lora_modules_to_save}",
    f"--dpo_beta={dpo_beta}",
    f"--dpo_label_smoothing={dpo_label_smoothing}",
]

print("[INFO] Command:", command)

subprocess.run(command, shell=False, check=True)
