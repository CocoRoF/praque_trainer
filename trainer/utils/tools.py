import re
from transformers import AutoModel, AutoTokenizer
from sentence_transformers import SentenceTransformer
from huggingface_hub import hf_hub_download, snapshot_download
import bitsandbytes as bnb
import torch.nn as nn

def is_quantized_model(model: nn.Module) -> bool:
    for module in model.modules():
        if isinstance(module, (bnb.nn.Linear4bit, bnb.nn.Linear8bitLt)):
            return True
    return False

def print_trainable_parameters(model):
    trainable = 0
    total = 0
    for name, param in model.named_parameters():
        num_params = param.numel()
        total += num_params
        if param.requires_grad:
            trainable += num_params
    print(f"Trainable params: {trainable:,} | Total params: {total:,} | Trainable%: {100 * trainable / total:.2f}%")

def selective_freeze(model, unfreeze_keywords):
    """
    model: nn.Module (Huggingface 모델)
    unfreeze_keywords: list of str (이 문자열이 파라미터 이름에 포함된 경우 unfreeze)
    """
    for name, param in model.named_parameters():
        if any(keyword in name for keyword in unfreeze_keywords):
            param.requires_grad = True
            # print(f"Unfrozen: {name}")
        else:
            param.requires_grad = False
            # print(f"Frozen: {name}")

def select_best_checkpoint_folder(folders, pattern=r'\D*(\d+)\D*$'):
    """
    폴더 이름에서 정규표현식에 맞는 숫자를 추출하여, 마지막 숫자(여기서는 마지막에 위치한 숫자 그룹)를 기준으로
    가장 큰 숫자를 가진 폴더명을 반환합니다.
    
    기본 패턴은 폴더 이름 끝에 있는 숫자를 추출하는 예제로, 예) 'son-project-v2/checkpoint-4452' 또는 'STR4452'
    """
    compiled_pattern = re.compile(pattern)
    candidates = []
    for folder in folders:
        match = compiled_pattern.search(folder)
        if match:
            num = int(match.group(1))
            candidates.append((folder, num))
    if candidates:
        return max(candidates, key=lambda x: x[1])[0]
    return None

def parse_csv_string(input_string: str):
    """
    입력된 문자열에서 모든 공백(' ')과 개행('\n')을 제거하고,
    ','를 기준으로 리스트를 만들어 반환합니다.
    """
    tmp = input_string.replace(" ", "").replace("\n", "")
    items = tmp.split(",")
    items = [x for x in items if x]

    return items

def huggingface_repo_downloader(repo_id: str, save_dir: str = "./local_save", token: str = None) -> None:
    """
    Hugging Face 레포지토리의 모든 파일을 로컬로 다운로드합니다.
    
    Parameters:
        repo_id (str): 다운로드할 Hugging Face 레포의 식별자 (예: "username/repo_name").
        save_dir (str): 파일들을 저장할 로컬 디렉토리 경로.
        token (str, optional): 프라이빗 레포 접근 시 필요한 인증 토큰.
    
    Returns:
        None
    """
    snapshot_download(repo_id, local_dir=save_dir, token=token)
    print(f"[INFO] {repo_id} 레포의 모든 파일이 {save_dir}에 다운로드 되었습니다.")


def huggingface_model_saver(model_name, save_dir: str = "./local_save", token:str = None) -> None:
    """
    Hugging Face 모델과 토크나이저를 로드하여 로컬에 저장합니다.
    
    Parameters:
        model_name (str): Hugging Face Hub의 모델 식별자.
        save_dir (str): 모델을 저장할 로컬 디렉토리 경로.
    
    Returns:
        None
    """
    model = AutoModel.from_pretrained(model_name, token=token)
    tokenizer = AutoTokenizer.from_pretrained(model_name, token=token)
    model.save_pretrained(save_dir)
    tokenizer.save_pretrained(save_dir)
    print(f"[INFO] HuggingFace모델이 {save_dir}에 저장되었습니다.")

def sentence_transformer_model_saver(model_name, save_dir: str = "./local_save") -> None:
    """
    SentenceTransformer 모델을 로드하여 로컬에 저장합니다.
    
    Parameters:
        model_name (str): SentenceTransformer 모델 이름 또는 Hugging Face Hub의 식별자.
        save_dir (str): 모델을 저장할 로컬 디렉토리 경로.
    
    Returns:
        None
    """
    model = SentenceTransformer(model_name)
    model.save(save_dir)
    print(f"[INFO] SentenceTransformer모델이 {save_dir}에 저장되었습니다.")

def automatic_model_saver(model_name, save_dir: str = None) -> None:
    """
    주어진 모델 리포지토리에서 "config_sentence_transformers.json" 파일을 확인하여,
    파일이 존재하면 SentenceTransformer 방식을 사용하여 모델을 로드 및 저장하고,
    파일이 없거나 로드에 실패하면 HuggingFace 모델 로더를 사용합니다.
    
    Parameters:
        model_name (str): 모델 이름 또는 Hugging Face Hub의 식별자.
        save_dir (str, optional): 모델을 저장할 로컬 디렉토리 경로. 미지정시 './local_{model_name}'에 저장됨.
    
    Returns:
        None
    """
    if save_dir is None:
        model_path = model_name.replace("/", "__")
        save_dir = f"./local_{model_path}"
    
    config_file = "config_sentence_transformers.json"
    
    try:
        print("[INFO] 모델 리포지토리에서 config_sentence_transformers.json 파일 검사 중...")
        hf_hub_download(repo_id=model_name, filename=config_file)
        print("[INFO] sentence_transformers.config 파일이 확인되었습니다. SentenceTransformer 방식으로 로드합니다.")
        sentence_transformer_model_saver(model_name, save_dir)
    except Exception as e_config:
        print("[WARNING] sentence_transformers.config 파일을 찾지 못했습니다. HuggingFace 모델 방식으로 로드합니다.")
        try:
            huggingface_model_saver(model_name, save_dir)
        except Exception as e_hf:
            print("[ERROR] 두 방식 모두 모델 로드에 실패하였습니다.")
            print(f"HuggingFace 모델 로더 에러: {e_hf}")

if __name__ == "__main__":
    model_name = "x2bee/ModernBERT-ecs-GIST-category"
    automatic_model_saver(model_name)