import os
import json
import datasets
import shutil
from dotenv import load_dotenv
from minio import Minio
from typing import Union, Optional
from huggingface_hub import create_repo, HfApi
from trainer.utils.tools import huggingface_repo_downloader

load_dotenv()
PLATEER_HGF_TOKEN = os.getenv("PLATEER_HGF_TOKEN")

import datasets

def messages_form_dataset_filter(dataset: datasets.Dataset) -> datasets.Dataset:
    """
    주어진 datasets.Dataset의 'messages' 컬럼을 검증하고,
    유효하지 않은 데이터를 필터링하여 새로운 데이터셋을 반환합니다.

    Args:
        dataset (datasets.Dataset): 검증할 원본 데이터셋.

    Returns:
        datasets.Dataset: 정합성 검증을 통과한 데이터만 포함된 새로운 데이터셋.
    """

    # 데이터 정합성을 검증하는 내부 함수
    def is_valid_entry(example):
        # 'messages' 키 존재 여부 및 리스트 형태 확인
        messages = example.get('messages')
        if not isinstance(messages, list) or not messages:
            return False

        # 리스트 내 각 메시지의 형식 확인
        for message in messages:
            # 딕셔너리 형태 확인
            if not isinstance(message, dict):
                return False

            # 'role'과 'content' 키 존재 확인
            if 'role' not in message or 'content' not in message:
                return False

            # 'role' 값의 타입 및 내용 확인
            if not isinstance(message['role'], str) or message['role'] not in ['system', 'user', 'assistant']:
                return False

            # 'content' 값의 타입 확인 (가장 중요한 부분)
            if not isinstance(message['content'], str):
                return False

        return True

    original_num_rows = len(dataset)

    # .filter() 메소드를 사용하여 유효한 데이터만 남김
    # num_proc를 설정하여 병렬 처리로 속도를 높일 수 있습니다.
    filtered_dataset = dataset.filter(is_valid_entry, num_proc=4)

    filtered_num_rows = len(filtered_dataset)
    num_removed = original_num_rows - filtered_num_rows

    print("=" * 50)
    print("데이터 정합성 검증 및 필터링 완료")
    print(f"원본 데이터 개수: {original_num_rows}")
    print(f"정제 후 데이터 개수: {filtered_num_rows}")
    print(f"제외된 데이터 개수: {num_removed}")
    print("=" * 50)

    return filtered_dataset

def is_valid_input(example, col_name):
    try:
        value = example.get(col_name)
        if value is None or value == "None":
            return False
        if isinstance(value, str) and value.strip() == "":
            return False
        if isinstance(value, dict) and len(value) == 0:
            return False
        if isinstance(value, list) and all(isinstance(item, dict) and len(item) == 0 for item in value):
            return False
        return True
    except Exception:
        return False

def rename_dataset_columns(dataset: datasets.Dataset, *new_names: str) -> datasets.Dataset:
    original_columns = dataset.column_names.copy()
    if len(new_names) > len(original_columns):
        raise ValueError()

    new_dataset = dataset
    for i, new_name in enumerate(new_names):
        current_columns = new_dataset.column_names
        old_name = current_columns[i]
        if old_name != new_name:
            new_dataset = new_dataset.rename_column(old_name, new_name)
    return new_dataset

class DataLoader():
    def __init__(self, hugging_face_user_id, hugging_face_token, minio_url, minio_access_key, minio_secret_key, **kwargs):
        """
        DataLoader 클래스의 생성자입니다.

        Parameters:
            hugging_face_user_id (str): Hugging Face 사용자 ID.
            hugging_face_token (str): Hugging Face 접근 토큰.
            minio_url (str): MinIO 서버 URL.
            minio_access_key (str): MinIO 접근 키.
            minio_secret_key (str): MinIO 비밀 키.
            **kwargs: 추가 인자들을 받아들이며, 확장성을 위해 사용됩니다.

        기능:
            - Hugging Face와 MinIO 관련 설정값을 초기화합니다.
            - MinIO 클라이언트를 생성하며, 해당 클라이언트를 통해 후속 데이터 및 모델 업로드/다운로드 작업을 수행합니다.
        """
        self.HUGGING_FACE_USER_ID = hugging_face_user_id
        self.HUGGING_FACE_TOKEN = hugging_face_token
        self.MINIO_ENDPOINT = minio_url
        self.MINIO_ACCESS_KEY = minio_access_key
        self.MINIO_SECRET_KEY = minio_secret_key

        try:
            self.minio_client = Minio(
                endpoint=self.MINIO_ENDPOINT.replace("http://", "").replace("https://", ""),
                access_key=self.MINIO_ACCESS_KEY,
                secret_key=self.MINIO_SECRET_KEY,
                secure=True,
            )

            self.minio_storage_options = {
                "client_kwargs": {"endpoint_url": self.MINIO_ENDPOINT.replace("http://", "").replace("https://", "")},
                "key": self.MINIO_ACCESS_KEY,
                "secret": self.MINIO_SECRET_KEY,
            }

        except Exception as e:
            print("[ERROR] MinIO Setting Error!")
            print(f"[ERROR] MINIO_ENDPOINT = {self.MINIO_ENDPOINT}")
            print(f"[ERROR] MINIO_ACCESS_KEY = {self.MINIO_ACCESS_KEY}")
            print(f"[ERROR] MINIO_SECRET_KEY = {self.MINIO_SECRET_KEY}")
            print("[ERROR] Please Check MinIO Environment.")
            print(f"[ERROR] {e}")

    def minio_get_subfolders(self, bucket_name: str = "data", is_metadata: bool = True) -> list[str]:
        """
        주어진 버킷(bucket_name)의 최상위에 존재하는 폴더(디렉토리) 이름 리스트를 반환합니다.

        Parameters:
            bucket_name (str): 조회할 버킷 이름. 기본값은 "data".

        Returns:
            list[str]: 버킷 내 최상위 폴더 이름 리스트.

        Raises:
            ValueError: 버킷 이름 또는 MinIO 클라이언트 설정이 잘못된 경우.
        """
        subfolders = set()
        try:
            objects = self.minio_client.list_objects(bucket_name, recursive=False)
        except Exception:
            raise ValueError("[ERROR] Please Check Bucket Name or MinIO Client.")

        for obj in objects:
            parts = obj.object_name.split('/')
            if len(parts) > 1:
                subfolders.add(parts[0])

        result_list = []
        for subfolder in subfolders:
            if bucket_name == "models":
                result_dict = {
                    "name": subfolder,
                    "base_model": "Unknown",
                    "training_method": "Unknown",
                    "commit_msg": "Unknown",
                    "user_name": "Unknown",
                    "use_deepspeed": False,
                    "use_peft": False,
                    "use_sfttrainer": False,
                    "use_stableadamw": False,
                    "use_flash_attention": False,
                }
            elif bucket_name == "data":
                result_dict = {
                    "name": subfolder,
                    "main_task": "Unknown",
                    "number_rows": "Unknown",
                    "description": "Unknown",
                    "user_name": "Unknown",
                }

            if is_metadata:
                try:
                    if bucket_name == "models":
                        metadata = self.minio_get_metadata(subfolder, bucket_name=bucket_name, local_save_path="/models/minio")
                    elif bucket_name == "data":
                        metadata = self.minio_get_metadata(subfolder, bucket_name=bucket_name, local_save_path="/datasets/minio")
                    result_dict.update(metadata)
                except:
                    print(f"[WARNING] No Metadata File Exist in {subfolder}.")

            result_list.append(result_dict)

        return result_list

    def minio_dataset_info(self, dataset_name: str, bucket_name: str = "data", return_origin: bool = True):
        """
        주어진 버킷 및 데이터 셋 이름으로부터 정보를 찾아냅니다.

        Parameters:
            dataset_name (str): 조회할 데이터셋의 이름. 버킷 내부의 서브폴더여야함.
            bucket_name (str): 조회할 버킷 이름. 기본값은 "data".

        Returns:
            ...

        Raises:
            ValueError:.
        """
        try:
            dict_path = self.minio_data_pull(dataset_name, minio_bucket_name=bucket_name, file_list=[f'{dataset_name}/dataset_dict.json'])
            with open(dict_path[0], 'r') as f:
                dataset_dict = json.load(f)
            split_list = dataset_dict['splits']
        except Exception as e:
            print(e)
            print("[INFO] No Split Information.")
            split_list = None

        if split_list is not None:
            column_dict = {}
            for split in split_list:
                split_dict_path = self.minio_data_pull(dataset_name, minio_bucket_name=bucket_name, file_list=[f'{dataset_name}/{split}/dataset_info.json'])
                with open(split_dict_path[0], 'r') as f:
                    dataset_dict = json.load(f)
                column_list = list(dataset_dict['features'].keys())

                column_dict[split] = column_list

            column_dict = {
                'default': column_dict
            }

        else:
            split_dict_path = self.minio_data_pull(dataset_name, minio_bucket_name=bucket_name, file_list=[f'{dataset_name}/dataset_info.json'])
            with open(split_dict_path[0], 'r') as f:
                dataset_dict = json.load(f)
            column_list = list(dataset_dict['features'].keys())

            column_dict = {
                'default': {
                    'default': column_list,
                }
            }

        if return_origin:
            return column_dict
        else:
            return (split_list, column_dict['default'])


    def minio_dataset_loader(self, dataset_name: str, bucket_name: str = "data", local_save_directory: str = "/datasets/minio", save_only: bool = False):
        """
        MinIO S3에 Push된 Hugging Face Dataset을 로컬로 다운로드하여, Hugging Face Datasets 형식으로 로드합니다.

        데이터셋은 원래 Dataset 혹은 DatasetDict 형태로 Push된 상태여야 하며,
        다운로드 후 local_save_directory에 저장된 후 load_from_disk()로 불러옵니다.

        Parameters:
            dataset_name (str): MinIO에 저장된 데이터셋 이름 (Hugging Face 식별자).
            bucket_name (str): 데이터를 다운로드할 버킷 이름. 기본값은 "data".
            local_save_directory (str): 로컬에 저장할 디렉토리 경로. 기본값은 빈 문자열.
            save_only (bool): True이면 다운로드만 진행하고, 로드하지 않습니다.

        Returns:
            datasets.Dataset 또는 None: save_only가 False인 경우 로드한 Dataset 객체를 반환하며, True이면 None을 반환합니다.
        """
        dataset_name = dataset_name.replace("/", "__")
        local_save_path = f"{local_save_directory}/{dataset_name}"
        self.minio_data_pull(minio_bucket_folder=dataset_name, save_path=local_save_path, minio_bucket_name=bucket_name)

        if save_only:
            return None

        else:
            dataset = datasets.load_from_disk(local_save_path)
            print(f"[INFO] Successfully load dataset.")
            return dataset


    def minio_model_push(self, folder_path: str, minio_folder_name: str, minio_bucket_name: str = "models", object_prefix: str = None) -> None:
        """
        로컬 폴더의 모델 파일들을 MinIO 버킷으로 업로드합니다.

        Parameters:
            folder_path (str): 업로드할 로컬 폴더 경로.
            minio_folder_name (str): MinIO 버킷 내 저장될 폴더 이름.
            minio_bucket_name (str, optional): 파일을 업로드할 버킷 이름. 기본값은 "models".
            object_prefix (str, optional): 각 파일의 상대 경로 앞에 추가할 접두어.

        Returns:
            None

        Note:
            지정한 버킷이 존재하지 않으면 새로 생성합니다.
        """
        if not self.minio_client.bucket_exists(minio_bucket_name):
            self.minio_client.make_bucket(minio_bucket_name)

        for root, _, files in os.walk(folder_path):
            for file_name in files:
                file_path = os.path.join(root, file_name)
                relative_path = os.path.relpath(file_path, folder_path)

                if object_prefix:
                    object_name = f"{minio_folder_name}/{object_prefix}{relative_path}"
                else:
                    object_name = f"{minio_folder_name}/{relative_path}"

                object_name = object_name.replace("\\", "/")

                self.minio_client.fput_object(minio_bucket_name, object_name, file_path)
                print(f"[INFO] Uploaded {file_name} to MinIO as {object_name}")


    def minio_data_pull(self, minio_bucket_folder: str, save_path: str = "/models/minio", minio_bucket_name: str = "models", file_list: list[str] = None, force_download: bool = False) -> None:
        """
        MinIO 버킷에서 지정된 폴더의 모델 파일들을 로컬로 다운로드합니다.

        Parameters:
            minio_bucket_folder (str): 버킷 내에서 다운로드할 폴더 이름.
            save_path (str): 로컬에 파일들을 저장할 경로.
            minio_bucket_name (str, optional): 다운로드할 버킷 이름. 기본값은 "models".
            file_list (list[str], optional): 다운로드할 파일 이름 리스트. 제공하지 않으면, 해당 폴더 내 모든 파일을 다운로드합니다.
            force_download (bool, optional): True일 경우, 로컬에 동일 파일이 있어도 덮어쓰고 다운로드합니다.

        Returns:
            다운로드된 파일들의 로컬 경로 리스트 (file_list 제공 시) 또는 None.
        """
        if (save_path is None) or len(save_path) < 1:
            save_path = "/models/minio"

        os.makedirs(save_path, exist_ok=True)

        if file_list:
            # file_list가 제공되면, 각 파일의 경로(버킷 내 전체 경로 또는 상대 경로)를 그대로 사용합니다.
            file_local_path = []
            for file in file_list:
                local_path = os.path.join(save_path, file)
                os.makedirs(os.path.dirname(local_path), exist_ok=True)
                if not force_download and os.path.exists(local_path):
                    print(f"[INFO] {file} already exists at {local_path}. Skipping download.")
                    file_local_path.append(local_path)

                else:
                    try:
                        self.minio_client.fget_object(minio_bucket_name, file, local_path)
                        print(f"[INFO] Successfully downloaded {file} to {local_path}")
                        file_local_path.append(local_path)
                    except Exception as e:
                        raise ValueError("[ERROR] No File Exist.") from e
            return file_local_path

        else:
            prefix = f"{minio_bucket_folder}/"
            objects = self.minio_client.list_objects(minio_bucket_name, prefix=prefix, recursive=True)
            for obj in objects:
                object_name = obj.object_name

                if object_name.startswith(prefix):
                    relative_object_path = object_name[len(prefix):]
                else:
                    relative_object_path = object_name

                local_path = os.path.join(save_path, relative_object_path)
                os.makedirs(os.path.dirname(local_path), exist_ok=True)

                if not force_download and os.path.exists(local_path):
                    print(f"[INFO] {object_name} already exists at {local_path}. Skipping download.")
                else:
                    self.minio_client.fget_object(minio_bucket_name, object_name, local_path)
                    print(f"[INFO] Successfully downloaded {object_name} to {local_path}")

            print(f"[INFO] Download Done. Local Save Path: {save_path}")

    def hgf_ds_concat_tool(
            self,
            hgf_path: str,
            hgf_data_dir: list[str],
            split: str = 'train',
            push_to: str = None,
            config_name: str = 'default',
            data_dir: str = None
        ):
        """
        여러 데이터 디렉토리에서 Hugging Face Dataset을 로드한 후, 이를 합쳐서 하나의 Dataset으로 만듭니다.
        선택적으로, 합쳐진 Dataset을 Hugging Face Hub에 푸시할 수 있습니다.

        Parameters:
            hgf_path (str): Hugging Face Dataset의 경로 또는 식별자.
            hgf_data_dir (list[str]): 데이터를 로드할 디렉토리 리스트.
            split (str, optional): 로드할 데이터셋의 분할 이름 (기본값 'train').
            push_to (str, optional): 데이터를 푸시할 Hugging Face Hub 리포지토리 이름. None이면 푸시하지 않습니다.
            config_name (str, optional): 푸시할 때 사용할 구성 이름 (기본값 'default').
            data_dir (str, optional): 푸시할 때 사용할 데이터 디렉토리. (필요에 따라 지정)

        Returns:
            datasets.Dataset 또는 None: push_to가 None이면 합쳐진 Dataset을 반환하며, 그렇지 않으면 푸시 후 None을 반환.

        Raises:
            ValueError: 데이터셋 로드 과정에서 오류가 발생한 경우.
        """
        try:
            dataset_list = [
                datasets.load_dataset(hgf_path, data_dir=hgf_dir, split=split, token=PLATEER_HGF_TOKEN)
                for hgf_dir in hgf_data_dir
            ]
            dataset = datasets.concatenate_datasets(dataset_list)
        except Exception as e:
            raise ValueError(f"[ERROR] {e}")

        if push_to is None:
            return dataset
        else:
            create_repo(repo_id=push_to, token=PLATEER_HGF_TOKEN, exist_ok=True)
            dataset.push_to_hub(push_to, config_name=config_name, data_dir=data_dir, token=PLATEER_HGF_TOKEN)
            print(f"[INFO] Successfully Push to '{push_to}'")

    def hgf_ds_to_minio(
            self,
            hgf_path: str,
            hgf_data_dir: Optional[Union[str, list[str]]] = None,
            data_split: Optional[str] = None,
            minio_folder_name: Optional[str] = None,
            minio_bucket_name: str = "data",
            sample_num: int = None,
        ):
        """
        Hugging Face Dataset을 로드하여 로컬에 Arrow 형식으로 저장한 후,
        이를 MinIO S3 버킷에 업로드합니다.
        hgf_data_dir가 단일 문자열인 경우와 리스트인 경우 모두 처리하며,
        업로드 완료 후 로컬에 생성된 임시 데이터셋 디렉토리를 삭제합니다.

        Parameters:
            hgf_path (str): Hugging Face Dataset의 경로 또는 식별자.
            hgf_data_dir (Optional[Union[str, list[str]]]): 데이터를 로드할 디렉토리 경로 또는 경로 리스트.
            data_split (Optional[str]): 로드할 데이터셋의 분할 이름 (기본값 'None').
            minio_folder_name (Optional[str]): MinIO 버킷 내 저장할 폴더 이름. 지정하지 않으면 hgf_path를 변환한 이름을 사용합니다.
            minio_bucket_name (str): 업로드할 MinIO 버킷 이름. 기본값은 "data".

        Returns:
            None

        Note:
            업로드 도중 예외가 발생하면, 로컬에 생성된 임시 데이터셋 디렉토리를 삭제한 후 오류 메시지를 출력합니다.
        """
        if hgf_data_dir is None:
            dataset = datasets.load_dataset(hgf_path, split=data_split, token=PLATEER_HGF_TOKEN)
        elif isinstance(hgf_data_dir, str):
            dataset = datasets.load_dataset(hgf_path, data_dir=hgf_data_dir, split=data_split, token=PLATEER_HGF_TOKEN)
        elif isinstance(hgf_data_dir, list):
            dataset_list = [datasets.load_dataset(hgf_path, data_dir=hgf_dir, split=data_split, token=PLATEER_HGF_TOKEN) for hgf_dir in hgf_data_dir]
            dataset = datasets.concatenate_datasets(dataset_list)
        else:
            raise ValueError("[ERROR] Unvalid type. Should check 'hgf_path'.")

        if sample_num is not None:
            dataset = dataset.shuffle().select(range(sample_num))

        try:
            hgf_repath = hgf_path.replace("/", "__")
            local_path = f"/temp_dataset_{hgf_repath}"
            dataset.save_to_disk(local_path)

            if (minio_folder_name is None) or len(minio_folder_name) < 1:
                minio_folder_name = hgf_repath

            for root, _, files in os.walk(local_path):
                for file_name in files:
                    file_path = os.path.join(root, file_name)
                    relative_path = os.path.relpath(file_path, local_path)

                    object_name = f"{minio_folder_name}/{relative_path}"
                    object_name = object_name.replace("\\", "/")

                    self.minio_client.fput_object(minio_bucket_name, object_name, file_path)
                    print(f"[INFO] Uploaded {file_name} to MinIO as {object_name}")

            shutil.rmtree(local_path)
            print("[INFO] Upload success. Remove local temp dataset.")

        except Exception as e:
            shutil.rmtree(local_path)
            print("[ERROR] Upload Error. Local Dataset has been removed.")
            print("[ERROR] ", e)

    def hgf_model_to_minio(
            self,
            hgf_path: str,
            minio_folder_name: Optional[str] = None,
            minio_bucket_name: str = "models"
        ):
        """
        Parameters:
            hgf_path (str): Hugging Face Dataset의 경로 또는 식별자.
            hgf_data_dir (Optional[Union[str, list[str]]]): 데이터를 로드할 디렉토리 경로 또는 경로 리스트.
            data_split (Optional[str]): 로드할 데이터셋의 분할 이름 (기본값 'None').
            minio_folder_name (Optional[str]): MinIO 버킷 내 저장할 폴더 이름. 지정하지 않으면 hgf_path를 변환한 이름을 사용합니다.
            minio_bucket_name (str): 업로드할 MinIO 버킷 이름. 기본값은 "data".

        Returns:
            None

        Note:
            업로드 도중 예외가 발생하면, 로컬에 생성된 임시 데이터셋 디렉토리를 삭제한 후 오류 메시지를 출력합니다.
        """
        try:
            hgf_repath = hgf_path.replace("/", "__")
            local_path = f"/temp_model_{hgf_repath}"
            huggingface_repo_downloader(hgf_path, local_path, token=self.HUGGING_FACE_TOKEN)

            if (minio_folder_name is None) or len(minio_folder_name) < 1:
                minio_folder_name = hgf_repath

            for root, dirs, files in os.walk(local_path):
                # '.cache' 디렉토리는 탐색하지 않도록 제거
                dirs[:] = [d for d in dirs if d != '.cache']

                for file_name in files:
                    file_path = os.path.join(root, file_name)
                    relative_path = os.path.relpath(file_path, local_path)

                    object_name = f"{minio_folder_name}/{relative_path}"
                    object_name = object_name.replace("\\", "/")

                    self.minio_client.fput_object(minio_bucket_name, object_name, file_path)
                    print(f"[INFO] Uploaded {file_name} to MinIO as {object_name}")

            shutil.rmtree(local_path)
            print("[INFO] Upload success. Remove local temp folder.")

        except Exception as e:
            shutil.rmtree(local_path)
            print("[ERROR] Upload Error. Local folder has been removed.")
            print("[ERROR] ", e)

    def get_hgf_model_list(self, author:str = None):
        """
        Hugging Face Hub에서 지정된 사용자(author)의 모델 리스트를 가져옵니다.

        Parameters:
            author (str, optional): 모델 정보를 조회할 Hugging Face 사용자 ID입니다.
                만약 이 값이 제공되지 않으면, 인스턴스 생성 시 설정한 기본 Hugging Face User ID를 사용합니다.

        Returns:
            list[str]: 해당 사용자와 연관된 Hugging Face 모델들의 고유 식별자(ID) 리스트입니다.

        동작:
            - 주어진 사용자(author)의 모델 정보를 Hugging Face Hub API를 통해 요청합니다.
            - 요청에 성공하면, 반환된 모델 리스트에서 각 모델의 ID를 추출하여 리스트로 반환합니다.
            - 요청 실패 시, 경고 메시지를 출력하고 빈 리스트를 반환합니다.
        """
        api = HfApi(token=self.HUGGING_FACE_TOKEN)
        try:
            if author:
                self.HUGGING_FACE_USER_ID = author
            models = api.list_models(author=self.HUGGING_FACE_USER_ID)
            result = [model.id for model in models]

            print(f"[INFO] SUCCESS: Load Model List from UserID: {self.HUGGING_FACE_USER_ID}")
        except:
            print(f"[WARNING] FAIL: Load Model List from UserID: {self.HUGGING_FACE_USER_ID}")
            print(f"[WARNING] RETURN Empty List")
            result = []

        return result

    def get_hgf_dataset_list(self, author:str = None):
        """
        Hugging Face Hub에서 지정된 사용자(author)의 데이터셋 리스트를 가져옵니다.

        Parameters:
            author (str, optional): 데이터셋 정보를 조회할 Hugging Face 사용자 ID입니다.
                만약 이 값이 제공되지 않으면, 인스턴스 생성 시 설정한 기본 Hugging Face User ID를 사용합니다.

        Returns:
            list[str]: 해당 사용자와 연관된 Hugging Face 데이터셋들의 고유 식별자(ID) 리스트입니다.

        동작:
            - 주어진 사용자(author)의 데이터셋 정보를 Hugging Face Hub API를 통해 요청합니다.
            - 요청에 성공하면, 반환된 데이터셋 리스트에서 각 모델의 ID를 추출하여 리스트로 반환합니다.
            - 요청 실패 시, 경고 메시지를 출력하고 빈 리스트를 반환합니다.
        """
        api = HfApi(token=self.HUGGING_FACE_TOKEN)
        try:
            if author:
                self.HUGGING_FACE_USER_ID = author
            models = api.list_datasets(author=self.HUGGING_FACE_USER_ID)
            print(models)
            result = [model.id for model in models]

            print(f"[INFO] SUCCESS: Load Dataset List from UserID: {self.HUGGING_FACE_USER_ID}")
        except:
            print(f"[WARNING] FAIL: Load Dataset List from UserID: {self.HUGGING_FACE_USER_ID}")
            print(f"[WARNING] RETURN Empty List")
            result = []

        return result

    def get_hgf_dataset_info(self, dataset_path:str, return_all:bool = False) -> tuple[dict, list]:
        """
        Hugging Face Hub에서 특정 데이터셋의 정보를 불러옵니다.

        Parameters:
            dataset_path (str): 데이터셋의 경로 또는 식별자.
            return_all (bool, optional): True인 경우 전체 정보를 반환합니다. False인 경우 간략한 정보(컬럼, 스플릿 정보 등)와 가능한 subset 리스트를 반환합니다.

        Returns:
            tuple[dict, list]: 반환 형식은 (데이터셋 정보 딕셔너리, 가능한 subset 리스트) 입니다.

        Raises:
            RuntimeError: 데이터셋 정보를 불러오는 데 실패한 경우.
        """
        api = HfApi(token=self.HUGGING_FACE_TOKEN)
        try:
            dataset_info = api.dataset_info(dataset_path)
            print(f"[INFO] SUCCESS: Load Dataset Information from: {dataset_path}")
        except:
            print(f"[WARNING] FAIL: Load Dataset Information from: {dataset_path}")
            raise RuntimeError()

        if return_all:
            return dataset_info

        else:
            result = {}
            if type(dataset_info.card_data.dataset_info) == dict:
                possible_subset_list = None
                columns = []
                for column in dataset_info.card_data.dataset_info['features']:
                    col_name = column['name']
                    columns.append(col_name)
                splits = []
                for split in dataset_info.card_data.dataset_info['splits']:
                    split_name = split['name']
                    splits.append(split_name)

                temp_result = {}
                for split in splits:
                    temp_result[split] = columns

                result['default'] = temp_result

            else:
                possible_subset_list = []
                for subset in dataset_info.card_data.dataset_info:
                    subset_name = subset['config_name']
                    possible_subset_list.append(subset_name)

                    columns = []
                    for column in subset['features']:
                        col_name = column['name']
                        columns.append(col_name)

                    splits = []
                    for split in subset['splits']:
                        split_name = split['name']
                        splits.append(split_name)

                    temp_result = {}
                    for split in splits:
                        temp_result[split] = columns

                    result[subset_name] = temp_result

            return result

    def minio_get_metadata(self, model_name:str, bucket_name:str = "models", local_save_path:str = "/models/minio") -> dict:
        """
        MinIO S3에 Push된 'metadata.json' 파일을 다운로드하여,
        해당 파일의 내용을 JSON 형식으로 반환합니다.

        Parameters:
            model_name (str): MinIO에 저장된 데이터셋 이름.
            bucket_name (str): 데이터를 다운로드할 버킷 이름. 기본값은 "models".

        Returns:
            dict: 데이터셋의 메타데이터 정보.

        Raises:
            ValueError: 메타데이터 파일을 찾을 수 없는 경우.
        """

        minio_local_save_path = f"{local_save_path}/{model_name}"
        os.makedirs(minio_local_save_path, exist_ok=True)
        metadata_path = os.path.join(minio_local_save_path, "metadata.json")

        if os.path.exists(metadata_path):
            try:
                print("Load Existing Metadata File")
                print(metadata_path)
                with open(metadata_path, 'r', encoding="utf-8") as f:
                    metadata = json.load(f)
                return metadata
            except Exception as e:
                print("[WARNING] Can't Read Metadata File. It might be broken.")
        else:
            try:
                print(f'{model_name}/metadata.json')
                metadata_path = self.minio_data_pull(model_name, save_path=local_save_path, minio_bucket_name=bucket_name, file_list=[f'{model_name}/metadata.json'], force_download=True)
                with open(metadata_path[0], 'r', encoding="utf-8") as f:
                    metadata = json.load(f)
                return metadata
            except Exception as e:
                print("[WARNING] No Metadata File Exist.")

    def minio_model_upload_metadata(
        self,
        base_model: str,
        training_method: str,
        commit_msg: str,
        user_name: str,
        use_deepspeed:bool = False,
        use_peft:bool = False,
        use_sfttrainer:bool = False,
        use_stableadamw: bool = False,
        use_flash_attention:bool = False,
        model_name:str = None,
        bucket_name:str = "models",
        local_save_path:str = "/models/minio"):
        """
        """

        minio_local_save_path = f"{local_save_path}/{model_name}"
        os.makedirs(minio_local_save_path, exist_ok=True)
        metadata_path = os.path.join(minio_local_save_path, "metadata.json")

        metadata = {
            "base_model": base_model,
            "training_method": training_method,
            "commit_msg": commit_msg,
            "user_name": user_name,
            "use_deepspeed": use_deepspeed,
            "use_peft": use_peft,
            "use_flash_attention": use_flash_attention,
            "use_sfttrainer": use_sfttrainer,
            "use_stableadamw": use_stableadamw,
        }
        with open(metadata_path, "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=4)

        if not self.minio_client.bucket_exists(bucket_name):
            self.minio_client.make_bucket(bucket_name)

        object_name = f"{model_name}/metadata.json"
        object_name = object_name.replace("\\", "/")

        self.minio_client.fput_object(bucket_name, object_name, metadata_path)
        print(f"[INFO] Uploaded 'metadata.json' to MinIO as {object_name}")

    def minio_dataset_upload_metadata(
        self,
        main_task: str,
        number_rows: int,
        description: str,
        user_name: str,
        dataset_name:str = None,
        bucket_name:str = "data",
        local_save_path:str = "/datasets/minio"):
        """
        """

        minio_local_save_path = f"{local_save_path}/{dataset_name}"
        os.makedirs(minio_local_save_path, exist_ok=True)
        metadata_path = os.path.join(minio_local_save_path, "metadata.json")

        metadata = {
            "name": dataset_name,
            "main_task": main_task,
            "number_rows": number_rows,
            "description": description,
            "user_name": user_name,
        }
        with open(metadata_path, "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=4)

        if not self.minio_client.bucket_exists(bucket_name):
            self.minio_client.make_bucket(bucket_name)

        object_name = f"{dataset_name}/metadata.json"
        object_name = object_name.replace("\\", "/")

        self.minio_client.fput_object(bucket_name, object_name, metadata_path)
        print(f"[INFO] Uploaded 'metadata.json' to MinIO as {object_name}")

if __name__ == '__main__':
    from env_config import get_huggingface_token, get_huggingface_user_id, get_minio_config

    print("TEST")
    minio_config = get_minio_config()
    toolkit = DataLoader(
        get_huggingface_user_id(),
        get_huggingface_token(),
        minio_config["url"],
        minio_config["access_key"],
        minio_config["secret_key"])

    result = toolkit.get_hgf_dataset_list(author=get_huggingface_user_id())
    print(result)

    # from huggingface_hub import hf_hub_download, snapshot_download
    # def huggingface_repo_downloader(repo_id: str, save_dir: str = "./local_save", token: str = None) -> None:
    #     """
    #     Hugging Face 레포지토리의 모든 파일을 로컬로 다운로드합니다.

    #     Parameters:
    #         repo_id (str): 다운로드할 Hugging Face 레포의 식별자 (예: "username/repo_name").
    #         save_dir (str): 파일들을 저장할 로컬 디렉토리 경로.
    #         token (str, optional): 프라이빗 레포 접근 시 필요한 인증 토큰.

    #     Returns:
    #         None
    #     """
    #     snapshot_download(repo_id, local_dir=save_dir, token=token)
    #     print(f"[INFO] {repo_id} 레포의 모든 파일이 {save_dir}에 다운로드 되었습니다.")

    # toolkit.hgf_model_to_minio("CocoRoF/POLAR_gemma_1b", minio_folder_name="POLAR_gemma_1b")
    # toolkit.hgf_model_to_minio("Qwen/Qwen2.5-0.5B-Instruct")
