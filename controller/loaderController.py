from fastapi import APIRouter, HTTPException, Depends, Query
from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any, Union
import os
from trainer.utils.dataloader_toolkit import DataLoader
from trainer.utils.env_config import get_huggingface_token, get_huggingface_user_id, get_minio_config
from controller.huggingface_default import default_model_list

# Helper functions for default values
def _get_default_hf_user() -> str:
    return get_huggingface_user_id()

def _get_default_hf_token() -> str:
    return get_huggingface_token()

def _get_default_minio_url() -> str:
    return get_minio_config()["url"]

def _get_default_minio_access_key() -> str:
    return get_minio_config()["access_key"]

def _get_default_minio_secret_key() -> str:
    return get_minio_config()["secret_key"]
# 라우터 생성
router = APIRouter(
    prefix="/loader",
    tags=["dataloader"],
    responses={404: {"description": "Not found"}},
)

# 기본 설정 모델
class LoaderConfig(BaseModel):
    model_config = {"protected_namespaces": ()}

    hugging_face_user_id: str = Field(default_factory=get_huggingface_user_id, description="HuggingFace ID")
    hugging_face_token: str = Field(default_factory=get_huggingface_token, description="HuggingFace Token")
    minio_url: str = Field(default_factory=lambda: get_minio_config()["url"], description="MinIO URL")
    minio_access_key: str = Field(default_factory=lambda: get_minio_config()["access_key"], description="MinIO Access Key")
    minio_secret_key: str = Field(default_factory=lambda: get_minio_config()["secret_key"], description="MinIO Secret Key")

# MinIO 관련 모델
class MinioSubfoldersRequest(LoaderConfig):
    bucket_name: str = Field("data", description="MinIO bucket name")

class MinioMetadataRequest(LoaderConfig):
    model_name: str = Field(..., description="Model name in MinIO")
    bucket_name: str = Field("models", description="MinIO bucket name")

class MinioModelUploadMetadata(LoaderConfig):
    base_model: str = Field(..., description="Base model name")
    training_method: str = Field(..., description="Training method")
    commit_msg: str = Field(..., description="Commit message")
    user_name: str = Field(..., description="User name")
    use_deepspeed: bool = Field(False, description="Whether to use DeepSpeed")
    use_peft: bool = Field(False, description="Whether to use PEFT")
    use_sfttrainer: bool = Field(False, description="Whether to use SFTTrainer")
    use_stableadamw: bool = Field(False, description="Whether to use StableAdamW")
    use_flash_attention: bool = Field(False, description="Whether to use Flash Attention")
    model_name: str = Field("", description="Model name in MinIO")
    bucket_name: str = Field("models", description="MinIO bucket name")

class MinioDatasetInfoRequest(LoaderConfig):
    dataset_name: str = Field(..., description="Dataset name in MinIO")
    bucket_name: str = Field("data", description="MinIO bucket name")
    return_origin: bool = Field(True, description="Whether to return original data")

class MinioDatasetUploadMetadata(LoaderConfig):
    main_task: str = Field(..., description="Main task")
    number_rows: int = Field(..., description="Number of rows")
    description: str = Field(..., description="Description")
    user_name: str = Field(..., description="User name")
    dataset_name: str = Field(..., description="Dataset name in MinIO")
    bucket_name: str = Field("data", description="MinIO bucket name")

class MinioDatasetLoaderRequest(LoaderConfig):
    dataset_name: str = Field(..., description="Dataset name in MinIO")
    bucket_name: str = Field("data", description="MinIO bucket name")
    local_save_directory: str = Field("", description="Local directory to save dataset")
    local_dir_prefix: str = Field("dataset", description="Prefix for local directory")
    save_only: bool = Field(False, description="Whether to only save without loading")

class MinioModelPushRequest(LoaderConfig):
    folder_path: str = Field(..., description="Local folder path containing model files")
    minio_folder_name: str = Field(..., description="Folder name in MinIO bucket")
    minio_bucket_name: str = Field("models", description="MinIO bucket name")
    object_prefix: Optional[str] = Field(None, description="Prefix for object names")

class MinioDataPullRequest(LoaderConfig):
    minio_bucket_folder: str = Field(..., description="Folder name in MinIO bucket")
    save_path: str = Field(..., description="Local path to save files")
    minio_bucket_name: str = Field("models", description="MinIO bucket name")
    file_list: Optional[List[str]] = Field(None, description="List of files to download")

# Hugging Face 관련 모델
class HFModelListRequest(LoaderConfig):
    author: Optional[str] = Field(None, description="Hugging Face author ID")

class HFModelToMinioRequest(LoaderConfig):
    hgf_path: str = Field(..., description="Hugging Face model path")
    minio_folder_name: Optional[str] = Field(None, description="Folder name in MinIO")
    minio_bucket_name: str = Field("models", description="MinIO bucket name")

class HFDatasetListRequest(LoaderConfig):
    author: Optional[str] = Field(None, description="Hugging Face author ID")

class HFDatasetInfoRequest(LoaderConfig):
    dataset_path: str = Field(..., description="Dataset path or identifier")
    return_all: bool = Field(False, description="Whether to return all information")

class HFDSConcatToolRequest(LoaderConfig):
    hgf_path: str = Field(..., description="Hugging Face dataset path")
    hgf_data_dir: List[str] = Field(..., description="List of data directories")
    split: str = Field("train", description="Dataset split")
    push_to: Optional[str] = Field(None, description="Repository to push to")
    config_name: str = Field("default", description="Configuration name")
    data_dir: Optional[str] = Field(None, description="Data directory for push")

class HFDSToMinioRequest(LoaderConfig):
    hgf_path: str = Field(..., description="Hugging Face dataset path")
    hgf_data_dir: Optional[Union[str, List[str]]] = Field(None, description="Data directory or list of directories")
    data_split: Optional[str] = Field(None, description="Dataset split")
    minio_folder_name: Optional[str] = Field(None, description="Folder name in MinIO")
    minio_bucket_name: str = Field("data", description="MinIO bucket name")

# 응답 모델
class SuccessResponse(BaseModel):
    status: str = "success"
    message: str
    data: Optional[Any] = None

# DataLoader 인스턴스 생성 함수
def create_dataloader(
    hugging_face_user_id: str = None,
    hugging_face_token: str = None,
    minio_url: str = None,
    minio_access_key: str = None,
    minio_secret_key: str = None
) -> DataLoader:
    # Use environment variables if parameters are not provided
    if hugging_face_user_id is None:
        hugging_face_user_id = get_huggingface_user_id()
    if hugging_face_token is None:
        hugging_face_token = get_huggingface_token()

    minio_config = get_minio_config()
    if minio_url is None:
        minio_url = minio_config["url"]
    if minio_access_key is None:
        minio_access_key = minio_config["access_key"]
    if minio_secret_key is None:
        minio_secret_key = minio_config["secret_key"]

    return DataLoader(
        hugging_face_user_id=hugging_face_user_id,
        hugging_face_token=hugging_face_token,
        minio_url=minio_url,
        minio_access_key=minio_access_key,
        minio_secret_key=minio_secret_key
    )

# MinIO 관련 엔드포인트
@router.get("/minio/subfolders", response_model=SuccessResponse)
async def get_minio_subfolders(
    bucket_name: str = Query("data", description="MinIO bucket name"),
    hugging_face_user_id: str = Query(default=None, description="HuggingFace ID"),
    hugging_face_token: str = Query(default=None, description="HuggingFace Token"),
    minio_url: str = Query(default=None, description="MinIO URL"),
    minio_access_key: str = Query(default=None, description="MinIO Access Key"),
    minio_secret_key: str = Query(default=None, description="MinIO Secret Key")
):
    """
    MinIO 버킷의 최상위 폴더 목록을 반환합니다.
    """
    try:
        dataloader = create_dataloader(
            hugging_face_user_id=hugging_face_user_id,
            hugging_face_token=hugging_face_token,
            minio_url=minio_url,
            minio_access_key=minio_access_key,
            minio_secret_key=minio_secret_key
        )
        subfolders = dataloader.minio_get_subfolders(bucket_name=bucket_name)
        return {
            "status": "success",
            "message": f"Successfully retrieved subfolders from bucket {bucket_name}",
            "data": subfolders
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/minio/dataset/info", response_model=SuccessResponse)
async def minio_dataset_info(params: MinioDatasetInfoRequest):
    """
    MinIO에서 데이터셋 정보를 가져옵니다.
    """
    try:
        dataloader = create_dataloader(
            hugging_face_user_id=params.hugging_face_user_id,
            hugging_face_token=params.hugging_face_token,
            minio_url=params.minio_url,
            minio_access_key=params.minio_access_key,
            minio_secret_key=params.minio_secret_key
        )
        info = dataloader.minio_dataset_info(
            dataset_name=params.dataset_name,
            bucket_name=params.bucket_name,
            return_origin=params.return_origin
        )

        return {
            "status": "success",
            "message": f"Successfully retrieved dataset info for {params.dataset_name}",
            "data": info
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/minio/dataset/info", response_model=SuccessResponse)
async def get_minio_dataset_info(
    dataset_name: str = Query(..., description="Dataset name in MinIO"),
    bucket_name: str = Query("data", description="MinIO bucket name"),
    return_origin: bool = Query(True, description="Whether to return original data"),
    hugging_face_user_id: str = Query(default=None, description="HuggingFace ID"),
    hugging_face_token: str = Query(default=None, description="HuggingFace Token"),
    minio_url: str = Query(default=None, description="MinIO URL"),
    minio_access_key: str = Query(default=None, description="MinIO Access Key"),
    minio_secret_key: str = Query(default=None, description="MinIO Secret Key")
):
    """
    MinIO에서 데이터셋 정보를 가져옵니다.
    """
    try:
        dataloader = create_dataloader(
            hugging_face_user_id=hugging_face_user_id,
            hugging_face_token=hugging_face_token,
            minio_url=minio_url,
            minio_access_key=minio_access_key,
            minio_secret_key=minio_secret_key
        )
        info = dataloader.minio_dataset_info(
            dataset_name=dataset_name,
            bucket_name=bucket_name,
            return_origin=return_origin
        )

        return {
            "status": "success",
            "message": f"Successfully retrieved dataset info for {dataset_name}",
            "data": info
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/minio/dataset/load", response_model=SuccessResponse)
async def minio_dataset_load(params: MinioDatasetLoaderRequest):
    """
    MinIO에서 데이터셋을 로드합니다.
    """
    try:
        dataloader = create_dataloader(params)
        dataset = dataloader.minio_dataset_loader(
            dataset_name=params.dataset_name,
            bucket_name=params.bucket_name,
            local_save_directory="/datasets/minio" if params.local_save_directory == "" else params.local_save_directory,
            local_dir_prefix=params.local_dir_prefix,
            save_only=params.save_only
        )

        return {
            "status": "success",
            "message": f"Successfully loaded dataset {params.dataset_name}",
            "data": {
                "dataset_info": str(dataset) if dataset else "Dataset saved only",
                "local_path": f"{params.local_save_directory}{params.local_dir_prefix}_{params.dataset_name}"
            }
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/minio/model/push", response_model=SuccessResponse)
async def minio_model_push(params: MinioModelPushRequest):
    """
    로컬 폴더의 모델 파일들을 MinIO로 업로드합니다.
    """
    try:
        dataloader = create_dataloader(params)
        dataloader.minio_model_push(
            folder_path=params.folder_path,
            minio_folder_name=params.minio_folder_name,
            minio_bucket_name=params.minio_bucket_name,
            object_prefix=params.object_prefix
        )

        return {
            "status": "success",
            "message": f"Successfully pushed model from {params.folder_path} to MinIO bucket {params.minio_bucket_name}/{params.minio_folder_name}"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/minio/model/metadata/get", response_model=SuccessResponse)
async def get_minio_model_metadata(
    model_name: str = Query(..., description="Model name in MinIO"),
    bucket_name: str = Query("models", description="MinIO bucket name"),
    hugging_face_user_id: str = Query(default=None, description="HuggingFace ID"),
    hugging_face_token: str = Query(default=None, description="HuggingFace Token"),
    minio_url: str = Query(default=None, description="MinIO URL"),
    minio_access_key: str = Query(default=None, description="MinIO Access Key"),
    minio_secret_key: str = Query(default=None, description="MinIO Secret Key")
):
    """
    MinIO 모델의 Metadata를 로드하여 보여줍니다.
    """
    try:
        dataloader = create_dataloader(
            hugging_face_user_id=hugging_face_user_id,
            hugging_face_token=hugging_face_token,
            minio_url=minio_url,
            minio_access_key=minio_access_key,
            minio_secret_key=minio_secret_key
        )
        metadata = dataloader.minio_get_metadata(model_name=model_name, bucket_name=bucket_name)
        return {
            "status": "success",
            "message": f"Successfully retrieved metadata from bucket {bucket_name}",
            "data": metadata
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/minio/model/metadata/push", response_model=SuccessResponse)
async def minio_model_push(params: MinioModelUploadMetadata):
    """
    """
    try:
        dataloader = create_dataloader(params)
        dataloader.minio_model_upload_metadata(
            base_model=params.base_model,
            training_method=params.training_method,
            commit_msg=params.commit_msg,
            user_name=params.user_name,
            use_deepspeed=params.use_deepspeed,
            use_peft=params.use_peft,
            use_sfttrainer=params.use_sfttrainer,
            use_stableadamw=params.use_stableadamw,
            use_flash_attention=params.use_flash_attention,
            model_name=params.model_name,
            bucket_name=params.bucket_name,
            local_save_path="models/minio"
        )

        return {
            "status": "success",
            "message": f"Successfully upload metadata for model {params.model_name}"
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/minio/data/pull", response_model=SuccessResponse)
async def minio_data_pull(params: MinioDataPullRequest):
    """
    MinIO에서 데이터를 로컬로 다운로드합니다.
    """
    try:
        dataloader = create_dataloader(params)
        dataloader.minio_data_pull(
            minio_bucket_folder=params.minio_bucket_folder,
            save_path="/models/minio",
            minio_bucket_name=params.minio_bucket_name,
            file_list=params.file_list,
        )

        return {
            "status": "success",
            "message": f"Successfully pulled data from MinIO bucket {params.minio_bucket_name}/{params.minio_bucket_folder} to {params.save_path}"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/minio/dataset/metadata/push", response_model=SuccessResponse)
async def minio_model_push(params: MinioDatasetUploadMetadata):
    """
    """
    try:
        dataloader = create_dataloader(params)
        dataloader.minio_dataset_upload_metadata(
            main_task=params.main_task,
            number_rows=params.number_rows,
            description=params.description,
            user_name=params.user_name,
            dataset_name=params.dataset_name,
            bucket_name=params.bucket_name,
        )

        return {
            "status": "success",
            "message": f"Successfully upload metadata for model {params.dataset_name}"
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/minio/dataset/metadata/get", response_model=SuccessResponse)
async def get_minio_model_metadata(
    dataset_name: str = Query(..., description="Model name in MinIO"),
    bucket_name: str = Query("data", description="MinIO bucket name"),
    hugging_face_user_id: str = Query(default=None, description="HuggingFace ID"),
    hugging_face_token: str = Query(default=None, description="HuggingFace Token"),
    minio_url: str = Query(default=None, description="MinIO URL"),
    minio_access_key: str = Query(default=None, description="MinIO Access Key"),
    minio_secret_key: str = Query(default=None, description="MinIO Secret Key")
):
    """
    MinIO 모델의 Metadata를 로드하여 보여줍니다.
    """
    try:
        dataloader = create_dataloader(
            hugging_face_user_id=hugging_face_user_id,
            hugging_face_token=hugging_face_token,
            minio_url=minio_url,
            minio_access_key=minio_access_key,
            minio_secret_key=minio_secret_key
        )
        metadata = dataloader.minio_get_metadata(model_name=dataset_name, bucket_name=bucket_name)
        return {
            "status": "success",
            "message": f"Successfully retrieved metadata from bucket {bucket_name}",
            "data": metadata
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Hugging Face 관련 엔드포인트
@router.get("/hf/models", response_model=SuccessResponse)
async def get_hf_models(
    author: Optional[str] = Query(None, description="Hugging Face author ID"),
    hugging_face_user_id: str = Query(default=None, description="HuggingFace ID"),
    hugging_face_token: str = Query(default=None, description="HuggingFace Token"),
    minio_url: str = Query(default=None, description="MinIO URL"),
    minio_access_key: str = Query(default=None, description="MinIO Access Key"),
    minio_secret_key: str = Query(default=None, description="MinIO Secret Key")
):
    """
    Hugging Face Hub에서 모델 목록을 가져옵니다.
    """
    try:
        try:
            print("Try to fetch all models from 'author'")
            dataloader = create_dataloader(
                hugging_face_user_id=hugging_face_user_id,
                hugging_face_token=hugging_face_token,
                minio_url=minio_url,
                minio_access_key=minio_access_key,
                minio_secret_key=minio_secret_key
            )
            models = dataloader.get_hgf_model_list(author=author)

            response = []
            for model in models:
                item = {
                    "name": model,
                    "user_name": author
                }
                response.append(item)

            default_model_list.extend(response)

            return {
                "status": "success",
                "message": f"Successfully retrieved models from Hugging Face Hub",
                "data": default_model_list
            }

        except Exception as e:
            print("Fail to fetc models from 'author'. Only return base models.")
            print(e)
            return {
                "status": "success",
                "message": f"Successfully retrieved models from Hugging Face Hub",
                "data": default_model_list
            }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/hf/models/to-minio", response_model=SuccessResponse)
async def get_hf_models(params: HFModelToMinioRequest):
    """
    Hugging Face 모델을 minio로 migration 합니다.
    """
    try:
        dataloader = create_dataloader(params)
        dataloader.hgf_model_to_minio(
            hgf_path=params.hgf_path,
            minio_folder_name=params.minio_folder_name,
            minio_bucket_name=params.minio_bucket_name
        )

        return {
            "status": "success",
            "message": f"Successfully uploaded dataset {params.hgf_path} to MinIO bucket {params.minio_bucket_name}",
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/hf/datasets", response_model=SuccessResponse)
async def get_hf_datasets(
    author: Optional[str] = Query(None, description="Hugging Face author ID"),
    hugging_face_user_id: str = Query(default=None, description="HuggingFace ID"),
    hugging_face_token: str = Query(default=None, description="HuggingFace Token"),
    minio_url: str = Query(default=None, description="MinIO URL"),
    minio_access_key: str = Query(default=None, description="MinIO Access Key"),
    minio_secret_key: str = Query(default=None, description="MinIO Secret Key")
):
    """
    Hugging Face Hub에서 데이터셋 목록을 가져옵니다.
    """
    try:
        dataloader = create_dataloader(
            hugging_face_user_id=hugging_face_user_id,
            hugging_face_token=hugging_face_token,
            minio_url=minio_url,
            minio_access_key=minio_access_key,
            minio_secret_key=minio_secret_key
        )
        datasets = dataloader.get_hgf_dataset_list(author=author)
        response = []
        for data in datasets:
            item = {
                "name": data,
                "user_name": author
            }
            response.append(item)

        return {
            "status": "success",
            "message": f"Successfully retrieved datasets from Hugging Face Hub",
            "data": response
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/hf/dataset/info", response_model=SuccessResponse)
async def get_hf_dataset_info(
    dataset_path: str = Query(..., description="Dataset path or identifier"),
    return_all: bool = Query(False, description="Whether to return all information"),
    hugging_face_user_id: str = Query(default=None, description="HuggingFace ID"),
    hugging_face_token: str = Query(default=None, description="HuggingFace Token"),
    minio_url: str = Query(default=None, description="MinIO URL"),
    minio_access_key: str = Query(default=None, description="MinIO Access Key"),
    minio_secret_key: str = Query(default=None, description="MinIO Secret Key")
):
    """
    Hugging Face Hub에서 특정 데이터셋의 정보를 가져옵니다.
    """
    try:
        dataloader = create_dataloader(
            hugging_face_user_id=hugging_face_user_id,
            hugging_face_token=hugging_face_token,
            minio_url=minio_url,
            minio_access_key=minio_access_key,
            minio_secret_key=minio_secret_key
        )
        info = dataloader.get_hgf_dataset_info(
            dataset_path=dataset_path,
            return_all=return_all
        )

        return {
            "status": "success",
            "message": f"Successfully retrieved dataset info for {dataset_path}",
            "data": info
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/hf/dataset/concat", response_model=SuccessResponse)
async def hf_dataset_concat(params: HFDSConcatToolRequest):
    """
    여러 Hugging Face 데이터셋을 합쳐서 하나의 데이터셋으로 만듭니다.
    """
    try:
        dataloader = create_dataloader(params)
        result = dataloader.hgf_ds_concat_tool(
            hgf_path=params.hgf_path,
            hgf_data_dir=params.hgf_data_dir,
            split=params.split,
            push_to=params.push_to,
            config_name=params.config_name,
            data_dir=params.data_dir
        )

        return {
            "status": "success",
            "message": f"Successfully concatenated datasets from {params.hgf_path}",
            "data": {
                "result": str(result) if result else "Dataset pushed to hub",
                "push_to": params.push_to
            }
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/hf/dataset/to-minio", response_model=SuccessResponse)
async def hf_dataset_to_minio(params: HFDSToMinioRequest):
    """
    Hugging Face 데이터셋을 MinIO로 업로드합니다.
    """
    try:
        dataloader = create_dataloader(params)
        dataloader.hgf_ds_to_minio(
            hgf_path=params.hgf_path,
            hgf_data_dir=params.hgf_data_dir,
            data_split=params.data_split,
            minio_folder_name=params.minio_folder_name,
            minio_bucket_name=params.minio_bucket_name
        )

        return {
            "status": "success",
            "message": f"Successfully uploaded dataset {params.hgf_path} to MinIO bucket {params.minio_bucket_name}"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
