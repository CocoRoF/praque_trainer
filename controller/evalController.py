from fastapi import APIRouter, HTTPException, Query, BackgroundTasks
from pydantic import BaseModel, Field
from typing import Dict, Optional, Any, List
from eval.evaluator import evaluator_bert, evaluator_LM, HfCheck, evaluator_LM_task # 평가 함수 임포트
import logging
import datetime, json, os
import threading
import uuid

# 로깅 설정
logger = logging.getLogger("polar-evaluator")

# 평가 작업 데이터를 저장할 디렉터리
EVAL_JOB_DATA_DIR = "eval_job_data"

# 디렉터리가 없으면 생성
os.makedirs(EVAL_JOB_DATA_DIR, exist_ok=True)

router = APIRouter(
    prefix="/eval",
    tags=["evaluation"],
    responses={404: {"description": "Not found"}},
)

# 요청 모델 정의
class EvalRequest(BaseModel):
    job_name: str = Field(..., description="job_name")
    task: str = Field(..., description="작업 타입")
    model_name: str = Field(..., description="Minio 모델 폴더 명 또는 허깅페이스 주소")
    dataset_name: str = Field(..., description="Minio 데이터셋 폴더 명 또는 허깅페이스 주소")
    column1: Optional[str] = Field(None, description="독립변수 1")  # 필수가 아니라면 None
    column2: Optional[str] = Field(None, description="독립변수 2")
    column3: Optional[str] = Field(None, description="독립변수 3")
    label: Optional[str] = Field(None, description="종속 변수")
    top_k: Optional[int] = Field(1, description="상위 k 값")
    gpu_num: Optional[int] = Field(1, description="사용 GPU 수")
    model_minio_enabled: bool = Field(..., description="Minio 사용 시 True, 아니면 False")
    dataset_minio_enabled: bool = Field(..., description="Minio 사용 시 True, 아니면 False")  # Optional 제거
    use_cot: Optional[bool] = Field(False, description="LLM 평가 시 CoT 사용 여부")
    base_model: Optional[str] = Field(None, description="Minio 모델 폴더 명 또는 허깅페이스 주소")  # 수정: default None

# 응답 모델 정의
class EvalResponse(BaseModel):
    job_id: str
    status: str
    message: str

# 평가 상태 응답 모델
class EvalStatusResponse(BaseModel):
    job_id: str
    status: str
    start_time: Optional[str] = None
    end_time: Optional[str] = None
    job_info: Dict[str, Any]
    result: Optional[Dict[str, Any]]
    base_model_result: Optional[Dict[str, Any]] = None
    base_model_name: Optional[str] = None

# 실행 중인 평가 작업을 추적하기 위한 딕셔너리
evaluation_jobs = {}
# 동시성 제어를 위한 락
jobs_lock = threading.Lock()

def save_eval_job_to_json(job_id: str, job_data: Dict[str, Any]) -> None:
    """
    평가 작업 데이터를 JSON 파일로 저장합니다.
    """
    file_path = os.path.join(EVAL_JOB_DATA_DIR, f"{job_id}.json")
    with open(file_path, 'w', encoding='utf-8') as f:
        json.dump(job_data, f, ensure_ascii=False, indent=4)
    logger.info(f"Evaluation job data saved to {file_path}")

def load_eval_job_from_json(job_id: str) -> Dict[str, Any]:
    """
    JSON 파일에서 평가 작업 데이터를 로드합니다.
    """
    file_path = os.path.join(EVAL_JOB_DATA_DIR, f"{job_id}.json")
    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail="Evaluation job not found")
    
    with open(file_path, 'r', encoding='utf-8') as f:
        return json.load(f)

def load_all_eval_jobs_from_json() -> Dict[str, Dict[str, Any]]:
    """
    모든 평가 작업 데이터를 JSON 파일에서 로드합니다.
    """
    jobs = {}
    job_files = os.listdir(EVAL_JOB_DATA_DIR)
    
    for file_name in job_files:
        if file_name.endswith('.json'):
            try:
                job_id = file_name.replace('.json', '')
                file_path = os.path.join(EVAL_JOB_DATA_DIR, file_name)
                with open(file_path, 'r', encoding='utf-8') as f:
                    jobs[job_id] = json.load(f)
            except Exception as e:
                logger.error(f"Error loading job data from {file_name}: {str(e)}")
    
    return jobs

def run_evaluation_thread(request: EvalRequest, job_id: str):
    """
    평가 작업을 실행하는 스레드 함수
    """
    # 로그 파일 경로 설정 (기존 EVAL_JOB_DATA_DIR 사용)
    log_filename = os.path.join(EVAL_JOB_DATA_DIR, f"{job_id}.json")
    
    # 결과 변수들 초기화
    final_result = None
    base_model_result = None
    base_model_name = None
    
    # 요청 시작 시, job_info와 빈 logs 배열을 기록하여 초기 JSON 파일 생성
    initial_data = {
        "job_info": request.model_dump(), 
        "logs": [],
        "status": "running",
        "start_time": datetime.datetime.now().isoformat()
    }
    
    with open(log_filename, "w", encoding="utf-8") as f:
        json.dump(initial_data, f, ensure_ascii=False, indent=4)
    
    # 헬퍼 함수: 파일에 새 로그 항목을 추가
    def update_log_file(log_entry):
        try:
            with open(log_filename, "r+", encoding="utf-8") as f:
                data = json.load(f)
                # logs 키가 없는 경우 초기화
                if "logs" not in data:
                    data["logs"] = []
                data["logs"].append(log_entry)
                f.seek(0)
                json.dump(data, f, ensure_ascii=False, indent=4)
                f.truncate()
        except Exception as e:
            # 로그 파일 업데이트 실패 시 무한 루프 방지
            print(f"Failed to update log file: {e}")
    
    # 커스텀 로그 핸들러: 로그 발생 시 update_log_file() 호출
    class CaptureHandler(logging.Handler):
        def emit(self, record):
            try:
                timestamp = datetime.datetime.fromtimestamp(record.created).isoformat()
                log_entry = {
                    "timestamp": timestamp,
                    "level": record.levelname,
                    "message": record.getMessage()
                }
                update_log_file(log_entry)
            except Exception:
                # 로그 핸들러에서 예외 발생 시 무시 (무한 루프 방지)
                pass
    
    capture_handler = CaptureHandler()
    capture_handler.setLevel(logging.DEBUG)
    root_logger = logging.getLogger()
    root_logger.addHandler(capture_handler)
    
    try:
        # evaluation_jobs와 JSON 파일 모두 업데이트
        with jobs_lock:
            job_data = {
                "status": "running",
                "start_time": datetime.datetime.now().isoformat(),
                "job_info": request.model_dump()
            }
            evaluation_jobs[job_id] = job_data
            
            # JSON 파일 업데이트
            with open(log_filename, "r+", encoding="utf-8") as f:
                data = json.load(f)
                data.update(job_data)
                f.seek(0)
                json.dump(data, f, ensure_ascii=False, indent=4)
                f.truncate()
        
        logger.info(f"Starting evaluation job {job_id}")
        logger.info(f"Parameters: {request.model_dump()}")
        
        # 평가 로직 실행
        if request.task == 'CausalLM':
            logger.info(f"Running CausalLM evaluation with model {request.model_name} on dataset {request.dataset_name}")
            result = evaluator_LM(
                model_name=request.model_name,
                dataset_name=request.dataset_name,
                gpu_count=request.gpu_num,
                use_cot=request.use_cot,
                column1=request.column1,
                column2=request.column2,
                column3=request.column3,
                label=request.label,
                model_minio_enabled=request.model_minio_enabled,
                dataset_minio_enabled=request.dataset_minio_enabled
            )
            final_result = result
            logger.info(f"CausalLM evaluation completed with result: {result}")
            
        elif request.task == "CausalLM_task":
            logger.info(f"Running CausalLM_task evaluation with model {request.model_name}")
            
            # base_model이 있는지 확인
            has_base_model = request.base_model is not None and request.base_model.strip() != ""
            
            if has_base_model:
                logger.info(f"Base model specified: {request.base_model}")
                base_model_name = request.base_model
                
                # Main model 평가
                logger.info("Starting main model evaluation...")
                main_result, base_result = evaluator_LM_task(
                    job_name=job_id,
                    model_name=request.model_name,
                    dataset_name=request.dataset_name,
                    gpu_count=request.gpu_num,
                    model_minio_enabled=request.model_minio_enabled,
                    base_model = request.base_model
                )
                
                # Base model 평가
                final_result = main_result
                base_model_result = base_result
                
                logger.info(f"Main model evaluation completed")
                logger.info(f"Base model evaluation completed")
                
            else:
                # Base model 없이 단일 모델 평가
                logger.info("Running single model evaluation (no base model)")
                result, _ = evaluator_LM_task(
                    job_name=job_id,
                    model_name=request.model_name,
                    dataset_name=request.dataset_name,
                    gpu_count=request.gpu_num,
                    model_minio_enabled=request.model_minio_enabled,
                )
                final_result = result
                logger.info(f"Single model evaluation completed")
            
            # Main model 결과 에러 체크
            if final_result and 'error' in final_result:
                error_message = final_result.get('error', 'Unknown error occurred in main model')
                logger.error(f"Main model evaluation failed: {error_message}")
                raise Exception(error_message)
            
            # Base model 결과 에러 체크 (있는 경우)
            if base_model_result and 'error' in base_model_result:
                error_message = base_model_result.get('error', 'Unknown error occurred in base model')
                logger.error(f"Base model evaluation failed: {error_message}")
                # Base model 실패는 경고로 처리하고 계속 진행
                logger.warning(f"Base model evaluation failed but continuing with main model results")
                base_model_result = {"error": error_message}
                
        else:
            logger.info(f"Running {request.task} evaluation with model {request.model_name} on dataset {request.dataset_name}")
            result = evaluator_bert(
                task=request.task,
                model_name=request.model_name,
                dataset_name=request.dataset_name,
                column1=request.column1,
                column2=request.column2,
                column3=request.column3,
                label=request.label,
                top_k=request.top_k,
                gpu_num=request.gpu_num,
                model_minio_enabled=request.model_minio_enabled,
                dataset_minio_enabled=request.dataset_minio_enabled
            )
            final_result = result
            logger.info(f"{request.task} evaluation completed with result: {result}")
        
        # 평가 작업 완료 후 상태 업데이트
        with jobs_lock:
            evaluation_jobs[job_id]["status"] = "completed"
            evaluation_jobs[job_id]["end_time"] = datetime.datetime.now().isoformat()
            evaluation_jobs[job_id]["result"] = final_result
            
            # Base model 결과가 있으면 추가
            if base_model_result is not None:
                evaluation_jobs[job_id]["base_model_result"] = base_model_result
                evaluation_jobs[job_id]["base_model_name"] = base_model_name
            
            # JSON 파일도 업데이트
            with open(log_filename, "r+", encoding="utf-8") as f:
                data = json.load(f)
                data["status"] = "completed"
                data["end_time"] = datetime.datetime.now().isoformat()
                data["result"] = final_result
                
                # Base model 관련 정보가 있으면 추가
                if base_model_result is not None:
                    data["base_model_result"] = base_model_result
                    data["base_model_name"] = base_model_name
                
                f.seek(0)
                json.dump(data, f, ensure_ascii=False, indent=4)
                f.truncate()
        
        logger.info(f"Evaluation job {job_id} completed successfully")
        return final_result
    
    except Exception as e:
        error_message = f"Evaluation job {job_id} failed: {str(e)}"
        logger.error(error_message)
        
        # 실패 상태 업데이트
        with jobs_lock:
            evaluation_jobs[job_id]["status"] = "failed"
            evaluation_jobs[job_id]["error"] = str(e)
            evaluation_jobs[job_id]["end_time"] = datetime.datetime.now().isoformat()
            
            # JSON 파일도 업데이트
            try:
                with open(log_filename, "r+", encoding="utf-8") as f:
                    data = json.load(f)
                    data["status"] = "failed"
                    data["error"] = str(e)
                    data["end_time"] = datetime.datetime.now().isoformat()
                    f.seek(0)
                    json.dump(data, f, ensure_ascii=False, indent=4)
                    f.truncate()
            except Exception:
                pass  # JSON 파일 업데이트 실패해도 계속 진행
        
        raise HTTPException(status_code=500, detail=error_message)
    
    finally:
        root_logger.removeHandler(capture_handler)
        capture_handler.close()

def run_evaluation(request: EvalRequest, job_id: str):
    """
    평가 작업을 실행하는 스레드를 시작합니다.
    """
    thread = threading.Thread(
        target=run_evaluation_thread,
        args=(request, job_id),
        daemon=True
    )
    thread.start()
    logger.info(f"Started evaluation thread for job {job_id}")

@router.delete("/{job_id}")
async def delete_evaluation_job(job_id: str):
    """
    특정 평가 작업을 삭제합니다.
    """
    try:
        # JSON 파일 경로
        json_file_path = os.path.join(EVAL_JOB_DATA_DIR, f"{job_id}.json")
        # TXT 파일 경로 (로그 파일)
        txt_file_path = os.path.join(EVAL_JOB_DATA_DIR, f"{job_id}.txt")
        
        deleted_files = []
        
        # JSON 파일 존재 확인 및 삭제
        if os.path.exists(json_file_path):
            os.remove(json_file_path)
            deleted_files.append(f"{job_id}.json")
            logger.info(f"Deleted JSON file: {json_file_path}")
        
        # TXT 파일 존재 확인 및 삭제
        if os.path.exists(txt_file_path):
            os.remove(txt_file_path)
            deleted_files.append(f"{job_id}.txt")
            logger.info(f"Deleted TXT file: {txt_file_path}")
        
        # 메모리에서도 제거 (실행 중인 작업이 있다면)
        with jobs_lock:
            if job_id in evaluation_jobs:
                del evaluation_jobs[job_id]
                logger.info(f"Removed job {job_id} from memory")
        
        if not deleted_files:
            raise HTTPException(status_code=404, detail="Evaluation job not found")
        
        return {
            "job_id": job_id,
            "message": f"Successfully deleted evaluation job",
            "deleted_files": deleted_files
        }
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error deleting evaluation job {job_id}: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error deleting evaluation job: {str(e)}")

# 여러 작업을 한번에 삭제하는 엔드포인트 (선택사항)
@router.delete("")
async def delete_multiple_evaluation_jobs(job_ids: List[str] = Query(..., description="List of job IDs to delete")):
    """
    여러 평가 작업을 한번에 삭제합니다.
    """
    try:
        deleted_jobs = []
        failed_jobs = []
        
        for job_id in job_ids:
            try:
                # JSON 파일 경로
                json_file_path = os.path.join(EVAL_JOB_DATA_DIR, f"{job_id}.json")
                # TXT 파일 경로
                txt_file_path = os.path.join(EVAL_JOB_DATA_DIR, f"{job_id}.txt")
                
                deleted_files = []
                
                # JSON 파일 삭제
                if os.path.exists(json_file_path):
                    os.remove(json_file_path)
                    deleted_files.append(f"{job_id}.json")
                
                # TXT 파일 삭제
                if os.path.exists(txt_file_path):
                    os.remove(txt_file_path)
                    deleted_files.append(f"{job_id}.txt")
                
                # 메모리에서 제거
                with jobs_lock:
                    if job_id in evaluation_jobs:
                        del evaluation_jobs[job_id]
                
                if deleted_files:
                    deleted_jobs.append({
                        "job_id": job_id,
                        "deleted_files": deleted_files
                    })
                    logger.info(f"Successfully deleted job {job_id}")
                else:
                    failed_jobs.append({
                        "job_id": job_id,
                        "error": "Job not found"
                    })
            
            except Exception as e:
                failed_jobs.append({
                    "job_id": job_id,
                    "error": str(e)
                })
                logger.error(f"Failed to delete job {job_id}: {str(e)}")
        
        return {
            "message": f"Processed {len(job_ids)} jobs",
            "deleted_jobs": deleted_jobs,
            "failed_jobs": failed_jobs
        }
    
    except Exception as e:
        logger.error(f"Error in bulk delete operation: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error in bulk delete operation: {str(e)}")
        
@router.post("", response_model=EvalResponse)
async def eval_endpoint(request: EvalRequest, background_tasks: BackgroundTasks):
    """
    평가 작업을 비동기적으로 시작합니다.
    """
    # 고유한 작업 ID 생성
    job_id = f"{request.job_name}_{uuid.uuid4().hex[:8]}"
    
    # 백그라운드에서 평가 작업 실행
    background_tasks.add_task(run_evaluation, request, job_id)
    
    return EvalResponse(
        job_id=job_id,
        status="accepted",
        message="Evaluation job started"
    )

@router.get("/{job_id}", response_model=EvalStatusResponse)
async def get_evaluation_status(job_id: str):
    """
    특정 평가 작업의 상태를 조회합니다.
    """
    try:
        # JSON 파일에서 작업 데이터 로드
        job_data = load_eval_job_from_json(job_id)
        
        return EvalStatusResponse(
            job_id=job_id,
            status=job_data.get("status", "unknown"),
            start_time=job_data.get("start_time"),
            end_time=job_data.get("end_time"),
            job_info=job_data.get("job_info", {}),
            result=job_data.get("result"),
            base_model_result=job_data.get("base_model_result"),
            base_model_name=job_data.get("base_model_name")
        )
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting evaluation status for {job_id}: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error getting evaluation status: {str(e)}")

@router.get("", response_model=Dict[str, Dict[str, Any]])
async def get_all_evaluations():
    """
    모든 평가 작업의 상태를 조회합니다.
    """
    try:
        return load_all_eval_jobs_from_json()
    except Exception as e:
        logger.error(f"Error getting all evaluation jobs: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error getting all evaluation jobs: {str(e)}")