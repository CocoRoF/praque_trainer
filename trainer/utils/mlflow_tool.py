import mlflow
import time
from mlflow.entities import ViewType
from mlflow.tracking import MlflowClient

def init_mlflow(mlflow_url: str, experiment_name: str, run_name: str, mlflow_force_run:bool = False, rank: int = 0):
    """
    MLflow URI 설정, 실험 복구 또는 생성, run 시작까지 모두 수행합니다.
    
    Args:
        mlflow_url (str): MLflow Tracking URL
        experiment_name (str): 실험 이름 (model_name 등)
        run_name (str): 실행 이름 (고유 ID 등)
        rank (int, optional): 분산 학습 시 rank (기본값: 0)
    """
    mlflow.set_tracking_uri(mlflow_url)

    client = MlflowClient()
    experiment = client.get_experiment_by_name(experiment_name)

    if experiment is not None:
        if experiment.lifecycle_stage == "deleted":
            if mlflow_force_run:
                print(f"[WARNING][MLFLOW] '{experiment_name}' is deleted. Deleting permanently and recreating...")
                client.delete_experiment(experiment.experiment_id)
                time.sleep(1)

                for _ in range(5):
                    try:
                        experiment_id = client.create_experiment(experiment_name)
                        print(f"[INFO][MLFLOW] Re-created Experiment: {experiment_id}")
                        break
                    except Exception as e:
                        print(f"[WARN][MLFLOW] Waiting for experiment deletion... ({e})")
                        time.sleep(1)
                else:
                    raise RuntimeError("[ERROR][MLFLOW] Failed to re-create experiment after 5 attempts.")
            else:
                print(f"[INFO][MLFLOW] '{experiment_name}' is deleted. Restoring...")
                client.restore_experiment(experiment.experiment_id)
                experiment_id = experiment.experiment_id
        else:
            print(f"[INFO][MLFLOW] '{experiment_name}' already exists.")
            experiment_id = experiment.experiment_id
    else:
        print(f"[INFO][MLFLOW] '{experiment_name}' does not exist. Creating...")
        experiment_id = client.create_experiment(experiment_name)

    if experiment_id is None:
        experiment = client.get_experiment_by_name(experiment_name)
        experiment_id = experiment.experiment_id

    mlflow.set_experiment(experiment_id=experiment_id)
    mlflow.start_run(run_name=run_name)

    print("============== [INFO][MLFLOW] Trainer Start MLFlow ==============")
    print(f"============== [INFO][MLFLOW] rank: {rank} ==============")
    print(f"============== [INFO][MLFLOW] MLFlow URL: {mlflow_url} ==============")
    print(f"============== [INFO][MLFLOW] Experiment Name: {experiment_name} ==============")
    print(f"============== [INFO][MLFLOW] Run Name: {run_name} ==============")

def get_mlflow_info(mlflow_url:str = "https://polar-mlflow-git.x2bee.com/", experiment_name: str = "", run_id:str = "" ) -> str:
    mlflow.set_tracking_uri(mlflow_url)
    experiment_name = experiment_name
    experiment = mlflow.get_experiment_by_name(experiment_name)

    if experiment is None:
        raise ValueError(f"[ERROR][MLFLOW] Experiment '{experiment_name}' not found.")

    print("[INFO][MLFLOW] Experiment ID:", experiment.experiment_id)

    runs = mlflow.search_runs(
        experiment_ids=[experiment.experiment_id],
        filter_string=f"tags.mlflow.runName = '{run_id}'",
        run_view_type=ViewType.ACTIVE_ONLY,
    )

    if runs.empty:
        print("Run not found.")
    else:
        run_id = runs.iloc[0]["run_id"]
        print("Run ID:", run_id)
        
    return experiment.experiment_id, run_id

if __name__ == "__main__":
    a, b = get_mlflow_info(mlflow_url="https://polar-mlflow-git.x2bee.com/", experiment_name="KoModernBERT-large-mlm-v20", run_id="KoModernBERT-large-mlm-v20_20250325_000756")
    
    print(a, b)
    