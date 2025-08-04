import os
from dotenv import load_dotenv

load_dotenv()

def get_env_list(var_name: str, delimiter: str = ",") -> list:
    """
    주어진 환경 변수 값을 리스트로 변환하여 반환합니다.

    Parameters:
    - var_name (str): .env에 정의된 환경 변수 이름
    - delimiter (str): 분리자 (기본값: ',')

    Returns:
    - list[str]: 공백이 제거된 문자열 리스트. 환경 변수가 없으면 빈 리스트 반환
    """
    raw = os.getenv(var_name, "")
    return [item.strip() for item in raw.split(delimiter) if item.strip()]