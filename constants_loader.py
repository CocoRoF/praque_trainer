import constants

def get_constant_list(var_name: str) -> list:
    """
    주어진 상수 변수 값을 리스트로 반환합니다.

    Parameters:
    - var_name (str): constants.py에 정의된 상수 변수 이름

    Returns:
    - list[str]: 문자열 리스트. 상수가 없으면 빈 리스트 반환
    """
    return getattr(constants, var_name, [])
