import json

def load_json(string: str) -> dict:
    """
    JSON형태의 str을 dict로 받는 함수
    """

    return json.loads(string)

def dumps_json(dictionary: dict) -> str:
    """
    dict를 JSON형태의 str로 바꾸는 함수
    """
    return json.dumps(dictionary, ensure_ascii=False)

def detect_language(text: str) -> str:
    """
    한국어 영어만 판단
    """
    if any('\uac00' <= ch <= '\ud7a3' for ch in text):
        return "Korean"
    return "English"
