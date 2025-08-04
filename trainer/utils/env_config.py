"""
Environment configuration utilities for loading secrets from environment variables.
"""
import os
from typing import Optional


def get_huggingface_token() -> str:
    """
    Get HuggingFace token from environment variable.

    Returns:
        str: HuggingFace token

    Raises:
        ValueError: If token is not found in environment
    """
    token = os.getenv("HUGGING_FACE_HUB_TOKEN")
    if not token:
        raise ValueError(
            "HUGGING_FACE_HUB_TOKEN environment variable is required but not set. "
            "Please set it in your environment or .env file."
        )
    return token


def get_openai_api_key() -> str:
    """
    Get OpenAI API key from environment variable.

    Returns:
        str: OpenAI API key

    Raises:
        ValueError: If API key is not found in environment
    """
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError(
            "OPENAI_API_KEY environment variable is required but not set. "
            "Please set it in your environment or .env file."
        )
    return api_key


def get_minio_config() -> dict:
    """
    Get MinIO configuration from environment variables.
    Falls back to default values if not set.

    Returns:
        dict: MinIO configuration
    """
    return {
        "url": os.getenv("MINIO_URL", "polar-store-api.x2bee.com"),
        "access_key": os.getenv("MINIO_ACCESS_KEY", "W2EmgRFVCGzlQ8u5wUUW"),
        "secret_key": os.getenv("MINIO_SECRET_KEY", "NEfmJTEwDWm5XSyM6imBsl1QjrnmaZSAB37bRnDk"),
    }


def get_huggingface_user_id() -> str:
    """
    Get HuggingFace user ID from environment variable.
    Falls back to default if not set.

    Returns:
        str: HuggingFace user ID
    """
    return os.getenv("HUGGING_FACE_USER_ID", "CocoRoF")
