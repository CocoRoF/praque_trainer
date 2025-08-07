"""
Hugging Face Repository Pull and Push Utilities

이 모듈은 Hugging Face Hub에서 모델 repository를 다운로드하고
다른 repository로 업로드하는 기능을 제공합니다.
"""

import os
import shutil
import tempfile
from typing import Optional, List
from huggingface_hub import (
    snapshot_download,
    upload_folder,
    HfApi,
    login
)
import logging

# 로깅 설정
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class HuggingFaceRepoManager:
    """Hugging Face Repository 관리 클래스"""

    def __init__(self, token: Optional[str] = None):
        """
        Args:
            token: Hugging Face access token. None이면 환경변수에서 가져옴
        """
        self.api = HfApi()
        if token:
            login(token=token)
        self.token = token

    def download_repo(
        self,
        repo_id: str,
        local_dir: Optional[str] = None,
        repo_type: str = "model",
        revision: str = "main",
        ignore_patterns: Optional[List[str]] = None
    ) -> str:
        """
        Hugging Face repository의 모든 파일을 다운로드합니다.

        Args:
            repo_id: 다운로드할 repository ID (예: "microsoft/DialoGPT-medium")
            local_dir: 로컬 저장 경로. None이면 임시 디렉토리 사용
            repo_type: repository 타입 ("model", "dataset", "space")
            revision: 브랜치/태그/커밋 (기본값: "main")
            ignore_patterns: 무시할 파일 패턴 리스트

        Returns:
            str: 다운로드된 로컬 디렉토리 경로
        """
        try:
            if local_dir is None:
                local_dir = tempfile.mkdtemp(prefix=f"hf_repo_{repo_id.replace('/', '_')}_")

            logger.info("Downloading repository '%s' to '%s'...", repo_id, local_dir)

            # repository 전체 다운로드
            downloaded_path = snapshot_download(
                repo_id=repo_id,
                repo_type=repo_type,
                revision=revision,
                local_dir=local_dir,
                local_dir_use_symlinks=False,  # 심볼릭 링크 대신 실제 파일 복사
                ignore_patterns=ignore_patterns
            )

            logger.info("Successfully downloaded repository to: %s", downloaded_path)
            return downloaded_path

        except Exception as e:
            logger.error("Error downloading repository '%s': %s", repo_id, str(e))
            raise

    def upload_repo(
        self,
        local_dir: str,
        target_repo_id: str,
        repo_type: str = "model",
        private: bool = False,
        commit_message: Optional[str] = None,
        create_pr: bool = False
    ) -> str:
        """
        로컬 디렉토리의 모든 파일을 Hugging Face repository로 업로드합니다.

        Args:
            local_dir: 업로드할 로컬 디렉토리 경로
            target_repo_id: 대상 repository ID
            repo_type: repository 타입 ("model", "dataset", "space")
            private: private repository 여부
            commit_message: 커밋 메시지
            create_pr: Pull Request 생성 여부

        Returns:
            str: 업로드된 repository URL
        """
        try:
            if not os.path.exists(local_dir):
                raise ValueError(f"Local directory does not exist: {local_dir}")

            # repository 존재 확인 및 생성
            try:
                self.api.repo_info(repo_id=target_repo_id, repo_type=repo_type)
                logger.info("Repository '%s' already exists", target_repo_id)
            except Exception:  # HfHubHTTPError나 기타 HF 관련 예외들을 포괄적으로 처리
                logger.info("Creating new repository: '%s'", target_repo_id)
                self.api.create_repo(
                    repo_id=target_repo_id,
                    repo_type=repo_type,
                    private=private
                )

            if commit_message is None:
                commit_message = "Upload files from local directory"

            logger.info("Uploading directory '%s' to repository '%s'...", local_dir, target_repo_id)

            # 폴더 전체 업로드
            upload_folder(
                folder_path=local_dir,
                repo_id=target_repo_id,
                repo_type=repo_type,
                commit_message=commit_message,
                create_pr=create_pr
            )

            repo_url = f"https://huggingface.co/{target_repo_id}"
            logger.info("Successfully uploaded to: %s", repo_url)
            return repo_url

        except Exception as e:
            logger.error("Error uploading to repository '%s': %s", target_repo_id, str(e))
            raise

    def pull_and_push(
        self,
        source_repo_id: str,
        target_repo_id: str,
        source_repo_type: str = "model",
        target_repo_type: str = "model",
        revision: str = "main",
        private: bool = False,
        commit_message: Optional[str] = None,
        ignore_patterns: Optional[List[str]] = None,
        cleanup_temp: bool = True
    ) -> str:
        """
        Hugging Face repository를 다운로드한 후 다른 repository로 업로드합니다.

        Args:
            source_repo_id: 소스 repository ID
            target_repo_id: 대상 repository ID
            source_repo_type: 소스 repository 타입
            target_repo_type: 대상 repository 타입
            revision: 소스의 브랜치/태그/커밋
            private: 대상 repository private 여부
            commit_message: 커밋 메시지
            ignore_patterns: 무시할 파일 패턴
            cleanup_temp: 임시 디렉토리 정리 여부

        Returns:
            str: 업로드된 repository URL
        """
        temp_dir = None
        try:
            # 1. 소스 repository 다운로드
            logger.info("Starting pull and push: %s -> %s", source_repo_id, target_repo_id)
            temp_dir = self.download_repo(
                repo_id=source_repo_id,
                repo_type=source_repo_type,
                revision=revision,
                ignore_patterns=ignore_patterns
            )

            # 2. 대상 repository로 업로드
            if commit_message is None:
                commit_message = f"Mirror from {source_repo_id}"

            repo_url = self.upload_repo(
                local_dir=temp_dir,
                target_repo_id=target_repo_id,
                repo_type=target_repo_type,
                private=private,
                commit_message=commit_message
            )

            logger.info("Successfully completed pull and push operation")
            return repo_url

        except Exception as e:
            logger.error("Error in pull and push operation: %s", str(e))
            raise
        finally:
            # 임시 디렉토리 정리
            if cleanup_temp and temp_dir and os.path.exists(temp_dir):
                try:
                    shutil.rmtree(temp_dir)
                    logger.info("Cleaned up temporary directory: %s", temp_dir)
                except OSError as e:
                    logger.warning("Failed to cleanup temporary directory: %s", str(e))


def pull_and_push_repo(
    source_repo_id: str,
    target_repo_id: str,
    token: Optional[str] = None,
    source_repo_type: str = "model",
    target_repo_type: str = "model",
    revision: str = "main",
    private: bool = False,
    commit_message: Optional[str] = None,
    ignore_patterns: Optional[List[str]] = None
) -> str:
    """
    간편한 pull and push 함수

    Args:
        source_repo_id: 소스 repository ID (예: "microsoft/DialoGPT-medium")
        target_repo_id: 대상 repository ID (예: "myusername/my-model")
        token: Hugging Face access token
        source_repo_type: 소스 repository 타입
        target_repo_type: 대상 repository 타입
        revision: 소스의 브랜치/태그/커밋
        private: 대상 repository private 여부
        commit_message: 커밋 메시지
        ignore_patterns: 무시할 파일 패턴 (예: ["*.git*", "*.md"])

    Returns:
        str: 업로드된 repository URL

    Example:
        >>> pull_and_push_repo(
        ...     source_repo_id="microsoft/DialoGPT-medium",
        ...     target_repo_id="myusername/my-dialogpt",
        ...     token="hf_...",
        ...     private=True,
        ...     commit_message="Mirror DialoGPT model"
        ... )
    """
    manager = HuggingFaceRepoManager(token=token)
    return manager.pull_and_push(
        source_repo_id=source_repo_id,
        target_repo_id=target_repo_id,
        source_repo_type=source_repo_type,
        target_repo_type=target_repo_type,
        revision=revision,
        private=private,
        commit_message=commit_message,
        ignore_patterns=ignore_patterns
    )


if __name__ == "__main__":
    # 사용 예시
    import argparse

    parser = argparse.ArgumentParser(description="Hugging Face Repository Pull and Push")
    parser.add_argument("--source", required=True, help="Source repository ID")
    parser.add_argument("--target", required=True, help="Target repository ID")
    parser.add_argument("--token", help="Hugging Face access token")
    parser.add_argument("--source-type", default="model", choices=["model", "dataset", "space"])
    parser.add_argument("--target-type", default="model", choices=["model", "dataset", "space"])
    parser.add_argument("--revision", default="main", help="Source revision")
    parser.add_argument("--private", action="store_true", help="Make target repo private")
    parser.add_argument("--message", help="Commit message")

    args = parser.parse_args()

    try:
        url = pull_and_push_repo(
            source_repo_id=args.source,
            target_repo_id=args.target,
            token=args.token,
            source_repo_type=args.source_type,
            target_repo_type=args.target_type,
            revision=args.revision,
            private=args.private,
            commit_message=args.message
        )
        print(f"Success! Repository available at: {url}")
    except (ValueError, RuntimeError, OSError) as e:
        print(f"Error: {str(e)}")
        exit(1)
