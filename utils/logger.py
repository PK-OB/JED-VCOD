# utils/logger.py

import logging
import os
import sys

def setup_logger(log_dir, experiment_name):
    """실험 로그를 파일과 콘솔에 모두 출력하도록 로거를 설정합니다."""
    log_formatter = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    # 이전에 추가된 핸들러가 있다면 제거 (중복 출력 방지)
    if logger.hasHandlers():
        logger.handlers.clear()

    # 로그 파일 핸들러
    os.makedirs(log_dir, exist_ok=True)
    file_handler = logging.FileHandler(os.path.join(log_dir, f"{experiment_name}.log"))
    file_handler.setFormatter(log_formatter)
    logger.addHandler(file_handler)

    # 콘솔 핸들러
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(log_formatter)
    logger.addHandler(console_handler)

    return logger