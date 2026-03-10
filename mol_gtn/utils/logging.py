from __future__ import annotations

import logging
from logging.handlers import QueueHandler, QueueListener
from multiprocessing import Queue
from pathlib import Path
from typing import Optional, Tuple


LOG_FORMAT = "%(asctime)s | %(processName)s | %(levelname)s | %(message)s"


def ensure_output_dir(path: str) -> Path:
    directory = Path(path)
    directory.mkdir(parents=True, exist_ok=True)
    return directory


def configure_logging(log_path: str, level: int = logging.INFO) -> logging.Logger:
    ensure_output_dir(str(Path(log_path).parent))
    logger = logging.getLogger("mol_gtn")
    logger.setLevel(level)
    logger.handlers.clear()
    formatter = logging.Formatter(LOG_FORMAT)

    file_handler = logging.FileHandler(log_path)
    file_handler.setFormatter(formatter)
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)

    logger.addHandler(file_handler)
    logger.addHandler(stream_handler)
    logger.propagate = False
    return logger


def build_queue_logging(log_path: str) -> Tuple[Queue, QueueListener]:
    ensure_output_dir(str(Path(log_path).parent))
    formatter = logging.Formatter(LOG_FORMAT)
    file_handler = logging.FileHandler(log_path)
    file_handler.setFormatter(formatter)
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    queue: Queue = Queue()
    listener = QueueListener(queue, file_handler, stream_handler)
    listener.start()
    return queue, listener


def attach_queue_logger(queue: Optional[Queue], level: int = logging.INFO) -> logging.Logger:
    logger = logging.getLogger("mol_gtn")
    logger.setLevel(level)
    logger.handlers.clear()
    if queue is None:
        return configure_logging("/hy-tmp/result/project.log", level=level)
    logger.addHandler(QueueHandler(queue))
    logger.propagate = False
    return logger
