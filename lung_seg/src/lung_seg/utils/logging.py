import logging
from datetime import datetime
from pathlib import Path

_LOGGER_NAME = "lung_seg"


def setup_logger(log_dir: str = "logs", tag: str = "run") -> logging.Logger:
    Path(log_dir).mkdir(parents=True, exist_ok=True)
    fname = Path(log_dir) / f"{tag}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"

    logger = logging.getLogger(_LOGGER_NAME)
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    logger.propagate = False

    fmt = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    fh = logging.FileHandler(fname)
    fh.setFormatter(fmt)
    sh = logging.StreamHandler()
    sh.setFormatter(fmt)
    logger.addHandler(fh)
    logger.addHandler(sh)
    return logger


def get_logger() -> logging.Logger:
    return logging.getLogger(_LOGGER_NAME)
