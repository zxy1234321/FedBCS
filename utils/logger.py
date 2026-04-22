import datetime
from typing import Literal, Union
from pathlib import Path
import logging
import logging.handlers
import os
import sys


class Logger:
    def __init__(
        self,
        level: Literal["CRITICAL", "ERROR", "WARNING", "INFO", "DEBUG"],
        log_dir: Union[Path, str] = None,
        comment: str = "logs",
        formatter: str = None,
        use_timestamp: bool = False,
        file_level: Literal["CRITICAL", "ERROR", "WARNING", "INFO", "DEBUG"] = None,
    ) -> None:
        self.level = level
        self.comment = comment
        self.log_parent_dir = log_dir
        self.use_timestamp = use_timestamp
        if formatter is None:
            self.formatter = "%(asctime)s [%(levelname)s] - %(message)s"
        else:
            self.formatter = formatter
        if file_level is None:
            self.file_level = level
        else:
            self.file_level = file_level

    def create_handler(self, logger: logging.Logger) -> None:
        log_handlers = {"StreamHandler": logging.StreamHandler(stream=sys.stdout)}
        fh_formatter = logging.Formatter(f"{self.formatter}")
        log_handlers["StreamHandler"].setLevel(self.level)

        if self.log_parent_dir is not None:
            log_parent_dir = Path(self.log_parent_dir)
            if self.use_timestamp:
                log_name = f'{datetime.datetime.now().strftime("%Y-%m-%dT%H%M%S")}_{self.comment}.log'
            else:
                log_name = f"{self.comment}.log"
            log_parent_dir.mkdir(parents=True, exist_ok=True)

            should_roll_over = os.path.isfile(log_parent_dir / log_name)

            log_handlers["FileHandler"] = logging.handlers.RotatingFileHandler(
                log_parent_dir / log_name, backupCount=5
            )

            if should_roll_over:
                log_handlers["FileHandler"].doRollover()
            log_handlers["FileHandler"].setLevel(self.file_level)

        for handler in log_handlers.values():
            handler.setFormatter(fh_formatter)
            logger.addHandler(handler)

    def create_logger(self) -> logging.Logger:
        logger = logging.getLogger("__main__")
        logger.addHandler(logging.NullHandler())

        logger.setLevel("DEBUG")
        self.create_handler(logger)

        return logger
