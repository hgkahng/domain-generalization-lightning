
import os
import shutil
import logging

from rich.console import Console
from rich.logging import RichHandler


def modify_lightning_logger_settings(name: str = 'pytorch_lightning',
                                     level=logging.INFO) -> None:

    # get logger
    logger = logging.getLogger(name)
    
    # remove existing handlers
    while logger.hasHandlers():
        logger.removeHandler(logger.handlers[0])

    # rich handler
    width, _ = shutil.get_terminal_size()
    console = Console(color_system='256', width=width)
    richHandler = RichHandler(console=console)
    richHandler.setFormatter(logging.Formatter("%(message)s"))
    logger.addHandler(richHandler)

    # set level
    logger.setLevel(level=level)
