import logging
import os
from datetime import datetime

# Today's date for folder
today = datetime.now().strftime("%Y-%m-%d")
log_folder = os.path.join("logs", today)
os.makedirs(log_folder, exist_ok=True)

# Log file paths
success_log = os.path.join(log_folder, "success.log")
error_log = os.path.join(log_folder, "error.log")
debug_log = os.path.join(log_folder, "debug.log")

# Log formatter
# formatter = logging.Formatter("[%(asctime)s] [%(levelname)s] - %(message)s")
formatter = logging.Formatter(
    "[%(asctime)s] [%(levelname)s] [%(filename)s] - %(message)s"
)


# 🔹 Filters
class LevelFilter(logging.Filter):
    def __init__(self, level):
        self.level = level

    def filter(self, record):
        return record.levelno == self.level


# 🔸 Success (INFO only) - Fixed Unicode encoding for Windows
success_handler = logging.FileHandler(success_log, encoding='utf-8')
success_handler.setLevel(logging.INFO)
success_handler.setFormatter(formatter)
success_handler.addFilter(LevelFilter(logging.INFO))

# 🔸 Error (ERROR only) - Fixed Unicode encoding for Windows
error_handler = logging.FileHandler(error_log, encoding='utf-8')
error_handler.setLevel(logging.ERROR)
error_handler.setFormatter(formatter)
error_handler.addFilter(LevelFilter(logging.ERROR))

# 🔸 Debug (DEBUG only) - Fixed Unicode encoding for Windows
debug_handler = logging.FileHandler(debug_log, encoding='utf-8')
debug_handler.setLevel(logging.DEBUG)
debug_handler.setFormatter(formatter)
debug_handler.addFilter(LevelFilter(logging.DEBUG))


# 🔹 Main logger
logger = logging.getLogger("app-logger")
logger.setLevel(logging.DEBUG)
logger.addHandler(success_handler)
logger.addHandler(error_handler)
logger.addHandler(debug_handler)
logger.propagate = False 