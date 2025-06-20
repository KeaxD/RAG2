import logging
from pathlib import Path
from unstructured.partition.auto import partition
import sys
print("Python executable:", sys.executable)


DATA_PATH = "./data"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s"
)
logger = logging.getLogger(__name__)


def load_and_chunk(folder: str):
    logger.info(f"Looking for files under {folder}")
    for file_path in Path(folder).rglob("*"):
        logger.debug(f"Found path: {file_path}")
        if not file_path.is_file():
            logger.debug("  → skipping (not a file)")
            continue

        logger.info(f"Processing file: {file_path}")
        try:
            with open(file_path, "rb") as f:
                elements = partition(file=f, include_page_breaks=True)
        except Exception as e:
            logger.error(f"Partition failed for {file_path}: {e}")
            continue

        logger.info(f"  → got {len(elements)} elements")

def main():
    load_and_chunk(DATA_PATH)

if __name__ == "__main__":
    main()