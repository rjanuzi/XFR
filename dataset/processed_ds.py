from pathlib import Path

PROCESSED_DATASET_FOLDER = Path(Path(__file__).parent, "processed")
PROCESSED_DATASET_FOLDER.mkdir(exist_ok=True)
