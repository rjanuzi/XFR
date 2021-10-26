import gdown

DATASET_DRIVE_SHARED_FOLDER = (
    "https://drive.google.com/drive/folders/1C5XZIPeARoVjotVdYCNRVqolIFOGV6BU"
)

if __name__ == "__main__":
    gdown.download_folder(url=DATASET_DRIVE_SHARED_FOLDER, output="dataset")
