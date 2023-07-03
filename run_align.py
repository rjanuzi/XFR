import dataset as ds
from util._telegram import send_simple_message
from util.align_images import align_images
from pathlib import Path

__OUTPUT_SIZE = 256
__TRANSFORM_SIZE = 512

if __name__ == "__main__":
    try:
        raw_imgs_idx = ds.get_raw_imgs_dataset_index()
        raw_imgs_idx["aligned_path"] = raw_imgs_idx.img_path.apply(
            lambda path: path.replace(ds.DATASET_KIND_RAW, ds.DATASET_KIND_ALIGNED)
        )

        # Create output folders
        _ = raw_imgs_idx.aligned_path.apply(
            lambda path: Path(path).parent.mkdir(parents=True, exist_ok=True)
        )

        align_images(
            imgs_path_lst=raw_imgs_idx.img_path.tolist(),
            output_path_lst=raw_imgs_idx.aligned_path.tolist(),
            output_size=__OUTPUT_SIZE,
            transform_size=__TRANSFORM_SIZE,
        )
        send_simple_message("Alignment finished")
    except Exception as e:
        send_simple_message("Some error occurred while aligning images")
        raise e
