# # Files source
# 1) ffhq-dataset-v2.json - https://drive.google.com/open?id=16N0RV4fHI6joBuKbQAoG34V_cQk7vxSA
# 2) Annotation - https://github.com/DCGM/ffhq-features-dataset

import json
import traceback
from glob import glob
from pathlib import Path

ffhq_meta = json.load(open("ffhq-dataset-v2.json", "r"))
good_imgs = []
for meta_path in glob("ffhq-features-dataset\\json\\*"):
    meta = json.load(open(meta_path, "r"))
    try:
        checks = [
            meta[0]["faceAttributes"]["headPose"]["yaw"] <= 5,
            meta[0]["faceAttributes"]["headPose"]["yaw"] >= -5,
            meta[0]["faceAttributes"]["headPose"]["roll"] <= 5,
            meta[0]["faceAttributes"]["headPose"]["roll"] >= -5,
            meta[0]["faceAttributes"]["headPose"]["pitch"] <= 5,
            meta[0]["faceAttributes"]["headPose"]["pitch"] >= -5,
            meta[0]["faceAttributes"]["glasses"] == "NoGlasses",
            meta[0]["faceAttributes"]["smile"] <= 0.2,
            not meta[0]["faceAttributes"]["accessories"],
            not meta[0]["faceAttributes"]["occlusion"]["foreheadOccluded"],
            not meta[0]["faceAttributes"]["occlusion"]["eyeOccluded"],
            not meta[0]["faceAttributes"]["occlusion"]["mouthOccluded"],
        ]
        if all(checks):
            tmp_ffhq_meta = ffhq_meta[str(int(Path(meta_path).stem))]
            tmp_meta = {
                        "name": Path(meta_path).stem,
                        "flickr_url": tmp_ffhq_meta["metadata"]["photo_url"],
                        "drive_url": tmp_ffhq_meta["image"]["file_url"],
                        "face_attributes": meta[0]["faceAttributes"]
            }
            good_imgs.append(tmp_meta)
    except IndexError:
        pass
    except:
        print("Error: ", meta_path)
        print(traceback.format_exc())
len(good_imgs)

json.dump(good_imgs, open("ffhq_frontal.json", "w"))
with open("ffhq_frontal_imgs.txt", "w") as f:
    f.write('\n'.join((i['name'] for i in good_imgs)))