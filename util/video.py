import cv2


def imgs_to_video(img_paths, output_path, fps=10, size=(512, 512)):
    """
    imgs: list of images
    output_path: path to output video
    fps: frames per second
    size: size of the video
    """
    fourcc = cv2.VideoWriter_fourcc(*"MJPG")
    writer = cv2.VideoWriter(str(output_path), fourcc, fps, size)
    for img_path in img_paths:
        cv_img = cv2.imread(str(img_path))
        cv_img = cv2.resize(cv_img, size)
        writer.write(cv_img)
    writer.release()

    return output_path
