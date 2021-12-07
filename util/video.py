import cv2


def imgs_to_video(imgs, output_path, fps=30, size=(512, 512)):
    """
    imgs: list of images
    output_path: path to output video
    fps: frames per second
    size: size of the video
    """
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    video = cv2.VideoWriter(output_path, fourcc, fps, size)
    for img in imgs:
        video.write(img)
    video.release()


# height, width, layers = frame.shape

# video = cv2.VideoWriter(video_name, 0, 1, (width, height))

# for image in images:
#     video.write(cv2.imread(os.path.join(image_folder, image)))

# cv2.destroyAllWindows()
# video.release()
