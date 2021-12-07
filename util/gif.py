from PIL import Image


def imgs_to_gif(imgs: list, gif_name: str, duration: int = 100) -> None:
    """
    imgs: list of images
    gif_name: name of gif
    duration: duration of gif
    """
    imgs[0].save(
        gif_name, save_all=True, append_images=imgs[1:], duration=duration, loop=0
    )
