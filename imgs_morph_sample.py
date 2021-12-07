from img_factory.imgs_morph import morph
from img_factory.latent2img import add_original_background

PERSON_1 = "ffaria"
PERSON_2 = "raonifst"

generated_imgs = morph(PERSON_1, PERSON_2)
with_background = [add_original_background(PERSON_1, img) for img in generated_imgs]
