from img_factory.latent2img import add_original_background, generate

PERSON = "ffaria"

generated_img = generate(PERSON)
reconstructed = add_original_background(person_name=PERSON, generated_img=generated_img)
