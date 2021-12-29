from img_factory.latent2img import add_original_background, generate

NAMES = ["25078", "25169", "25711"]

generated_img = generate(names=NAMES)
reconstructed = (
    add_original_background(name=name, generated_img=generated_img)
    for name, generated_img in zip(NAMES, generated_img)
)

for img in generated_img:
    img.show()

for img in reconstructed:
    img.show()
