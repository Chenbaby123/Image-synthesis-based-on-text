from diffusers import DiffusionPipeline
from PIL import Image
import time

import torch

startTime = time.time()
# load both base & refiner
base = DiffusionPipeline.from_pretrained(
    "stabilityai/stable-diffusion-xl-base-1.0", torch_dtype=torch.float16, variant="fp16", use_safetensors=True
)
base.to("cuda")
refiner = DiffusionPipeline.from_pretrained(
    "stabilityai/stable-diffusion-xl-refiner-1.0",
    text_encoder_2=base.text_encoder_2,
    vae=base.vae,
    torch_dtype=torch.float16,
    use_safetensors=True,
    variant="fp16",
)
refiner.to("cuda")

# Define how many steps and what % of steps to be run on each experts (80/20) here
n_steps = 100
high_noise_frac = 0.8
low_noise_frac = 0.2

prompt = "A majestic lion jumping from a big stone at night"

# run both experts
image = base(
    prompt=prompt,
    num_inference_steps=n_steps,
    denoising_start=high_noise_frac,
    output_type="latent",
).images
image = refiner(
    prompt=prompt,
    num_inference_steps=n_steps,
    denoising_start=high_noise_frac,
    image=image,
).images[0]

image.save('a3.png')


# prompt = "Three students are walking on the street"
#
# # run both experts
# image = base(
#     prompt=prompt,
#     num_inference_steps=n_steps,
#     denoising_start=high_noise_frac,
#     output_type="latent",
# ).images
# image = refiner(
#     prompt=prompt,
#     num_inference_steps=n_steps,
#     denoising_start=high_noise_frac,
#     image=image,
# ).images[0]
#
# image.save('b3.png')
#
#
#


# prompt = "Tom is enjoying sunshine on palm beach"
#
# # run both experts
# image = base(
#     prompt=prompt,
#     num_inference_steps=n_steps,
#     denoising_start=high_noise_frac,
#     output_type="latent",
# ).images
# image = refiner(
#     prompt=prompt,
#     num_inference_steps=n_steps,
#     denoising_start=high_noise_frac,
#     image=image,
# ).images[0]
#
# image.save('d3.png')
#
#
# prompt = "Three red apples and a banana on the desk"
#
# # run both experts
# image = base(
#     prompt=prompt,
#     num_inference_steps=n_steps,
#     denoising_start=high_noise_frac,
#     output_type="latent",
# ).images
# image = refiner(
#     prompt=prompt,
#     num_inference_steps=n_steps,
#     denoising_start=high_noise_frac,
#     image=image,
# ).images[0]
#
# image.save('e3.png')

# prompt = "Paint an oil painting"
# #
# #
# # # run both experts
# image = base(
#     prompt=prompt,
#     num_inference_steps=n_steps,
#     denoising_start=high_noise_frac,
#     output_type="latent",
# ).images
# image = refiner(
#     prompt=prompt,
#     num_inference_steps=n_steps,
#     denoising_start=high_noise_frac,
#     image=image,
# ).images[0]
#
# image.save('f3.png')

# prompt = "A robot painted as graffiti on a brick wall. a sidewalk is in front of the wall, and grass is growing out of cracks in the concrete"
#
#
# # run both experts
# image = base(
#     prompt=prompt,
#     num_inference_steps=n_steps,
#     denoising_start=high_noise_frac,
#     output_type="latent",
# ).images
# image = refiner(
#     prompt=prompt,
#     num_inference_steps=n_steps,
#     denoising_start=high_noise_frac,
#     image=image,
# ).images[0]
#
# image.save('g3.png')


# prompt = "Epic long distance cityscape photo of New York City flooded by the ocean and overgrown buildings and jungle ruins in rainforest, at sunset, cinematic shot, highly detailed, 8k, golden light"
#
# # prompt = "close up headshot, futuristic young woman, wild hair sly smile in front of gigantic UFO, dslr, sharp focus, dynamic composition"
# # run both experts
# image = base(
#     prompt=prompt,
#     num_inference_steps=n_steps,
#     denoising_start=high_noise_frac,
#     output_type="latent",
# ).images
# image = refiner(
#     prompt=prompt,
#     num_inference_steps=n_steps,
#     denoising_start=high_noise_frac,
#     image=image,
# ).images[0]
#
# image.save('h3.png')



# prompt = "A monkey is eatting banana"
# # run both experts
# image = base(
#     prompt=prompt,
#     num_inference_steps=n_steps,
#     denoising_start=high_noise_frac,
#     output_type="latent",
# ).images
# image = refiner(
#     prompt=prompt,
#     num_inference_steps=n_steps,
#     denoising_start=high_noise_frac,
#     image=image,
# ).images[0]
#
# image.save('j3.png')


endTime = time.time()
runTime = endTime - startTime
print("This operation takes: " + str(runTime) + " seconds.")


