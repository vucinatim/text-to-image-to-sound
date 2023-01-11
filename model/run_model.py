from diffusers import StableDiffusionPipeline
import torch

model_path = "path_to_saved_model"
pipe = StableDiffusionPipeline.from_pretrained(model_path, torch_dtype=torch.float16)
pipe.to("cuda")

image = pipe(prompt="birds chirping").images[0]
image.save("birds-chirping.png")
