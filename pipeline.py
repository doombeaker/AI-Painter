import os

import shared
from shared import logging
import oneflow as torch
from diffusers import (
    OneFlowStableDiffusionPipeline as DiffusionPipeline,
    OneFlowDPMSolverMultistepScheduler as DPMSolverMultistepScheduler,
)

logging.basicConfig(
    level=shared.logging.INFO,
    format="%(asctime)s %(filename)s[line:%(lineno)d] %(levelname)s %(message)s",
    filename="output.log",
    filemode="w",
)

device_placement = shared.cmd_opts.device
repo_id = shared.cmd_opts.ckpt
model_id = "stabilityai/stable-diffusion-2"

class DiffusionPipelineHandler:
    if not shared.cmd_opts.ui_debug_mode:
        logging.info("DiffusionPipeline initialization")
        if not os.path.exists(repo_id):
            repo_id = model_id
        pipe = DiffusionPipeline.from_pretrained(
            repo_id, torch_dtype=torch.float16, revision="fp16"
        )

        pipe.scheduler = DPMSolverMultistepScheduler.from_config(repo_id, subfolder="scheduler")
        pipe = pipe.to(device_placement)
        logging.info("DiffusionPipeline initialization completed")

    def __init__(
        self,
        prompt: str,
        width: int = 768,
        height: int = 768,
        num_inference_steps: int = 25,
        guidance_scale: float = 7.5,
        negative_prompt: str = None,
        num_images_per_prompt: int = 1,
        eta=0.0,
        seed: int = -1,
        output_type="pil",
        device_placement="cuda",
    ):
        self.prompt = prompt
        self.width = width
        self.hegiht = height
        self.num_inference_steps = num_inference_steps
        self.guidance_scale = guidance_scale
        self.negative_prompt = negative_prompt
        self.num_images_per_prompt = num_images_per_prompt
        self.eta = eta
        self.seed = seed
        self.output_type = output_type
        self.device_placement = device_placement

    def __call__(self):
        generator = None
        if self.seed != -1:
            generator = torch.Generator(device=device_placement)
            generator.manual_seed(self.seed)
        result = DiffusionPipelineHandler.pipe(
            self.prompt,
            height=self.hegiht,
            width=self.width,
            num_inference_steps=self.num_inference_steps,
            guidance_scale=self.guidance_scale,
            negative_prompt=self.negative_prompt,
            num_images_per_prompt=self.num_images_per_prompt,
            eta=self.eta,
            generator=generator,
            output_type=self.output_type,
        )
        return result.images


if __name__ == "__main__":
    if not shared.cmd_opts.ui_debug_mode:
        phandler = DiffusionPipelineHandler("a dog with glasses")
        imgs = phandler()
        print(type(imgs), type(imgs[0]))
        imgs[0].save("demo.png")
