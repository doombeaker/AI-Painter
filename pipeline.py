import os

import cv2
import numpy as np
import shared
from shared import logging

from onediff import OneFlowStableDiffusionImg2ImgPipeline
from onediff import OneFlowStableDiffusionPipeline
from onediff import OneFlowStableDiffusionControlNetPipeline

import oneflow as flow
flow.mock_torch.enable()

from PIL import Image

from diffusers import (
    ControlNetModel,
    DPMSolverMultistepScheduler,
)

logging.basicConfig(
    level=shared.logging.INFO,
    format="%(asctime)s %(filename)s[line:%(lineno)d] %(levelname)s %(message)s",
    filename="output.log",
    filemode="w",
)

device_placement = shared.cmd_opts.device
# model_id = "CompVis/stable-diffusion-v1-4"
model_id = "runwayml/stable-diffusion-v1-5"


class DiffusionImg2ImgPipelineHandler:
    if not shared.cmd_opts.ui_debug_mode:
        logging.info("OneFlowStableDiffusionImg2ImgPipeline initialization")

        pipe = OneFlowStableDiffusionImg2ImgPipeline.from_pretrained(
            model_id,
            revision="fp16",
            torch_dtype=flow.float16,
        )
        pipe.scheduler = DPMSolverMultistepScheduler.from_config(
            model_id, subfolder="scheduler"
        )
        pipe = pipe.to(device_placement)
        logging.info("OneFlowStableDiffusionImg2ImgPipeline initialization completed")

    def __init__(
        self,
        prompt: str,
        init_image: Image.Image,
        strength: float = 0.8,
        width: int = 768,
        height: int = 768,
        num_inference_steps: int = 50,
        guidance_scale: float = 7.5,
        negative_prompt: str = None,
        num_images_per_prompt: int = 1,
        eta=0.0,
        seed: int = -1,
        output_type="pil",
        device_placement="cuda",
    ):
        self.prompt = prompt
        self.image = init_image.resize((width, height))
        self.strength = strength
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
            generator = flow.Generator(device=device_placement)
            generator.manual_seed(self.seed)
        with flow.autocast("cuda"):
            result = DiffusionImg2ImgPipelineHandler.pipe(
                prompt=self.prompt,
                image=self.image,
                strength=self.strength,
                num_inference_steps=self.num_inference_steps,
                guidance_scale=self.guidance_scale,
                negative_prompt=self.negative_prompt,
                num_images_per_prompt=self.num_images_per_prompt,
                eta=self.eta,
                generator=generator,
                output_type=self.output_type,
                compile_unet=shared.cmd_opts.graph_mode,
            ).images
        return result

class DiffusionPipelineHandler:
    if not shared.cmd_opts.ui_debug_mode:
        logging.info("OneFlowStableDiffusionPipeline initialization")

        pipe = OneFlowStableDiffusionPipeline.from_pretrained(
            model_id, revision="fp16", torch_dtype=flow.float16
        )

        pipe.scheduler = DPMSolverMultistepScheduler.from_config(
            model_id, subfolder="scheduler"
        )
        pipe = pipe.to(device_placement)
        logging.info("OneFlowStableDiffusionPipeline initialization completed")

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
            generator = flow.Generator(device=device_placement)
            generator.manual_seed(self.seed)
        with flow.autocast("cuda"):
            result = DiffusionPipelineHandler.pipe(
                prompt=self.prompt,
                height=self.hegiht,
                width=self.width,
                num_inference_steps=self.num_inference_steps,
                guidance_scale=self.guidance_scale,
                negative_prompt=self.negative_prompt,
                num_images_per_prompt=self.num_images_per_prompt,
                eta=self.eta,
                generator=generator,
                output_type=self.output_type,
                compile_unet=shared.cmd_opts.graph_mode,
            ).images
        return result

class DiffusionControlNetCannyPipelineHandler:
    if not shared.cmd_opts.ui_debug_mode:
        logging.info("DiffusionControlNetCannyPipelineHandler initialization")
        controlnet = ControlNetModel.from_pretrained("lllyasviel/sd-controlnet-canny", torch_dtype=flow.float16)

        pipe = OneFlowStableDiffusionControlNetPipeline.from_pretrained(
            model_id,
            controlnet=controlnet,
            revision="fp16",
            torch_dtype=flow.float16,
        )

        pipe = pipe.to(device_placement)
        logging.info("DiffusionControlNetCannyPipelineHandler initialization completed")

    def __init__(
        self,
        prompt: str,
        init_image: Image.Image,
        strength: float = 0.8,
        width: int = 512,
        height: int = 512,
        num_inference_steps: int = 50,
        guidance_scale: float = 7.5,
        negative_prompt: str = None,
        num_images_per_prompt: int = 1,
        eta=0.0,
        seed: int = -1,
        output_type="pil",
        device_placement="cuda",
    ):
        self.prompt = prompt
        self.image = init_image.resize((width, height))
        self.strength = strength
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
            generator = flow.Generator(device=device_placement)
            generator.manual_seed(self.seed)
            
        image = np.array(self.image)

        low_threshold = 100
        high_threshold = 200

        image = cv2.Canny(image, low_threshold, high_threshold)
        image = image[:, :, None]
        image = np.concatenate([image, image, image], axis=2)
        canny_image = Image.fromarray(image)

        with flow.autocast("cuda"):
            result = DiffusionControlNetCannyPipelineHandler.pipe(
                prompt=self.prompt,
                image=canny_image,
                num_inference_steps=self.num_inference_steps,
                guidance_scale=self.guidance_scale,
                negative_prompt=self.negative_prompt,
                num_images_per_prompt=self.num_images_per_prompt,
                eta=self.eta,
                generator=generator,
                output_type=self.output_type,
            ).images
        return result

if __name__ == "__main__":
    if not shared.cmd_opts.ui_debug_mode:
        init_image = Image.open("test.jpg").convert("RGB")
        phandler = DiffusionImg2ImgPipelineHandler(
            "a dog with glasses", init_image=init_image
        )
        imgs = phandler()
        imgs[0].save("demo.png")
