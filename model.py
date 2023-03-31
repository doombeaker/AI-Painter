from __future__ import annotations

import gc

import cv2
import numpy as np
import PIL.Image
from onediff import OneFlowStableDiffusionControlNetPipeline as StableDiffusionControlNetPipeline

import oneflow as flow
flow.mock_torch.enable()

from diffusers import (ControlNetModel, DiffusionPipeline)

from annotator.canny import apply_canny
from annotator.util import HWC3, resize_image
from annotator.hed import nms

CONTROLNET_MODEL_IDS = {
    'canny': 'lllyasviel/sd-controlnet-canny',
    'hough': 'lllyasviel/sd-controlnet-mlsd',
    'hed': 'lllyasviel/sd-controlnet-hed',
    'scribble': 'lllyasviel/sd-controlnet-scribble',
    'pose': 'lllyasviel/sd-controlnet-openpose',
    'seg': 'lllyasviel/sd-controlnet-seg',
    'depth': 'lllyasviel/sd-controlnet-depth',
    'normal': 'lllyasviel/sd-controlnet-normal',
}


def download_all_controlnet_weights() -> None:
    for model_id in CONTROLNET_MODEL_IDS.values():
        ControlNetModel.from_pretrained(model_id)


class Model:
    def __init__(self,
                 base_model_id: str = "runwayml/stable-diffusion-v1-5",
                 task_name: str = 'canny'):
        self.device = flow.device(
            'cuda')
        self.base_model_id = ''
        self.task_name = ''
        self.pipe = self.load_pipe(base_model_id, task_name)

    def load_pipe(self, base_model_id: str, task_name) -> DiffusionPipeline:
        if base_model_id == self.base_model_id and task_name == self.task_name and hasattr(
                self, 'pipe'):
            return self.pipe
        model_id = CONTROLNET_MODEL_IDS[task_name]
        controlnet = ControlNetModel.from_pretrained(model_id,
                                                     torch_dtype=flow.float16)
        pipe = StableDiffusionControlNetPipeline.from_pretrained(
            base_model_id,
            safety_checker=None,
            controlnet=controlnet,
            torch_dtype=flow.float16)

        pipe.to(self.device)
        flow.cuda.empty_cache()
        gc.collect()
        self.base_model_id = base_model_id
        self.task_name = task_name
        return pipe

    def set_base_model(self, base_model_id: str) -> str:
        if not base_model_id or base_model_id == self.base_model_id:
            return self.base_model_id
        del self.pipe
        flow.cuda.empty_cache()
        gc.collect()
        try:
            self.pipe = self.load_pipe(base_model_id, self.task_name)
        except Exception:
            self.pipe = self.load_pipe(self.base_model_id, self.task_name)
        return self.base_model_id

    def load_controlnet_weight(self, task_name: str) -> None:
        if task_name == self.task_name:
            return
        if 'controlnet' in self.pipe.__dict__:
            del self.pipe.controlnet
        flow.cuda.empty_cache()
        gc.collect()
        model_id = CONTROLNET_MODEL_IDS[task_name]
        controlnet = ControlNetModel.from_pretrained(model_id,
                                                     torch_dtype=flow.float16)
        controlnet.to(self.device)
        flow.cuda.empty_cache()
        gc.collect()
        self.pipe.controlnet = controlnet
        self.task_name = task_name

    def get_prompt(self, prompt: str, additional_prompt: str) -> str:
        if not prompt:
            prompt = additional_prompt
        else:
            prompt = f'{prompt}, {additional_prompt}'
        return prompt

    @flow.autocast('cuda')
    def run_pipe(
        self,
        prompt: str,
        negative_prompt: str,
        control_image: PIL.Image.Image,
        num_images: int,
        num_steps: int,
        guidance_scale: float,
        seed: int,
    ) -> list[PIL.Image.Image]:
        if seed == -1:
            seed = np.random.randint(0, np.iinfo(np.int64).max)
        generator = flow.manual_seed(seed)
        return self.pipe(prompt=prompt,
                         negative_prompt=negative_prompt,
                         guidance_scale=guidance_scale,
                         num_images_per_prompt=num_images,
                         num_inference_steps=num_steps,
                         generator=generator,
                         image=control_image).images

    @staticmethod
    def preprocess_canny(
        input_image: np.ndarray,
        image_resolution: int,
        low_threshold: int,
        high_threshold: int,
    ) -> tuple[PIL.Image.Image, PIL.Image.Image]:
        image = resize_image(HWC3(input_image), image_resolution)
        control_image = apply_canny(image, low_threshold, high_threshold)
        control_image = HWC3(control_image)
        vis_control_image = 255 - control_image
        return PIL.Image.fromarray(control_image), PIL.Image.fromarray(
            vis_control_image)

    @flow.inference_mode()
    def process_canny(
        self,
        input_image: np.ndarray,
        prompt: str,
        additional_prompt: str,
        negative_prompt: str,
        num_images: int,
        image_resolution: int,
        num_steps: int,
        guidance_scale: float,
        seed: int,
        low_threshold: int,
        high_threshold: int,
    ) -> list[PIL.Image.Image]:
        control_image, vis_control_image = self.preprocess_canny(
            input_image=input_image,
            image_resolution=image_resolution,
            low_threshold=low_threshold,
            high_threshold=high_threshold,
        )
        self.load_controlnet_weight('canny')
        results = self.run_pipe(
            prompt=self.get_prompt(prompt, additional_prompt),
            negative_prompt=negative_prompt,
            control_image=control_image,
            num_images=num_images,
            num_steps=num_steps,
            guidance_scale=guidance_scale,
            seed=seed,
        )
        return [vis_control_image] + results

    @staticmethod
    def preprocess_hough(
        input_image: np.ndarray,
        image_resolution: int,
        detect_resolution: int,
        value_threshold: float,
        distance_threshold: float,
    ) -> tuple[PIL.Image.Image, PIL.Image.Image]:
        input_image = HWC3(input_image)
        with flow.mock_torch.disable():
            from annotator.mlsd import apply_mlsd
            control_image = apply_mlsd(
                resize_image(input_image, detect_resolution), value_threshold,
                distance_threshold)
            
        control_image = HWC3(control_image)
        image = resize_image(input_image, image_resolution)
        H, W = image.shape[:2]
        control_image = cv2.resize(control_image, (W, H),
                                   interpolation=cv2.INTER_NEAREST)

        vis_control_image = 255 - cv2.dilate(
            control_image, np.ones(shape=(3, 3), dtype=np.uint8), iterations=1)

        return PIL.Image.fromarray(control_image), PIL.Image.fromarray(
            vis_control_image)

    @flow.inference_mode()
    def process_hough(
        self,
        input_image: np.ndarray,
        prompt: str,
        additional_prompt: str,
        negative_prompt: str,
        num_images: int,
        image_resolution: int,
        detect_resolution: int,
        num_steps: int,
        guidance_scale: float,
        seed: int,
        value_threshold: float,
        distance_threshold: float,
    ) -> list[PIL.Image.Image]:
        control_image, vis_control_image = self.preprocess_hough(
            input_image=input_image,
            image_resolution=image_resolution,
            detect_resolution=detect_resolution,
            value_threshold=value_threshold,
            distance_threshold=distance_threshold,
        )
        self.load_controlnet_weight('hough')
        results = self.run_pipe(
            prompt=self.get_prompt(prompt, additional_prompt),
            negative_prompt=negative_prompt,
            control_image=control_image,
            num_images=num_images,
            num_steps=num_steps,
            guidance_scale=guidance_scale,
            seed=seed,
        )
        return [vis_control_image] + results

    @staticmethod
    def preprocess_hed(
        input_image: np.ndarray,
        image_resolution: int,
        detect_resolution: int,
    ) -> tuple[PIL.Image.Image, PIL.Image.Image]:
        input_image = HWC3(input_image)
        with flow.mock_torch.disable():
            from annotator.hed import apply_hed
            control_image = apply_hed(resize_image(input_image, detect_resolution))
            
        control_image = HWC3(control_image)
        image = resize_image(input_image, image_resolution)
        H, W = image.shape[:2]
        control_image = cv2.resize(control_image, (W, H),
                                   interpolation=cv2.INTER_LINEAR)
        return PIL.Image.fromarray(control_image), PIL.Image.fromarray(
            control_image)

    @flow.inference_mode()
    def process_hed(
        self,
        input_image: np.ndarray,
        prompt: str,
        additional_prompt: str,
        negative_prompt: str,
        num_images: int,
        image_resolution: int,
        detect_resolution: int,
        num_steps: int,
        guidance_scale: float,
        seed: int,
    ) -> list[PIL.Image.Image]:
        control_image, vis_control_image = self.preprocess_hed(
            input_image=input_image,
            image_resolution=image_resolution,
            detect_resolution=detect_resolution,
        )
        self.load_controlnet_weight('hed')
        results = self.run_pipe(
            prompt=self.get_prompt(prompt, additional_prompt),
            negative_prompt=negative_prompt,
            control_image=control_image,
            num_images=num_images,
            num_steps=num_steps,
            guidance_scale=guidance_scale,
            seed=seed,
        )
        return [vis_control_image] + results

    @staticmethod
    def preprocess_scribble(
        input_image: np.ndarray,
        image_resolution: int,
    ) -> tuple[PIL.Image.Image, PIL.Image.Image]:
        image = resize_image(HWC3(input_image), image_resolution)
        control_image = np.zeros_like(image, dtype=np.uint8)
        control_image[np.min(image, axis=2) < 127] = 255
        vis_control_image = 255 - control_image
        return PIL.Image.fromarray(control_image), PIL.Image.fromarray(
            vis_control_image)

    @flow.inference_mode()
    def process_scribble(
        self,
        input_image: np.ndarray,
        prompt: str,
        additional_prompt: str,
        negative_prompt: str,
        num_images: int,
        image_resolution: int,
        num_steps: int,
        guidance_scale: float,
        seed: int,
    ) -> list[PIL.Image.Image]:
        control_image, vis_control_image = self.preprocess_scribble(
            input_image=input_image,
            image_resolution=image_resolution,
        )
        self.load_controlnet_weight('scribble')
        results = self.run_pipe(
            prompt=self.get_prompt(prompt, additional_prompt),
            negative_prompt=negative_prompt,
            control_image=control_image,
            num_images=num_images,
            num_steps=num_steps,
            guidance_scale=guidance_scale,
            seed=seed,
        )
        return [vis_control_image] + results

    @staticmethod
    def preprocess_scribble_interactive(
        input_image: np.ndarray,
        image_resolution: int,
    ) -> tuple[PIL.Image.Image, PIL.Image.Image]:
        image = resize_image(HWC3(input_image['mask'][:, :, 0]),
                             image_resolution)
        control_image = np.zeros_like(image, dtype=np.uint8)
        control_image[np.min(image, axis=2) > 127] = 255
        vis_control_image = 255 - control_image
        return PIL.Image.fromarray(control_image), PIL.Image.fromarray(
            vis_control_image)

    @flow.inference_mode()
    def process_scribble_interactive(
        self,
        input_image: np.ndarray,
        prompt: str,
        additional_prompt: str,
        negative_prompt: str,
        num_images: int,
        image_resolution: int,
        num_steps: int,
        guidance_scale: float,
        seed: int,
    ) -> list[PIL.Image.Image]:
        control_image, vis_control_image = self.preprocess_scribble_interactive(
            input_image=input_image,
            image_resolution=image_resolution,
        )
        self.load_controlnet_weight('scribble')
        results = self.run_pipe(
            prompt=self.get_prompt(prompt, additional_prompt),
            negative_prompt=negative_prompt,
            control_image=control_image,
            num_images=num_images,
            num_steps=num_steps,
            guidance_scale=guidance_scale,
            seed=seed,
        )
        return [vis_control_image] + results

    @staticmethod
    def preprocess_fake_scribble(
        input_image: np.ndarray,
        image_resolution: int,
        detect_resolution: int,
    ) -> tuple[PIL.Image.Image, PIL.Image.Image]:
        input_image = HWC3(input_image)
        with flow.mock_torch.disable():
            from annotator.hed import apply_hed
            control_image = apply_hed(resize_image(input_image, detect_resolution))
            
        control_image = HWC3(control_image)
        image = resize_image(input_image, image_resolution)
        H, W = image.shape[:2]

        control_image = cv2.resize(control_image, (W, H),
                                   interpolation=cv2.INTER_LINEAR)
        control_image = nms(control_image, 127, 3.0)
        control_image = cv2.GaussianBlur(control_image, (0, 0), 3.0)
        control_image[control_image > 4] = 255
        control_image[control_image < 255] = 0

        vis_control_image = 255 - control_image

        return PIL.Image.fromarray(control_image), PIL.Image.fromarray(
            vis_control_image)

    @flow.inference_mode()
    def process_fake_scribble(
        self,
        input_image: np.ndarray,
        prompt: str,
        additional_prompt: str,
        negative_prompt: str,
        num_images: int,
        image_resolution: int,
        detect_resolution: int,
        num_steps: int,
        guidance_scale: float,
        seed: int,
    ) -> list[PIL.Image.Image]:
        control_image, vis_control_image = self.preprocess_fake_scribble(
            input_image=input_image,
            image_resolution=image_resolution,
            detect_resolution=detect_resolution,
        )
        self.load_controlnet_weight('scribble')
        results = self.run_pipe(
            prompt=self.get_prompt(prompt, additional_prompt),
            negative_prompt=negative_prompt,
            control_image=control_image,
            num_images=num_images,
            num_steps=num_steps,
            guidance_scale=guidance_scale,
            seed=seed,
        )
        return [vis_control_image] + results

    @staticmethod
    def preprocess_pose(
        input_image: np.ndarray,
        image_resolution: int,
        detect_resolution: int,
        is_pose_image: bool,
    ) -> tuple[PIL.Image.Image, PIL.Image.Image]:
        input_image = HWC3(input_image)
        if not is_pose_image:
            with flow.mock_torch.disable():
                from annotator.openpose import apply_openpose
                control_image, _ = apply_openpose(
                    resize_image(input_image, detect_resolution))
                
            control_image = HWC3(control_image)
            image = resize_image(input_image, image_resolution)
            H, W = image.shape[:2]
            control_image = cv2.resize(control_image, (W, H),
                                       interpolation=cv2.INTER_NEAREST)
        else:
            control_image = resize_image(input_image, image_resolution)

        return PIL.Image.fromarray(control_image), PIL.Image.fromarray(
            control_image)

    @flow.inference_mode()
    def process_pose(
        self,
        input_image: np.ndarray,
        prompt: str,
        additional_prompt: str,
        negative_prompt: str,
        num_images: int,
        image_resolution: int,
        detect_resolution: int,
        num_steps: int,
        guidance_scale: float,
        seed: int,
        is_pose_image: bool,
    ) -> list[PIL.Image.Image]:
        control_image, vis_control_image = self.preprocess_pose(
            input_image=input_image,
            image_resolution=image_resolution,
            detect_resolution=detect_resolution,
            is_pose_image=is_pose_image,
        )
        self.load_controlnet_weight('pose')
        results = self.run_pipe(
            prompt=self.get_prompt(prompt, additional_prompt),
            negative_prompt=negative_prompt,
            control_image=control_image,
            num_images=num_images,
            num_steps=num_steps,
            guidance_scale=guidance_scale,
            seed=seed,
        )
        return [vis_control_image] + results

    @staticmethod
    def preprocess_seg(
        input_image: np.ndarray,
        image_resolution: int,
        detect_resolution: int,
        is_segmentation_map: bool,
    ) -> tuple[PIL.Image.Image, PIL.Image.Image]:
        input_image = HWC3(input_image)
        if not is_segmentation_map:
            with flow.mock_torch.disable():
                from annotator.uniformer import apply_uniformer
                control_image = apply_uniformer(
                    resize_image(input_image, detect_resolution))
                
            image = resize_image(input_image, image_resolution)
            H, W = image.shape[:2]
            control_image = cv2.resize(control_image, (W, H),
                                       interpolation=cv2.INTER_NEAREST)
        else:
            control_image = resize_image(input_image, image_resolution)
        return PIL.Image.fromarray(control_image), PIL.Image.fromarray(
            control_image)

    @flow.inference_mode()
    def process_seg(
        self,
        input_image: np.ndarray,
        prompt: str,
        additional_prompt: str,
        negative_prompt: str,
        num_images: int,
        image_resolution: int,
        detect_resolution: int,
        num_steps: int,
        guidance_scale: float,
        seed: int,
        is_segmentation_map: bool,
    ) -> list[PIL.Image.Image]:
        control_image, vis_control_image = self.preprocess_seg(
            input_image=input_image,
            image_resolution=image_resolution,
            detect_resolution=detect_resolution,
            is_segmentation_map=is_segmentation_map,
        )
        self.load_controlnet_weight('seg')
        results = self.run_pipe(
            prompt=self.get_prompt(prompt, additional_prompt),
            negative_prompt=negative_prompt,
            control_image=control_image,
            num_images=num_images,
            num_steps=num_steps,
            guidance_scale=guidance_scale,
            seed=seed,
        )
        return [vis_control_image] + results

    @staticmethod
    def preprocess_depth(
        input_image: np.ndarray,
        image_resolution: int,
        detect_resolution: int,
        is_depth_image: bool,
    ) -> tuple[PIL.Image.Image, PIL.Image.Image]:
        input_image = HWC3(input_image)
        if not is_depth_image:
            with flow.mock_torch.disable():
                from annotator.midas import apply_midas
                control_image, _ = apply_midas(
                    resize_image(input_image, detect_resolution))
                
            control_image = HWC3(control_image)
            image = resize_image(input_image, image_resolution)
            H, W = image.shape[:2]
            control_image = cv2.resize(control_image, (W, H),
                                       interpolation=cv2.INTER_LINEAR)
        else:
            control_image = resize_image(input_image, image_resolution)
        return PIL.Image.fromarray(control_image), PIL.Image.fromarray(
            control_image)

    @flow.inference_mode()
    def process_depth(
        self,
        input_image: np.ndarray,
        prompt: str,
        additional_prompt: str,
        negative_prompt: str,
        num_images: int,
        image_resolution: int,
        detect_resolution: int,
        num_steps: int,
        guidance_scale: float,
        seed: int,
        is_depth_image: bool,
    ) -> list[PIL.Image.Image]:
        control_image, vis_control_image = self.preprocess_depth(
            input_image=input_image,
            image_resolution=image_resolution,
            detect_resolution=detect_resolution,
            is_depth_image=is_depth_image,
        )
        self.load_controlnet_weight('depth')
        results = self.run_pipe(
            prompt=self.get_prompt(prompt, additional_prompt),
            negative_prompt=negative_prompt,
            control_image=control_image,
            num_images=num_images,
            num_steps=num_steps,
            guidance_scale=guidance_scale,
            seed=seed,
        )
        return [vis_control_image] + results

    @staticmethod
    def preprocess_normal(
        input_image: np.ndarray,
        image_resolution: int,
        detect_resolution: int,
        bg_threshold: float,
        is_normal_image: bool,
    ) -> tuple[PIL.Image.Image, PIL.Image.Image]:
        input_image = HWC3(input_image)
        if not is_normal_image:
            with flow.mock_torch.disable():
                from annotator.midas import apply_midas
                _, control_image = apply_midas(resize_image(
                    input_image, detect_resolution),
                                            bg_th=bg_threshold)
                
            control_image = HWC3(control_image)
            image = resize_image(input_image, image_resolution)
            H, W = image.shape[:2]
            control_image = cv2.resize(control_image, (W, H),
                                       interpolation=cv2.INTER_LINEAR)
        else:
            control_image = resize_image(input_image, image_resolution)
        return PIL.Image.fromarray(control_image), PIL.Image.fromarray(
            control_image)

    @flow.inference_mode()
    def process_normal(
        self,
        input_image: np.ndarray,
        prompt: str,
        additional_prompt: str,
        negative_prompt: str,
        num_images: int,
        image_resolution: int,
        detect_resolution: int,
        num_steps: int,
        guidance_scale: float,
        seed: int,
        bg_threshold: float,
        is_normal_image: bool,
    ) -> list[PIL.Image.Image]:
        control_image, vis_control_image = self.preprocess_normal(
            input_image=input_image,
            image_resolution=image_resolution,
            detect_resolution=detect_resolution,
            bg_threshold=bg_threshold,
            is_normal_image=is_normal_image,
        )
        self.load_controlnet_weight('normal')
        results = self.run_pipe(
            prompt=self.get_prompt(prompt, additional_prompt),
            negative_prompt=negative_prompt,
            control_image=control_image,
            num_images=num_images,
            num_steps=num_steps,
            guidance_scale=guidance_scale,
            seed=seed,
        )
        return [vis_control_image] + results
