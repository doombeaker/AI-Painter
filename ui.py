import gradio as gr
import os
from pipeline import (
    DiffusionImg2ImgPipelineHandler,
    DiffusionPipelineHandler,
    device_placement,
)
from shared import cmd_opts
import shared
from control_net.cn_canny import create_demo as create_demo_canny
from control_net.cn_fake_scribble import create_demo as create_demo_fake_scribble
from control_net.cn_hed import create_demo as create_demo_hed
from control_net.cn_hough import create_demo as create_demo_hough
from control_net.cn_normal import create_demo as create_demo_normal
from control_net.cn_depth import create_demo as create_demo_depth
from control_net.cn_pose import create_demo as create_demo_pose
from control_net.cn_scribble import create_demo as create_demo_scribble
from control_net.cn_seg import create_demo as create_demo_seg
from model import Model
from PIL import Image

# Using constants for these since the variation selector isn't visible.
# Important that they exactly match script.js for tooltip to work.
random_symbol = "\U0001f3b2\ufe0f"  # 🎲️
reuse_symbol = "\u267b\ufe0f"  # ♻️
folder_symbol = "\U0001f4c2"  # 📂


def create_output_panel(tabname):
    with gr.Column(variant="panel"):
        with gr.Group():
            result_gallery = gr.Gallery(
                label="Output", show_label=False, elem_id=f"{tabname}_gallery"
            ).style(grid=4)

        generation_info = None
        with gr.Column():
            with gr.Group():
                html_info = gr.HTML()
                generation_info = gr.Textbox(visible=False)
                if tabname == "txt2img" or tabname == "img2img":
                    generation_info_button = gr.Button(
                        visible=False, elem_id=f"{tabname}_generation_info_button"
                    )
                else:
                    html_info_x = gr.HTML()
                    html_info = gr.HTML()

            return (result_gallery, generation_info, html_info)


def create_toprow(tabname):
    with gr.Row(elem_id="toprow"):
        with gr.Column(scale=6):
            with gr.Row():
                with gr.Column(scale=80):
                    with gr.Row():
                        prompt = gr.Textbox(
                            label="Prompt",
                            elem_id=f"{tabname}_prompt",
                            show_label=False,
                            lines=2,
                            placeholder="Prompt (press Ctrl+Enter or Alt+Enter to generate)",
                        )

            with gr.Row():
                with gr.Column(scale=80):
                    with gr.Row():
                        negative_prompt = gr.Textbox(
                            label="Negative prompt",
                            elem_id=f"{tabname}_neg_prompt",
                            show_label=False,
                            lines=2,
                            placeholder="Negative prompt (press Ctrl+Enter or Alt+Enter to generate)",
                        )

        button_interrogate = None
        button_deepbooru = None
        if tabname == "img2img":
            with gr.Column(scale=1, elem_id="interrogate_col"):
                button_interrogate = gr.Button(
                    "Interrogate\nCLIP", elem_id="interrogate"
                )
                button_deepbooru = gr.Button(
                    "Interrogate\nDeepBooru", elem_id="deepbooru"
                )

        with gr.Column(scale=1):
            submit = gr.Button(
                label="Generate", elem_id=f"{tabname}_generate", variant="primary"
            )
            submit.style(full_width=True)

    return prompt, negative_prompt, submit, button_interrogate, button_deepbooru


def setup_progressbar(progressbar, preview, id_part, textinfo=None):
    if textinfo is None:
        textinfo = gr.HTML(visible=False)

    check_progress = gr.Button(
        "Check progress", elem_id=f"{id_part}_check_progress", visible=False
    )
    check_progress.click(
        fn=lambda: check_progress_call(id_part),
        show_progress=False,
        inputs=[],
        outputs=[progressbar, preview, preview, textinfo],
    )

    check_progress_initial = gr.Button(
        "Check progress (first)",
        elem_id=f"{id_part}_check_progress_initial",
        visible=False,
    )
    check_progress_initial.click(
        fn=lambda: check_progress_call_initial(id_part),
        show_progress=False,
        inputs=[],
        outputs=[progressbar, preview, preview, textinfo],
    )


def mirror(x):
    return x


def create_ui():

    with gr.Blocks(analytics_enabled=False) as txt2img_interface:

        (
            prompt,
            negative_prompt,
            submit,
            button_interrogate,
            button_deepbooru,
        ) = create_toprow("txt2img")

        with gr.Row(elem_id="txt2img_progress_row"):
            with gr.Column(scale=1):
                pass

            with gr.Column(scale=1):
                progressbar = gr.HTML(elem_id="txt2img_progressbar")
                txt2img_preview = gr.Image(elem_id="txt2img_preview", visible=False)
                setup_progressbar(progressbar, txt2img_preview, "txt2img")

        with gr.Row().style(equal_height=False):
            with gr.Column(variant="panel"):
                steps = gr.Slider(
                    minimum=1, maximum=150, step=1, label="Inference Steps", value=25
                )

                with gr.Group():
                    width = gr.Slider(
                        minimum=64, maximum=2048, step=64, label="Width", value=768
                    )
                    height = gr.Slider(
                        minimum=64, maximum=2048, step=64, label="Height", value=768
                    )

                with gr.Row():
                    batch_size = gr.Slider(
                        minimum=1, maximum=8, step=1, label="Images per Prompt", value=1
                    )

                guidance_scale = gr.Slider(
                    minimum=1.0,
                    maximum=30.0,
                    step=0.5,
                    label="Guidance Scale",
                    value=7.5,
                )

                eta = gr.Slider(
                    minimum=0.0,
                    maximum=5.0,
                    step=0.1,
                    label="eta",
                    value=0.0,
                )

                with gr.Row():
                    with gr.Box():
                        with gr.Row(elem_id="seed_row"):
                            seed = gr.Number(label="Seed", value=-1, precision=0)
                            seed.style(container=False)
                            random_seed = gr.Button(
                                random_symbol, elem_id="random_seed"
                            )

                random_seed.click(
                    fn=lambda: -1, show_progress=False, inputs=[], outputs=[seed]
                )

            txt2img_gallery, generation_info, html_info = create_output_panel("txt2img")

        def run_diffusers_pipeline(
            prompt: str,
            width: int = 768,
            height: int = 768,
            num_inference_steps: int = 25,
            guidance_scale: float = 7.5,
            negative_prompt: str = None,
            num_images_per_prompt: int = 1,
            eta=0.0,
            seed: int = -1,
        ):
            handler = DiffusionPipelineHandler(
                prompt,
                width,
                height,
                num_inference_steps,
                guidance_scale,
                negative_prompt,
                num_images_per_prompt,
                eta,
                seed,
                "pil",
                device_placement,
            )
            imgs = handler()
            shared.txt2img_result = imgs
            return imgs, "", prompt

        submit.click(
            fn=run_diffusers_pipeline,
            inputs=[
                prompt,
                width,
                height,
                steps,
                guidance_scale,
                negative_prompt,
                batch_size,
                eta,
                seed,
            ],
            outputs=[
                txt2img_gallery,
                generation_info,
                html_info,
            ],
        )
    with gr.Blocks(analytics_enabled=False) as img2img_interface:
        (
            img2img_prompt,
            img2img_negative_prompt,
            submit,
            button_interrogate,
            button_deepbooru,
        ) = create_toprow("img2img")

        with gr.Row(elem_id="img2img_progress_row"):
            with gr.Column(scale=1):
                pass

            with gr.Column(scale=1):
                progressbar = gr.HTML(elem_id="img2img_progressbar")
                img2img_preview = gr.Image(elem_id="img2img_preview", visible=False)
                setup_progressbar(progressbar, img2img_preview, "img2img")

        with gr.Row().style(equal_height=False):
            with gr.Column(variant="panel"):

                with gr.Tabs(elem_id="mode_img2img") as tabs_img2img_mode:
                    with gr.TabItem("img2img", id="img2img"):
                        init_img = gr.Image(
                            label="Image for img2img",
                            elem_id="img2img_image",
                            show_label=False,
                            source="upload",
                            interactive=True,
                            type="pil",
                        ).style(height=480)

                steps = gr.Slider(
                    minimum=1, maximum=150, step=1, label="Sampling Steps", value=20
                )
                guidance_scale = gr.Slider(
                    minimum=1.0,
                    maximum=30.0,
                    step=0.5,
                    label="Guidance Scale",
                    value=7.5,
                )

                eta = gr.Slider(
                    minimum=0.0,
                    maximum=5.0,
                    step=0.1,
                    label="eta",
                    value=0.0,
                )

                with gr.Group():
                    width = gr.Slider(
                        minimum=64,
                        maximum=2048,
                        step=64,
                        label="Width",
                        value=768,
                        elem_id="img2img_width",
                    )
                    height = gr.Slider(
                        minimum=64,
                        maximum=2048,
                        step=64,
                        label="Height",
                        value=768,
                        elem_id="img2img_height",
                    )

                with gr.Group():
                    denoising_strength = gr.Slider(
                        minimum=0.0,
                        maximum=1.0,
                        step=0.01,
                        label="Denoising strength",
                        value=0.75,
                    )

                with gr.Row():
                    batch_size = gr.Slider(
                        minimum=1, maximum=8, step=1, label="Images per Prompt", value=1
                    )
                with gr.Row():
                    with gr.Box():
                        with gr.Row(elem_id="seed_row"):
                            seed = gr.Number(label="Seed", value=-1, precision=0)
                            seed.style(container=False)
                            random_seed = gr.Button(
                                random_symbol, elem_id="random_seed"
                            )

            (
                img2img_gallery,
                img2img_generation_info,
                img2img_html_info,
            ) = create_output_panel("img2img")

        def run_diffusers_img2img_pipeline(
            prompt: str,
            image: Image.Image,
            strength: float = 0.8,
            width: int = 768,
            height: int = 768,
            num_inference_steps: int = 25,
            guidance_scale: float = 7.5,
            negative_prompt: str = None,
            num_images_per_prompt: int = 1,
            eta=0.0,
            seed: int = -1,
        ):
            if image is None:
                imgs = None
                prompt = "Please drop image"
            else:
                handler = DiffusionImg2ImgPipelineHandler(
                    prompt,
                    image,
                    strength,
                    width,
                    height,
                    num_inference_steps,
                    guidance_scale,
                    negative_prompt,
                    num_images_per_prompt,
                    eta,
                    seed,
                    "pil",
                    device_placement,
                )
                imgs = handler()
            return imgs, "", prompt

        submit.click(
            fn=run_diffusers_img2img_pipeline,
            inputs=[
                img2img_prompt,
                init_img,
                denoising_strength,
                width,
                height,
                steps,
                guidance_scale,
                img2img_negative_prompt,
                batch_size,
                eta,
                seed,
            ],
            outputs=[
                img2img_gallery,
                img2img_generation_info,
                img2img_html_info,
            ],
        )

    with gr.Blocks(analytics_enabled=False) as control_net:
        model = Model(base_model_id="runwayml/stable-diffusion-v1-5", task_name='canny')
        with gr.Tabs():

            with gr.TabItem('Canny'):
                create_demo_canny(model.process_canny,
                                max_images=3,
                                default_num_images=1)
            with gr.TabItem('Hough'):
                create_demo_hough(model.process_hough,
                                max_images=3,
                                default_num_images=1)
            with gr.TabItem('HED'):
                create_demo_hed(model.process_hed,
                                max_images=3,
                                default_num_images=1)
            with gr.TabItem('Scribble'):
                create_demo_scribble(model.process_scribble,
                                    max_images=3,
                                    default_num_images=1)
            with gr.TabItem('Fake Scribble'):
                create_demo_fake_scribble(model.process_fake_scribble,
                                        max_images=3,
                                        default_num_images=1)
            with gr.TabItem('Pose'):
                create_demo_pose(model.process_pose,
                                max_images=3,
                                default_num_images=1)
            with gr.TabItem('Segmentation'):
                create_demo_seg(model.process_seg,
                                max_images=3,
                                default_num_images=1)
            with gr.TabItem('Depth'):
                create_demo_depth(model.process_depth,
                                max_images=3,
                                default_num_images=1)
            with gr.TabItem('Normal map'):
                create_demo_normal(model.process_normal,
                                max_images=3,
                                default_num_images=1)
                
    interfaces = [
        (txt2img_interface, "txt2img", "txt2img"),
        (img2img_interface, "img2img", "img2img"),
        (control_net, "controlnet", "controlnet"),
    ]
    with gr.Blocks(analytics_enabled=False) as launch_interface:

        with gr.Tabs(elem_id="tabs") as tabs:
            for interface, label, ifid in interfaces:
                with gr.TabItem(label, id=ifid, elem_id="tab_" + ifid):
                    interface.render()

    launch_interface.launch(
        share=cmd_opts.share, server_name=cmd_opts.ip, server_port=cmd_opts.port
    )


if __name__ == "__main__":
    create_ui()
