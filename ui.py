import gradio as gr
from pipeline import DiffusionPipelineHandler, device_placement

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
            return (
                result_gallery,
                generation_info,
                html_info,
            )


def create_seed_inputs():
    with gr.Row():
        with gr.Box():
            with gr.Row(elem_id="seed_row"):
                seed = gr.Number(label="Seed", value=-1, precision=0)
                seed.style(container=False)
                random_seed = gr.Button(random_symbol, elem_id="random_seed")
                reuse_seed = gr.Button(reuse_symbol, elem_id="reuse_seed")

    random_seed.click(fn=lambda: -1, show_progress=False, inputs=[], outputs=[seed])

    return seed, reuse_seed


def create_toprow(is_img2img):
    id_part = "img2img" if is_img2img else "txt2img"

    with gr.Row(elem_id="toprow"):
        with gr.Column(scale=6):
            with gr.Row():
                with gr.Column(scale=80):
                    with gr.Row():
                        prompt = gr.Textbox(
                            label="Prompt",
                            elem_id=f"{id_part}_prompt",
                            show_label=False,
                            lines=2,
                            placeholder="Prompt (press Ctrl+Enter or Alt+Enter to generate)",
                        )

            with gr.Row():
                with gr.Column(scale=80):
                    with gr.Row():
                        negative_prompt = gr.Textbox(
                            label="Negative prompt",
                            elem_id=f"{id_part}_neg_prompt",
                            show_label=False,
                            lines=2,
                            placeholder="Negative prompt (press Ctrl+Enter or Alt+Enter to generate)",
                        )

        button_interrogate = None
        button_deepbooru = None
        if is_img2img:
            with gr.Column(scale=1, elem_id="interrogate_col"):
                button_interrogate = gr.Button(
                    "Interrogate\nCLIP", elem_id="interrogate"
                )
                button_deepbooru = gr.Button(
                    "Interrogate\nDeepBooru", elem_id="deepbooru"
                )

        with gr.Column(scale=1):
            submit = gr.Button(
                "Generate", elem_id=f"{id_part}_generate", variant="primary"
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


def create_ui():
    with gr.Blocks(analytics_enabled=False) as txt2img_interface:
        (
            prompt,
            negative_prompt,
            submit,
            button_interrogate,
            button_deepbooru,
        ) = create_toprow(is_img2img=False)

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
                    batch_count = gr.Slider(
                        minimum=1, step=1, label="Batch count", value=1
                    )
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

                seed, reuse_seed = create_seed_inputs()

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
            handler = DiffusionPipelineHandler(prompt,
            width,
            height,
            num_inference_steps,
            guidance_scale,
            negative_prompt,
            num_images_per_prompt,
            eta,
            seed,
            "pil",
            device_placement)
            imgs = handler()
            return imgs*2, "hello", "world"

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
    txt2img_interface.launch()


if __name__ == "__main__":
    create_ui()
