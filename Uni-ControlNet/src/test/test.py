import sys
if "./" not in sys.path:
    sys.path.append("./")
from utils.share import *
import utils.config as config

import cv2
import einops
import io
import gradio as gr
import numpy as np
import os, traceback
import torch
from pytorch_lightning import seed_everything
print(f"Gradio version: {gr.__version__}")  # debug: verify 4.44.x

import logging
logger = logging.getLogger()
logger.setLevel(logging.INFO)

# Remove old handlers if any
for h in list(logger.handlers):
    logger.removeHandler(h)

fh = logging.FileHandler("gradio_debug.log", mode="w", encoding="utf-8")
formatter = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")
fh.setFormatter(formatter)
logger.addHandler(fh)
ch = logging.StreamHandler()
ch.setFormatter(formatter)
logger.addHandler(ch)

logging.info("Logger initialized.")

from annotator.util import resize_image, HWC3
from annotator.canny import CannyDetector
from annotator.mlsd import MLSDdetector
from annotator.hed import HEDdetector
from annotator.sketch import SketchDetector
from annotator.openpose import OpenposeDetector
from annotator.midas import MidasDetector
from annotator.uniformer import UniformerDetector
from annotator.content import ContentDetector
from annotator.eeg import EEGDetector

from models.util import create_model, load_state_dict
from models.ddim_hacked import DDIMSampler

logging.info("Creating EEG Detector")
apply_content = EEGDetector()

model = create_model("./configs/global_v21.yaml").cpu()
model.load_state_dict(load_state_dict("./log_global/lightning_logs/version_9/checkpoints/epoch=4-step=100000-006.ckpt", location="cuda"))
model = model.cuda()
ddim_sampler = DDIMSampler(model)


def _file_to_path(f):
    if f is None:
        return None
    if isinstance(f, dict) and "name" in f:
        return f["name"]
    if hasattr(f, "name"):
        return f.name
    if isinstance(f, str):
        return f
    return None


def _load_npy_from_any(x):
    try:
        if x is None:
            return None
        if isinstance(x, bytes):
            return np.load(io.BytesIO(x), allow_pickle=True)
        p = _file_to_path(x)
        if p and os.path.isfile(p):
            return np.load(p, allow_pickle=True)
    except Exception as e:
        logging.info(f"[content] load failed: {e}")
    return None


def process(
    canny_image,
    mlsd_image,
    hed_image,
    sketch_image,
    openpose_image,
    midas_image,
    seg_image,
    content_path,
    prompt,
    a_prompt,
    n_prompt,
    num_samples,
    image_resolution,
    ddim_steps,
    strength,
    scale,
    seed,
    eta,
    low_threshold,
    high_threshold,
    value_threshold,
    distance_threshold,
    alpha,
    global_strength,
):
    logging.info("[process] start")
    seed_everything(seed)
    anchor_image = np.zeros((image_resolution, image_resolution, 3), np.uint8)

    try:
        H, W, C = resize_image(HWC3(anchor_image), image_resolution).shape
        with torch.no_grad():
            canny_detected_map = np.zeros((H, W, C), np.uint8)
            mlsd_detected_map = np.zeros((H, W, C), np.uint8)
            hed_detected_map = np.zeros((H, W, C), np.uint8)
            sketch_detected_map = np.zeros((H, W, C), np.uint8)
            openpose_detected_map = np.zeros((H, W, C), np.uint8)
            midas_detected_map = np.zeros((H, W, C), np.uint8)
            seg_detected_map = np.zeros((H, W, C), np.uint8)

            logging.info(f"[content] loading from {content_path}")
            arr = _load_npy_from_any(content_path)
            if arr is not None:
                logging.info(f"[content] loaded npy shape={arr.shape}, dtype={arr.dtype}")
                vec = np.asarray(arr, np.float32)#.reshape(-1)
                content_emb = apply_content(vec) #if vec.size == 1024 else np.zeros((1024,), np.float32)

            else:
                logging.info("[content] no file or failed to load; using zeros")
                content_emb = np.zeros((1024,), np.float32)
            

            detected_maps_list = [
                canny_detected_map,
                mlsd_detected_map,
                hed_detected_map,
                sketch_detected_map,
                openpose_detected_map,
                midas_detected_map,
                seg_detected_map,
            ]
            detected_maps = np.concatenate(detected_maps_list, axis=2)

            local_control = torch.from_numpy(detected_maps.copy()).float().cuda() / 255.0
            local_control = torch.stack([local_control for _ in range(num_samples)], 0)
            local_control = einops.rearrange(local_control, "b h w c -> b c h w").clone()
            global_control = torch.from_numpy(content_emb.copy()).float().cuda().clone()
            global_control = torch.stack([global_control for _ in range(num_samples)], 0)

            if config.save_memory:
                model.low_vram_shift(is_diffusing=False)
            
            uc_local_control = local_control
            uc_global_control = torch.zeros_like(global_control)
            cond = {
                "local_control": [local_control],
                "c_crossattn": [
                    model.get_learned_conditioning([prompt + ", " + a_prompt] * num_samples)
                ],
                "global_control": [global_control],
            }
            un_cond = {
                "local_control": [uc_local_control],
                "c_crossattn": [model.get_learned_conditioning([n_prompt] * num_samples)],
                "global_control": [uc_global_control],
            }
            shape = (4, H // 8, W // 8)

            if config.save_memory:
                model.low_vram_shift(is_diffusing=True)

            model.control_scales = [strength] * 13
            samples, _ = ddim_sampler.sample(
                ddim_steps,
                num_samples,
                shape,
                cond,
                verbose=False,
                eta=eta,
                unconditional_guidance_scale=scale,
                unconditional_conditioning=un_cond,
                global_strength=global_strength,
            )

            if config.save_memory:
                model.low_vram_shift(is_diffusing=False)

            x_samples = model.decode_first_stage(samples)
            x_samples = (
                einops.rearrange(x_samples, "b c h w -> b h w c") * 127.5 + 127.5
            ).cpu().numpy().clip(0, 255).astype(np.uint8)
            results = [x_samples[i] for i in range(num_samples)]

        logging.info("[process] done")
        return [results, detected_maps_list]
    except Exception:
        print("[process] exception:")
        traceback.print_exc()
        return [[], []]


# ---------------- UI ----------------
with gr.Blocks(title="Uni-ControlNet Demo") as demo:
    gr.Markdown("## Uni-ControlNet Demo")

    with gr.Row():
        canny_image = gr.Image(sources=["upload"], type="numpy", label="canny")
        mlsd_image = gr.Image(sources=["upload"], type="numpy", label="mlsd")
        hed_image = gr.Image(sources=["upload"], type="numpy", label="hed")
        sketch_image = gr.Image(sources=["upload"], type="numpy", label="sketch")

    with gr.Row():
        openpose_image = gr.Image(sources=["upload"], type="numpy", label="openpose")
        midas_image = gr.Image(sources=["upload"], type="numpy", label="midas")
        seg_image = gr.Image(sources=["upload"], type="numpy", label="seg")
        content_path = gr.File(file_types=[".npy"], label="eeg (.npy)")

    content_status = gr.Markdown("No file.")
    prompt = gr.Textbox(label="Prompt")
    run_button = gr.Button("Run")

    with gr.Accordion("Advanced options", open=False):
        num_samples = gr.Slider(1, 12, value=4, step=1, label="Images")
        image_resolution = gr.Slider(256, 768, value=512, step=64, label="Image Resolution")
        strength = gr.Slider(0.0, 2.0, value=1.0, step=0.01, label="Control Strength")
        global_strength = gr.Slider(0.0, 2.0, value=1.0, step=0.01, label="Global Strength")
        low_threshold = gr.Slider(1, 255, value=100, step=1, label="Canny Low Threshold")
        high_threshold = gr.Slider(1, 255, value=200, step=1, label="Canny High Threshold")
        value_threshold = gr.Slider(0.01, 2.0, value=0.1, step=0.01, label="Hough Value Threshold (MLSD)")
        distance_threshold = gr.Slider(0.01, 20.0, value=0.1, step=0.01, label="Hough Distance Threshold (MLSD)")
        alpha = gr.Slider(0.1, 20.0, value=6.2, step=0.01, label="Alpha")
        ddim_steps = gr.Slider(1, 100, value=50, step=1, label="Steps")
        scale = gr.Slider(0.1, 30.0, value=7.5, step=0.1, label="Guidance Scale")
        seed = gr.Slider(-1, 2147483647, value=42, step=1, label="Seed")
        eta = gr.Number(value=0.0, label="Eta (DDIM)")
        a_prompt = gr.Textbox(value="best quality, extremely detailed", label="Added Prompt")
        n_prompt = gr.Textbox(
            value="longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality",
            label="Negative Prompt",
        )

    image_gallery = gr.Gallery(label="Output", columns=4, height="auto", show_label=False)
    cond_gallery = gr.Gallery(label="Detected Maps", columns=4, height="auto", show_label=False)

    inputs = [
        canny_image,
        mlsd_image,
        hed_image,
        sketch_image,
        openpose_image,
        midas_image,
        seg_image,
        content_path,
        prompt,
        a_prompt,
        n_prompt,
        num_samples,
        image_resolution,
        ddim_steps,
        strength,
        scale,
        seed,
        eta,
        low_threshold,
        high_threshold,
        value_threshold,
        distance_threshold,
        alpha,
        global_strength,
    ]

    run_button.click(process, inputs=inputs, outputs=[image_gallery, cond_gallery])

    def _update_content_status(x):
        arr = _load_npy_from_any(x)
        if arr is None:
            return "Failed to load .npy"
        return f"Loaded: shape={list(arr.shape)}, dtype={arr.dtype}"

    content_path.change(_update_content_status, inputs=content_path, outputs=content_status)

demo.launch(server_name="0.0.0.0", server_port=7860, show_api=True)
