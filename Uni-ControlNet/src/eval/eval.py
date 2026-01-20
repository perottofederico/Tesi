
import sys

if "./" not in sys.path:
    sys.path.append("./")
from annotator.eeg.thingseeg.ATMS import ATMS
from annotator.eeg.eegcvpr40.LSTM import EEGFeatNet
import argparse
import json
import logging
from pathlib import Path

import numpy as np
import torch
import pickle
import torch.nn as nn
import einops
from PIL import Image
from pytorch_lightning import seed_everything
import torch.nn.functional as F
from torchvision import transforms
from torchvision.transforms import InterpolationMode
from torchvision.models.feature_extraction import create_feature_extractor
from torchvision.models import (
    alexnet,
    AlexNet_Weights,
    inception_v3,
    Inception_V3_Weights,
    efficientnet_b1,
    EfficientNet_B1_Weights,
)
import clip
import torch.hub
from skimage.color import rgb2gray
from skimage.metrics import structural_similarity as skimage_ssim
from scipy.spatial import distance

# Project imports
from utils.share import *
import utils.config as config
from annotator.eeg.eegcvpr40 import EEGDetector
from annotator.eeg.thingseeg import EEGDetectorTHINGS
from annotator.eeg.eegcvpr40.eegcvpr_id_to_caption import id_to_caption
from models.util import create_model, load_state_dict
from models.ddim_hacked import DDIMSampler

# Metrics
try:
    from torchmetrics.image.fid import FrechetInceptionDistance
    from torchmetrics.image.inception import InceptionScore
    from torchmetrics.image.kid import KernelInceptionDistance
    from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity
    from torchmetrics.image.ssim import StructuralSimilarityIndexMeasure
except Exception as ex:
    print("Missing torchmetrics. Install with: pip install torchmetrics")
    raise

# ------------- Logging -------------
logger = logging.getLogger("eval")
logger.setLevel(logging.INFO)
ch = logging.StreamHandler()
ch.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))
logger.addHandler(ch)



# ------------- Helpers -------------
def _to_chw_float01(np_img):
    # np_img: HxWx{1,3,4} uint8 -> torch.float32 [0,1], CxHxW
    t = torch.from_numpy(np_img)
    if t.ndim == 2:
        t = t.unsqueeze(-1).repeat(1, 1, 3)
    if t.shape[-1] == 4:
        t = t[..., :3]
    t = t.permute(2, 0, 1).contiguous().float() / 255.0
    return t

def _to_chw_uint8(np_img):
    # np_img: HxWx{1,3,4} uint8 -> torch.uint8 CxHxW
    t = torch.from_numpy(np_img)
    if t.ndim == 2:
        t = t.unsqueeze(-1).repeat(1, 1, 3)
    if t.shape[-1] == 4:
        t = t[..., :3]
    return t.permute(2, 0, 1).contiguous().to(torch.uint8)

def _load_npy(path):
    try:
        return np.load(str(path), allow_pickle=True)
    except Exception as e:
        logger.warning(f"Failed to load npy: {path} ({e})")
        return None

def pair_dataset_items(eeg_dir: Path, img_dir: Path):
    # Pair EEG npy with JPG image by filename stem
    eeg_files = sorted(eeg_dir.glob("*.npy"))
    img_files = {p.stem: p for p in img_dir.glob("*.jpg")}
    pairs = []
    for eeg in eeg_files:
        img = img_files.get(eeg.stem)
        if img is None:
            logger.warning(f"No matching image for {eeg.name}")
            continue
        pairs.append((eeg, img))
    logger.info(f"Found {len(pairs)} EEG-image pairs")
    return pairs

def _find_generated(out_samples: Path, stem: str):
    # Look for existing generated image by stem (png preferred, then jpg)
    p_png = out_samples / f"{stem}.png"
    if p_png.exists():
        return p_png
    p_jpg = out_samples / f"{stem}.jpg"
    if p_jpg.exists():
        return p_jpg
    return None

def volume_computation2(language, video):
    batch_size1 = language.shape[0]
    batch_size2 = video.shape[0]

    ll = torch.einsum('bi,bi->b', language, language).unsqueeze(1).expand(-1, batch_size2)
    vv = torch.einsum('bi,bi->b', video, video).unsqueeze(0).expand(batch_size1, -1)
    lv = language@video.T

    G = torch.stack([
        torch.stack([ll, lv], dim=-1),  # First row of the Gram matrix
        torch.stack([lv, vv], dim=-1),  # Second row of the Gram matrix
    ], dim=-2)

    # Compute the determinant for each Gram matrix (shape: [batch_size1, batch_size2])
    gram_det = torch.det(G.float())

    # Compute the square root of the absolute value of the determinants
    res = torch.sqrt(torch.abs(gram_det))
    return res

def load_things_eeg_classifier(ckpt_path: Path, device):
    print(f"Loading EEG classifier from {ckpt_path}...")
    encoder = ATMS().to(device)
    head = nn.Linear(encoder.proj_eeg[0].out_features, 27).to(device)
    ckpt = torch.load(ckpt_path, map_location=device)
    missing, unexpected = encoder.load_state_dict(ckpt["eeg_encoder"], strict=False)
    print(f"EEG Classifier Encoder loaded. Missing:  {missing},\nUnexpected: {unexpected}")
    missing, unexpected = head.load_state_dict(ckpt["classifier_head"], strict=False)
    print(f"EEG Classifier Head loaded. Missing:  {missing},\nUnexpected: {unexpected}")
    encoder.eval().requires_grad_(False)
    head.eval().requires_grad_(False)
    return encoder, head

def load_eegcvpr_eeg_classifier(ckpt_path: Path, device):
    print(f"Loading EEG classifier from {ckpt_path}...")
    encoder = EEGFeatNet(n_features=128, projection_dim=128, num_layers=4).to("cuda")
    encoder = encoder.to(device)
    encoder = torch.nn.DataParallel(encoder).to("cuda")
    ckpt = torch.load(ckpt_path, map_location=device)
    missing, unexpected = encoder.load_state_dict(ckpt["model_state_dict"])
    print(f"EEG Classifier Encoder loaded. Missing:  {missing},\nUnexpected: {unexpected}")
    encoder.eval().requires_grad_(False)
    return encoder
        
# ---------------- Additional evaluation utilities ----------------
@torch.no_grad()
def _batched_encode(images, model, preprocess, device, feature_layer=None, batch_size=32):
    """Encode images with a model in batches, optionally extracting a submodule output."""
    feats = []
    for start in range(0, len(images), batch_size):
        end = min(start + batch_size, len(images))
        batch = torch.stack([preprocess(img) for img in images[start:end]], dim=0).to(device)
        out = model(batch)
        if feature_layer is not None:
            out = out[feature_layer]
        out = out.float().flatten(1).cpu()
        feats.append(out)
    return torch.cat(feats, dim=0)


@torch.no_grad()
def two_way_identification(all_brain_recons, all_images, model, preprocess, device, feature_layer=None):
    """2-way identification as in notebook; returns mean percent correct."""
    preds = _batched_encode(all_brain_recons, model, preprocess, device, feature_layer)
    reals = _batched_encode(all_images, model, preprocess, device, feature_layer)

    preds_np = preds.numpy()
    reals_np = reals.numpy()

    # Correlation matrix across concatenated samples
    r = np.corrcoef(reals_np, preds_np)
    #slice the matrix to get real images on rows and generated on columns
    r = r[: len(all_images), len(all_images) :]
    # correct pairings are on the diagonal
    congruents = np.diag(r)
    #compare 
    success = r < congruents
    success_cnt = np.sum(success, axis=0)

    if len(all_images) <= 1:
        return float("nan")
    perf = np.mean(success_cnt) / (len(all_images) - 1)
    return float(perf)




def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--encoder", type=str, required=True, choices=["LSTM", "LSTM_crossatt", "GWIT", "THINGS_ATMS", "THINGS_LSTM_crossatt"],
                    help=" EEG encoder used to precompute the eeg embeddings.")
    ap.add_argument("--model_ckpt", type=str,  help="UniControl-Net checkpoint used for generation.")
    ap.add_argument("--eeg_ckpt", type=str,  help="EEG encoder checkpoint used to precompute the embeddings.")
    ap.add_argument("--eeg_dir", type=str, default="data_THINGSEEG/conditions/eeg/validation",
                    help="Directory containing EEG .npy files")
    ap.add_argument("--img_dir", type=str, default="data_THINGSEEG/images/validation",
                    help="Directory containing real .jpg images with matching stems")
    ap.add_argument("--out_dir", type=str, default="4.ATMS_sub08_str2_cfg9_outputs/", help="Output directory (samples saved to out_dir/samples)")
    ap.add_argument("--skip_generate", action="store_true",
                    help="Do not generate images; use already generated images in out_dir/samples")
    ap.add_argument("--use_retrieval_prompt", action="store_true" , help="Wheter to use prompts (obtained by doing retrieval on the test set) or not")
    ap.add_argument("--use_classification_prompt", action="store_true" , help="Wheter to use classification-based prompts or not")
    ap.add_argument("--text_only", action="store_true" , help="Wheter to use only text (no eeg) for generation")
    ap.add_argument("--a_prompt", type=str, default="")#"best quality, extremely detailed", help="Added prompt")
    ap.add_argument("--n_prompt", type=str, default="")#"longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality", help="Negative prompt")
    ap.add_argument("--image_resolution", type=int, default=256)
    ap.add_argument("--ddim_steps", type=int, default=50)
    ap.add_argument("--strength", type=float, default=1.0)
    ap.add_argument("--global_strength", type=float, default=2.0, help="Strength for the eeg condition")
    ap.add_argument("--scale", type=float, default=9.0)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--eta", type=float, default=0.0)
    ap.add_argument("--limit", type=int, default=0, help="Limit number of pairs (0=all)")
    args = ap.parse_args()

    eeg_dir = "data_THINGSEEG/conditions/eeg/validation" if args.encoder in ["THINGS_ATMS", "THINGS_LSTM_crossatt"] \
            else "data_EEGCVPR/conditions/eeg/validation"
    img_dir = "data_THINGSEEG/images/validation" if args.encoder in ["THINGS_ATMS", "THINGS_LSTM_crossatt"] \
            else "data_EEGCVPR/images/validation"

    eeg_dir = Path(eeg_dir)
    img_dir = Path(img_dir)
    if not eeg_dir.exists() or not img_dir.exists():
        logger.error(f"Dirs not found. eeg_dir={eeg_dir}, img_dir={img_dir}")
        return

    out_root = Path(args.out_dir)
    out_samples = out_root / "samples"
    out_root.mkdir(parents=True, exist_ok=True)
    out_samples.mkdir(parents=True, exist_ok=True)

    # Seed
    if args.seed >= 0:
        seed_everything(args.seed)

    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Device: {device}")

    # Load model if generating
    if not args.skip_generate:
        logger.info("Loading model and EEG detector...")
        if args.encoder == "THINGS_ATMS" or args.encoder == "THINGS_LSTM_crossatt":
            apply_content = EEGDetectorTHINGS(args.encoder, ckpt_path=args.eeg_ckpt)
            model = create_model("./configs/global_v21_thingseeg.yaml").cpu()
            #model.load_state_dict(
            #    load_state_dict("./log_thingseeg_ATMSsub8_str2cfg9/lightning_logs/version_1/checkpoints/epoch=14-step=495000.ckpt", location="cuda")
            #)
        elif args.encoder in ["LSTM", "LSTM_crossatt", "GWIT"]:
            apply_content = EEGDetector(encoder_type=args.encoder, ckpt_path=args.eeg_ckpt)
            model = create_model("./configs/global_v21_eegcvpr40.yaml").cpu()
            #model.load_state_dict(
            #    load_state_dict("./log_thingseeg_LSTMsub8_str2cfg9/lightning_logs/version_0/checkpoints/epoch=14-step=495000.ckpt", location="cuda")
            #)
        model.load_state_dict(load_state_dict(args.model_ckpt, location="cuda"))
        model = model.to(device).eval()
        ddim_sampler = DDIMSampler(model)
    else:
        apply_content = None
        model = None
        ddim_sampler = None

    # Prepare metrics
    # FID & IS expect uint8 CxHxW tensors (torch-fidelity backend handles resize/preprocess)
    fid = FrechetInceptionDistance(feature=2048).to(device)
    isc = InceptionScore(splits=10).to(device)
    # KID: default setup; torchmetrics handles preprocessing
    # KID uses the same Inception backend; with this torchmetrics build it expects uint8 inputs
    kid = KernelInceptionDistance(subset_size=50).to(device)
    # LPIPS and SSIM: operate on float images, same spatial size
    lpips = LearnedPerceptualImagePatchSimilarity(net_type="alex", reduction="mean").to(device)
    ssim = StructuralSimilarityIndexMeasure(data_range=1.0).to(device)

    # CLIP ViT-L/14 for cosine-based ranking and reuse in additional metrics
    clip_model, clip_preprocess_pil = clip.load("ViT-L/14", device=device)
    clip_model.eval().requires_grad_(False)
    clip_preprocess_tensor = transforms.Compose([
        transforms.Resize(224, interpolation=InterpolationMode.BILINEAR),
        transforms.Normalize(mean=[0.48145466, 0.4578275, 0.40821073], std=[0.26862954, 0.26130258, 0.27577711]),
    ])

    # Accumulators for storing images for additional metrics
    gen_imgs = []  # list of torch.float32 CHW in [0,1]
    real_imgs = []

    # Per-sample cosine similarity ranking
    cosine_rankings = []

    # Pair dataset
    pairs = pair_dataset_items(eeg_dir, img_dir)
    if args.limit and args.limit > 0:
        pairs = pairs[: args.limit]
    if not pairs:
        logger.error("No EEG-image pairs found.")
        return

    H = W = int(args.image_resolution)

    # Accumulators for per-pair metrics
    lpips_sum = 0.0
    ssim_sum = 0.0
    processed = 0
    
    #TODO generalize this for EEGCVPR too
    if args.use_retrieval_prompt:
        caption_embs_list = []
        captions = []
        with open("data_THINGSEEG/anno_test.txt", "r", encoding="utf-8") as f:
            for line in f:
                file_id, caption = line.strip().split("\t")
                emb_path = Path(f"data_THINGSEEG/text_features/{file_id}.npy")
                emb = _load_npy(emb_path)
                if emb is not None:
                    caption_embs_list.append(emb.astype(np.float32))
                    captions.append(caption)
        caption_embs = torch.from_numpy(np.stack(caption_embs_list, axis=0)).to(device)  # [N, D]
    
    #TODO generalize this for EEGCVPR too
    if args.use_classification_prompt:
        if args.encoder in ["THINGS_ATMS", "THINGS_LSTM_crossatt"]: #THINGS
            classifier, head = load_things_eeg_classifier(Path("annotator/ckpts/things_eeg_classifier.pt"), device)
        else: #EEGCVPR
            classifier = load_eegcvpr_eeg_classifier(Path("annotator/ckpts/eegcvpr_eeg_classifier.pth"), device)
            pkl_path = "annotator/ckpts/knn_model.pkl"
            with open(pkl_path, 'rb') as f:
                head = pickle.load(f)
    # Loop
    for i, (eeg_path, real_path) in enumerate(pairs):
        try:
            # Load or generate predicted image
            if args.skip_generate:
                gen_path = _find_generated(out_samples, eeg_path.stem)
                if gen_path is None:
                    logger.warning(f"Missing generated image for {eeg_path.stem} in {out_samples}; skipping")
                    continue
                x = np.array(Image.open(gen_path).convert("RGB"), dtype=np.uint8)
            else:
                if args.seed >= 0:
                    seed_everything(args.seed + i)
                if args.text_only:
                    content_emb = torch.zeros((1024,), dtype=torch.float32, device=device).unsqueeze(0)
                else:
                    arr = _load_npy(eeg_path)
                    if arr is None:
                        continue
                    vec = np.asarray(arr, np.float32)
                    
                    #only use sub_id if not using LSTM dor LSTM crossatt
                    if args.encoder in ["THINGS_ATMS", "GWIT", "THINGS_LSTM_crossatt"]:
                        sub_id = torch.tensor([int(eeg_path.stem.split('_')[-1])], dtype=torch.long)
                        content_emb = apply_content(vec, sub_id) 
                    else: #LSTM or LSTM_crossatt
                        content_emb = apply_content(vec)  
                    if isinstance(content_emb, np.ndarray):
                        content_emb = torch.from_numpy(content_emb)
                    content_emb = content_emb.to(device=device, dtype=torch.float32)
                    if content_emb.dim() == 1:
                        content_emb = content_emb.unsqueeze(0)
                
                #TODO generalize this for EEGCVPR too
                ## do retrieval to get prompt if specified
                if args.use_retrieval_prompt:
                    print("Doing retrieval to get prompt...")
                    volume = volume_computation2(content_emb, caption_embs)[0] #.squeeze(0)
                    best_idx = int(torch.argmin(volume).item())
                    best_caption = captions[best_idx]
                    prompt = best_caption
                    logger.info(f"Using retrieved prompt for {eeg_path.stem}: {prompt}")
                
                elif args.use_classification_prompt:
                    #THINGS
                    if args.encoder in ["THINGS_ATMS", "THINGS_LSTM_crossatt"]:
                        print("Doing classification to get prompt...")
                        CATEGORIES = [
                            "food", "animal", "clothing", "tool", "sports equipment", "vegetable",
                            "vehicle", "musical instrument", "fruit", "body part", "dessert", "toy",
                            "container", "part of car", "weapon", "bird", "furniture", "kitchen tool",
                            "office supply", "clothing accessory", "kitchen appliance", "plant",
                            "insect", "home décor", "medical equipment", "electronic device", "drink",
                        ]
                        eeg_tensor = torch.from_numpy(vec).float().unsqueeze(0).to(device)
                        sub_tensor = sub_id.to(device)
                        with torch.no_grad():
                            feats = classifier(eeg_tensor, sub_tensor)
                            logits = head(feats)
                            cls_idx = int(logits.argmax(dim=1).item())
                        prompt = f"a photo of a {CATEGORIES[cls_idx]}"
                        logger.info(f"Using class prompt for {eeg_path.stem}: {prompt}")
                    #EEGCVPR
                    if args.encoder in ["LSTM", "LSTM_crossatt", "GWIT"]:
                        eeg_tensor = torch.from_numpy(vec).float().unsqueeze(0).to(device)
                        with torch.no_grad():
                            x_proj = classifier(eeg_tensor.view(-1,eeg_tensor.shape[2],eeg_tensor.shape[1]))[0]
                            predicted_label = head.predict(x_proj.cpu().detach().numpy())
                            class_name = (id_to_caption[predicted_label[0]]).split(",")[0]
                        prompt = f"a photo of a {class_name}"
                        print(prompt)
                
                else:
                    prompt = ""

                with torch.no_grad():
                    # Local controls are zeroed for all 7 maps, 3 channels each
                    detected_maps = np.zeros((H, W, 3 * 7), np.uint8)

                    local_control = torch.from_numpy(detected_maps.copy()).float().to(device) / 255.0
                    local_control = local_control.unsqueeze(0)  # B=1
                    local_control = einops.rearrange(local_control, "b h w c -> b c h w").clone()

                    # This one for ATMS TODO why is it different
                    #global_control = torch.from_numpy(content_emb.squeeze(0).detach().cpu().numpy().copy()).float().to(device).clone().unsqueeze(0)
                    
                    if isinstance(content_emb, torch.Tensor):
                        global_control = content_emb if content_emb.dim() == 2 else content_emb.unsqueeze(0)
                    else:
                        global_control = torch.from_numpy(np.array(content_emb)).float()
                        global_control = global_control if global_control.dim() == 2 else global_control.unsqueeze(0)
                    global_control = global_control.to(device).clone()
                    
                    if config.save_memory:
                        model.low_vram_shift(is_diffusing=False)

                    uc_local_control = local_control
                    uc_global_control = torch.zeros_like(global_control)

                    cond = {
                        "local_control": [local_control],
                        "c_crossattn": [model.get_learned_conditioning([prompt + (", " + args.a_prompt if args.a_prompt else "")])],
                        "global_control": [global_control],
                    }
                    un_cond = {
                        "local_control": [uc_local_control],
                        "c_crossattn": [model.get_learned_conditioning([args.n_prompt])],
                        "global_control": [uc_global_control],
                    }
                    shape = (4, H // 8, W // 8)

                    if config.save_memory:
                        model.low_vram_shift(is_diffusing=True)

                    model.control_scales = [args.strength] * 13

                    samples, _ = ddim_sampler.sample(
                        args.ddim_steps,
                        1,
                        shape,
                        cond,
                        verbose=False,
                        eta=args.eta,
                        unconditional_guidance_scale=args.scale,
                        unconditional_conditioning=un_cond,
                        global_strength=args.global_strength,
                    )

                    if config.save_memory:
                        model.low_vram_shift(is_diffusing=False)

                    x_t = model.decode_first_stage(samples)
                    x_t = (einops.rearrange(x_t, "b c h w -> b h w c") * 127.5 + 127.5).clamp(0, 255)
                    x = x_t[0].detach().cpu().numpy().astype(np.uint8)

                # Save generated image
                gen_path = out_samples / f"{eeg_path.stem}.png"
                Image.fromarray(x).save(gen_path)

            # Load real image
            real_img = Image.open(real_path).convert("RGB")
            real_np = np.array(real_img, dtype=np.uint8)

            # FID / IS (uint8, CxHxW)
            gen_u8 = _to_chw_uint8(x).to(device).unsqueeze(0)
            real_u8 = _to_chw_uint8(real_np).to(device).unsqueeze(0)
            fid.update(real_u8, real=True)
            fid.update(gen_u8, real=False)
            isc.update(gen_u8)

            # KID (uint8, CxHxW)
            #gen_f = _to_chw_float01(x).to(device).unsqueeze(0)
            #real_f = _to_chw_float01(real_np).to(device).unsqueeze(0)
            kid.update(real_u8, real=True)
            kid.update(gen_u8, real=False)

            # Float tensors for LPIPS / SSIM
            gen_f = _to_chw_float01(x).to(device).unsqueeze(0)
            real_f = _to_chw_float01(real_np).to(device).unsqueeze(0)
            
            # LPIPS & SSIM require matching sizes; resize real to target HxW
            if real_np.shape[0] != H or real_np.shape[1] != W:
                real_resized = np.array(real_img.resize((W, H), Image.BICUBIC), dtype=np.uint8)
                real_f = _to_chw_float01(real_resized).to(device).unsqueeze(0)

            # LPIPS expects [-1, 1]
            gen_lp = gen_f * 2.0 - 1.0
            real_lp = real_f * 2.0 - 1.0
            lp_val = lpips(gen_lp, real_lp).detach().item()
            lpips_sum += lp_val

            # SSIM expects [0,1], data_range=1.0 set above
            ssim_val = ssim(gen_f, real_f).detach().item()
            ssim_sum += ssim_val

            # CLIP cosine similarity between generated and real image
            with torch.no_grad():
                gen_clip = clip_model.encode_image(clip_preprocess_pil(Image.fromarray(x)).unsqueeze(0).to(device))
                real_clip = clip_model.encode_image(clip_preprocess_pil(real_img).unsqueeze(0).to(device))
                gen_clip = gen_clip / gen_clip.norm(dim=-1, keepdim=True)
                real_clip = real_clip / real_clip.norm(dim=-1, keepdim=True)
                cosine_sim = torch.sum(gen_clip * real_clip).item()
            cosine_rankings.append({"file": f"{eeg_path.stem}.png", "cosine": cosine_sim})

            # Store for additional metrics (keep on CPU to save GPU memory)
            gen_imgs.append(gen_f.squeeze(0).cpu())
            real_imgs.append(real_f.squeeze(0).cpu())

            processed += 1
            if processed % 100 == 0:
                logger.info(f"Processed {processed}/{len(pairs)}")
        except Exception as e:
            logger.error(f"Error on {eeg_path.name}: {e}", exc_info=True)
            continue

    if processed == 0:
        logger.error("No pairs processed; aborting metrics computation.")
        return

    # Compute aggregate metrics
    print("Computing final metrics...")
    fid_score = float(fid.compute().item())
    print(f" - FID: {fid_score:.4f}")
    is_mean, is_std = isc.compute()
    is_mean = float(is_mean.item())
    is_std = float(is_std.item())
    print(f" - Inception Score: mean={is_mean:.4f}, std={is_std:.4f}")
    kid_mean, kid_std = kid.compute()
    kid_mean = float(kid_mean.item())
    kid_std = float(kid_std.item())
    print(f" - KID: mean={kid_mean:.4f}, std={kid_std:.4f}")
    lpips_mean = float(lpips_sum / processed)
    print(f" - LPIPS: {lpips_mean:.4f}")
    ssim_mean = float(ssim_sum / processed)
    print(f" - SSIM: {ssim_mean:.4f}")

    additional_metrics = compute_additional_metrics(
        gen_imgs,
        real_imgs,
        device,
        clip_model=clip_model,
        clip_preprocess_tensor=clip_preprocess_tensor,
    )

    metrics = {
        "pairs": len(pairs),
        "processed": processed,
        "FID": fid_score,
        "InceptionScore": {"mean": is_mean, "std": is_std},
        "KID": {"mean": kid_mean, "std": kid_std},
        "LPIPS": lpips_mean,
        "SSIM": ssim_mean,
        "output_dir": str(out_root),
        "used_generated_only": bool(args.skip_generate),
        "best_images_by_cosine": sorted(cosine_rankings, key=lambda d: d["cosine"], reverse=True)[: min(len(cosine_rankings), 20)],
        "worst_images_by_cosine": sorted(cosine_rankings, key=lambda d: d["cosine"])[: min(len(cosine_rankings), 20)],
    }

    metrics.update(additional_metrics)

    with open(out_root / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    logger.info(json.dumps(metrics, indent=2))
    logger.info(f"Samples dir: {out_samples}")




@torch.no_grad()
def compute_additional_metrics(gen_imgs, real_imgs, device, batch_size=32, clip_model=None, clip_preprocess_tensor=None):
    '''Compute additional metrics mirrored from ATM-S paper'''
    print("Computing additional metrics...")
    results = {}
    if not gen_imgs or not real_imgs:
        return results

    try:
        stacked_gen = torch.stack(gen_imgs, dim=0)
        stacked_real = torch.stack(real_imgs, dim=0)

        # PixCorr
        print(" - PixCorr")
        resize_425 = transforms.Resize(425, interpolation=InterpolationMode.BILINEAR)
        gen_flat = resize_425(stacked_gen).reshape(len(gen_imgs), -1).cpu().numpy()
        real_flat = resize_425(stacked_real).reshape(len(real_imgs), -1).cpu().numpy()
        corr_vals = [np.corrcoef(real_flat[i], gen_flat[i])[0, 1] for i in range(len(gen_imgs))]
        results["PixCorr"] = float(np.mean(corr_vals))

        # SSIM (skimage, grayscale)
        print(" - SSIM")
        gen_gray = rgb2gray(resize_425(stacked_gen).permute(0, 2, 3, 1).cpu().numpy())
        real_gray = rgb2gray(resize_425(stacked_real).permute(0, 2, 3, 1).cpu().numpy())
        ssim_scores = [
            skimage_ssim(rec, gt, channel_axis=-1, gaussian_weights=True, sigma=1.5, use_sample_covariance=False, data_range=1.0)
            for rec, gt in zip(gen_gray, real_gray)
        ]
        results["SSIM_skimage"] = float(np.mean(ssim_scores))

        # AlexNet (early and mid)
        print(" - AlexNet")
        alex_weights = AlexNet_Weights.IMAGENET1K_V1
        alex_model = create_feature_extractor(alexnet(weights=alex_weights), return_nodes={"features.4": "f4", "features.11": "f11"}).to(device)
        alex_model.eval().requires_grad_(False)
        alex_pre = transforms.Compose([
            transforms.Resize(256, interpolation=InterpolationMode.BILINEAR),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        alex_f4 = two_way_identification(gen_imgs, real_imgs, alex_model, alex_pre, device, feature_layer="f4")
        alex_f11 = two_way_identification(gen_imgs, real_imgs, alex_model, alex_pre, device, feature_layer="f11")
        results["AlexNet_early"] = alex_f4
        results["AlexNet_mid"] = alex_f11

        # InceptionV3
        print(" - InceptionV3")
        inc_model = create_feature_extractor(inception_v3(weights=Inception_V3_Weights.DEFAULT), return_nodes={"avgpool": "avgpool"}).to(device)
        inc_model.eval().requires_grad_(False)
        inc_pre = transforms.Compose([
            transforms.Resize(342, interpolation=InterpolationMode.BILINEAR),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        results["InceptionV3"] = two_way_identification(gen_imgs, real_imgs, inc_model, inc_pre, device, feature_layer="avgpool")

        # CLIP ViT-L/14
        print(" - CLIP ViT-L/14")
        clip_m = clip_model
        clip_pre = clip_preprocess_tensor
        if clip_m is None or clip_pre is None:
            clip_m, _ = clip.load("ViT-L/14", device=device)
            clip_m.eval().requires_grad_(False)
            clip_pre = transforms.Compose([
                transforms.Resize(224, interpolation=InterpolationMode.BILINEAR),
                transforms.Normalize(mean=[0.48145466, 0.4578275, 0.40821073], std=[0.26862954, 0.26130258, 0.27577711]),
            ])
        results["CLIP_ViT_L_14"] = two_way_identification(gen_imgs, real_imgs, clip_m.encode_image, clip_pre, device, feature_layer=None)

        # EfficientNet-B1 correlation distance
        print(" - EfficientNet-B1")
        eff_model = create_feature_extractor(efficientnet_b1(weights=EfficientNet_B1_Weights.DEFAULT), return_nodes={"avgpool": "avgpool"}).to(device)
        eff_model.eval().requires_grad_(False)
        eff_pre = transforms.Compose([
            transforms.Resize(255, interpolation=InterpolationMode.BILINEAR),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        eff_fake = _batched_encode(gen_imgs, eff_model, eff_pre, device, feature_layer="avgpool", batch_size=batch_size)
        eff_real = _batched_encode(real_imgs, eff_model, eff_pre, device, feature_layer="avgpool", batch_size=batch_size)
        eff_dists = [distance.correlation(eff_real[i].numpy(), eff_fake[i].numpy()) for i in range(len(gen_imgs))]
        results["EffNet_B1_corr"] = float(np.mean(eff_dists))

        # SwAV ResNet50 correlation distance
        print(" - SwAV ResNet50")
        swav_model = torch.hub.load("facebookresearch/swav:main", "resnet50")
        swav_model = create_feature_extractor(swav_model, return_nodes={"avgpool": "avgpool"}).to(device)
        swav_model.eval().requires_grad_(False)
        swav_pre = transforms.Compose([
            transforms.Resize(224, interpolation=InterpolationMode.BILINEAR),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        swav_fake = _batched_encode(gen_imgs, swav_model, swav_pre, device, feature_layer="avgpool", batch_size=batch_size)
        swav_real = _batched_encode(real_imgs, swav_model, swav_pre, device, feature_layer="avgpool", batch_size=batch_size)
        swav_dists = [distance.correlation(swav_real[i].numpy(), swav_fake[i].numpy()) for i in range(len(gen_imgs))]
        results["SwAV_corr"] = float(np.mean(swav_dists))

    except Exception as e:
        logger.warning(f"Failed to compute additional metrics: {e}")

    return results


if __name__ == "__main__":
    main()