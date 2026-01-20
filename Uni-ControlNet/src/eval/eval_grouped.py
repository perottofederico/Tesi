import argparse
import json
import logging
from collections import defaultdict
from pathlib import Path
import sys
import warnings
warnings.filterwarnings("ignore")
if "./" not in sys.path:
    sys.path.append("./")
import einops
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from scipy.spatial import distance
from skimage.color import rgb2gray
from skimage.metrics import structural_similarity as skimage_ssim
from torchvision import transforms
from torchvision.models import (
    alexnet,
    efficientnet_b1,
    inception_v3,
    AlexNet_Weights,
    EfficientNet_B1_Weights,
    Inception_V3_Weights,
)
from torchvision.models.feature_extraction import create_feature_extractor
from torchvision.transforms import InterpolationMode

import clip
from torchmetrics.image.fid import FrechetInceptionDistance
from torchmetrics.image.inception import InceptionScore
from torchmetrics.image.kid import KernelInceptionDistance
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity
from torchmetrics.image.ssim import StructuralSimilarityIndexMeasure

from annotator.eeg.eegcvpr40.eegcvpr_id_to_caption import id_to_caption

logger = logging.getLogger("eval_grouped")
logger.setLevel(logging.INFO)
ch = logging.StreamHandler()
ch.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))
logger.addHandler(ch)


# ---------------- Helper functions ----------------
def _to_chw_float01(np_img):
    t = torch.from_numpy(np_img)
    if t.ndim == 2:
        t = t.unsqueeze(-1).repeat(1, 1, 3)
    if t.shape[-1] == 4:
        t = t[..., :3]
    t = t.permute(2, 0, 1).contiguous().float() / 255.0
    return t


def _to_chw_uint8(np_img):
    t = torch.from_numpy(np_img)
    if t.ndim == 2:
        t = t.unsqueeze(-1).repeat(1, 1, 3)
    if t.shape[-1] == 4:
        t = t[..., :3]
    return t.permute(2, 0, 1).contiguous().to(torch.uint8)


def _batched_encode(images, model, preprocess, device, feature_layer=None, batch_size=32):
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


def two_way_identification(all_brain_recons, all_images, model, preprocess, device, feature_layer=None):
    preds = _batched_encode(all_brain_recons, model, preprocess, device, feature_layer)
    reals = _batched_encode(all_images, model, preprocess, device, feature_layer)

    preds_np = preds.numpy()
    reals_np = reals.numpy()

    r = np.corrcoef(reals_np, preds_np)
    r = r[: len(all_images), len(all_images) :]
    congruents = np.diag(r)
    success = r < congruents
    success_cnt = np.sum(success, axis=0)

    if len(all_images) <= 1:
        return float("nan")
    perf = np.mean(success_cnt) / (len(all_images) - 1)
    return float(perf)


def compute_additional_metrics(gen_imgs, real_imgs, device, batch_size=32, clip_model=None, clip_preprocess_tensor=None):
    results = {}
    if not gen_imgs or not real_imgs:
        return results

    try:
        stacked_gen = torch.stack(gen_imgs, dim=0)
        stacked_real = torch.stack(real_imgs, dim=0)

        resize_425 = transforms.Resize(425, interpolation=InterpolationMode.BILINEAR)
        gen_flat = resize_425(stacked_gen).reshape(len(gen_imgs), -1).cpu().numpy()
        real_flat = resize_425(stacked_real).reshape(len(real_imgs), -1).cpu().numpy()
        corr_vals = [np.corrcoef(real_flat[i], gen_flat[i])[0, 1] for i in range(len(gen_imgs))]
        results["PixCorr"] = float(np.mean(corr_vals))

        gen_gray = rgb2gray(resize_425(stacked_gen).permute(0, 2, 3, 1).cpu().numpy())
        real_gray = rgb2gray(resize_425(stacked_real).permute(0, 2, 3, 1).cpu().numpy())
        ssim_scores = [
            skimage_ssim(rec, gt, channel_axis=-1, gaussian_weights=True, sigma=1.5, use_sample_covariance=False, data_range=1.0)
            for rec, gt in zip(gen_gray, real_gray)
        ]
        results["SSIM_skimage"] = float(np.mean(ssim_scores))

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

        inc_model = create_feature_extractor(inception_v3(weights=Inception_V3_Weights.DEFAULT), return_nodes={"avgpool": "avgpool"}).to(device)
        inc_model.eval().requires_grad_(False)
        inc_pre = transforms.Compose([
            transforms.Resize(342, interpolation=InterpolationMode.BILINEAR),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        results["InceptionV3"] = two_way_identification(gen_imgs, real_imgs, inc_model, inc_pre, device, feature_layer="avgpool")

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


def load_pairs(gen_dir: Path, real_dir: Path):
    real_lookup = {}
    for ext in ("jpg", "jpeg", "png"):
        for p in real_dir.glob(f"*.{ext}"):
            real_lookup[p.stem] = p

    pairs = []
    missing_real = []
    for ext in ("png", "jpg", "jpeg"):
        for gen_path in gen_dir.glob(f"*.{ext}"):
            stem = gen_path.stem
            real_path = real_lookup.get(stem)
            if real_path is None:
                missing_real.append(stem)
                continue
            pairs.append((stem, gen_path, real_path))

    if missing_real:
        logger.warning(f"Missing real images for {len(missing_real)} generated files (first 5 shown): {missing_real[:5]}")
    logger.info(f"Paired {len(pairs)} images")
    return pairs


def subject_from_stem(stem: str):
    parts = stem.rsplit("_", 1)
    return parts[-1] if len(parts) > 1 else "unknown"


def load_class_labels(labels_path: Path):
    mapping = {}
    if not labels_path.exists():
        logger.warning(f"test_labels file not found: {labels_path}")
        return mapping
    with labels_path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            try:
                fname, cls = line.strip().split("\t")
                stem = Path(fname).stem
                mapping[stem] = int(cls)
            except ValueError:
                logger.warning(f"Skipping malformed line: {line.strip()}")
    return mapping


def evaluate_group(pairs, image_resolution, device, clip_model, clip_preprocess):
    fid = FrechetInceptionDistance(feature=2048).to(device)
    isc = InceptionScore(splits=10).to(device)
    kid = KernelInceptionDistance(subset_size=(min(200, len(pairs)))).to(device)
    lpips = LearnedPerceptualImagePatchSimilarity(net_type="alex", reduction="mean").to(device)
    ssim = StructuralSimilarityIndexMeasure(data_range=1.0).to(device)

    gen_imgs = []
    real_imgs = []
    lpips_sum = 0.0
    ssim_sum = 0.0
    processed = 0

    H = W = int(image_resolution)

    for stem, gen_path, real_path in pairs:
        try:
            gen_img = Image.open(gen_path).convert("RGB")
            real_img = Image.open(real_path).convert("RGB")
            gen_np = np.array(gen_img, dtype=np.uint8)
            real_np = np.array(real_img, dtype=np.uint8)

            gen_u8 = _to_chw_uint8(gen_np).to(device).unsqueeze(0)
            real_u8 = _to_chw_uint8(real_np).to(device).unsqueeze(0)
            fid.update(real_u8, real=True)
            fid.update(gen_u8, real=False)
            isc.update(gen_u8)
            kid.update(real_u8, real=True)
            kid.update(gen_u8, real=False)

            gen_f = _to_chw_float01(gen_np).to(device).unsqueeze(0)
            real_f = _to_chw_float01(real_np).to(device).unsqueeze(0)
            if real_np.shape[0] != H or real_np.shape[1] != W:
                real_resized = np.array(real_img.resize((W, H), Image.BICUBIC), dtype=np.uint8)
                real_f = _to_chw_float01(real_resized).to(device).unsqueeze(0)

            gen_lp = gen_f * 2.0 - 1.0
            real_lp = real_f * 2.0 - 1.0
            lpips_sum += lpips(gen_lp, real_lp).detach().item()
            ssim_sum += ssim(gen_f, real_f).detach().item()

            gen_imgs.append(gen_f.squeeze(0).cpu())
            real_imgs.append(real_f.squeeze(0).cpu())
            processed += 1
        except Exception as e:
            logger.error(f"Failed on pair {stem}: {e}", exc_info=True)
            continue

    if processed == 0:
        return None

    fid_score = float(fid.compute().item())
    is_mean, is_std = isc.compute()
    kid_mean, kid_std = kid.compute()
    lpips_mean = float(lpips_sum / processed)
    ssim_mean = float(ssim_sum / processed)

    additional = compute_additional_metrics(
        gen_imgs,
        real_imgs,
        device,
        clip_model=clip_model,
        clip_preprocess_tensor=clip_preprocess,
    )

    metrics = {
        "count": processed,
        "FID": fid_score,
        "InceptionScore": {"mean": float(is_mean.item()), "std": float(is_std.item())},
        "KID": {"mean": float(kid_mean.item()), "std": float(kid_std.item())},
        "LPIPS": lpips_mean,
        "SSIM": ssim_mean,
    }
    metrics.update(additional)
    return metrics


def _flatten_group_metrics(group_name, group_metrics):
    return {
        "name": group_name,
        "count": group_metrics.get("count", 0),
        "FID": group_metrics.get("FID"),
        "IS_mean": group_metrics.get("InceptionScore", {}).get("mean"),
        "IS_std": group_metrics.get("InceptionScore", {}).get("std"),
        "KID_mean": group_metrics.get("KID", {}).get("mean"),
        "KID_std": group_metrics.get("KID", {}).get("std"),
        "LPIPS": group_metrics.get("LPIPS"),
        "SSIM": group_metrics.get("SSIM"),
        "PixCorr": group_metrics.get("PixCorr"),
        "SSIM_skimage": group_metrics.get("SSIM_skimage"),
        "AlexNet_early": group_metrics.get("AlexNet_early"),
        "AlexNet_mid": group_metrics.get("AlexNet_mid"),
        "InceptionV3": group_metrics.get("InceptionV3"),
        "CLIP_ViT_L_14": group_metrics.get("CLIP_ViT_L_14"),
        "EffNet_B1_corr": group_metrics.get("EffNet_B1_corr"),
        "SwAV_corr": group_metrics.get("SwAV_corr"),
    }


def _analyze_dimension(groups_dict):
    rows = [_flatten_group_metrics(name, m) for name, m in groups_dict.items()]
    if not rows:
        return {}

    total_images = sum(r.get("count", 0) or 0 for r in rows)
    num_groups = len(rows)

    metric_prefs = {
        "FID": "lower",
        "IS_mean": "higher",
        "IS_std": "lower",
        "KID_mean": "lower",
        "KID_std": "lower",
        "LPIPS": "lower",
        "SSIM": "higher",
        "PixCorr": "higher",
        "SSIM_skimage": "higher",
        "AlexNet_early": "higher",
        "AlexNet_mid": "higher",
        "InceptionV3": "higher",
        "CLIP_ViT_L_14": "higher",
        "EffNet_B1_corr": "lower",
        "SwAV_corr": "lower",
    }

    averages = {}
    best = {}
    worst = {}

    for metric, pref in metric_prefs.items():
        available = [(r[metric], r.get("count", 0) or 0, r["name"]) for r in rows if r.get(metric) is not None]
        if not available:
            continue

        weight_sum = sum(w for _, w, _ in available)
        if weight_sum > 0:
            avg_val = sum(v * w for v, w, _ in available) / float(weight_sum)
        else:
            avg_val = sum(v for v, _, _ in available) / float(len(available))
        averages[metric] = avg_val

        reverse = pref == "higher"
        ordered = sorted(available, key=lambda t: t[0], reverse=reverse)
        best[metric] = {"group": ordered[0][2], "value": ordered[0][0]}
        worst[metric] = {"group": ordered[-1][2], "value": ordered[-1][0]}

    return {
        "num_groups": num_groups,
        "total_images": total_images,
        "averages": averages,
        "best": best,
        "worst": worst,
    }


def analyze_metrics(metrics_data):
    subjects = metrics_data.get("subjects", {}) if isinstance(metrics_data, dict) else {}
    classes = metrics_data.get("classes", {}) if isinstance(metrics_data, dict) else {}

    return {
        "subjects": _analyze_dimension(subjects),
        "classes": _analyze_dimension(classes),
    }


def main():
    ap = argparse.ArgumentParser(description="Per-subject and per-class evaluation on pre-generated samples")
    ap.add_argument("--gen_dir", type=str, help="Directory with generated images (png/jpg)")
    ap.add_argument("--real_dir", type=str, help="Directory with real reference images (jpg/png)")
    ap.add_argument("--test_labels", type=str, default="data_EEGCVPR/test_labels.txt", help="Path to test_labels.txt mapping file")
    ap.add_argument("--image_resolution", type=int, default=256, help="Target resolution for LPIPS/SSIM resize")
    ap.add_argument("--output", type=str, default="group_metrics.json", help="Where to store the resulting JSON report")
    ap.add_argument("--metrics_file", type=str, help="Existing grouped-metrics JSON to analyze without recomputing")
    ap.add_argument("--analyze_only", action="store_true", help="Skip evaluation and only run analysis on --metrics_file")
    ap.add_argument("--analysis_output", type=str, default="group_metrics_analysis.json", help="Where to store the analysis JSON report")
    args = ap.parse_args()

    metrics_data = None
    metrics_path = Path(args.metrics_file) if args.metrics_file else Path(args.output)

    if args.analyze_only:
        if not metrics_path.exists():
            logger.error(f"--analyze_only requested but metrics file not found: {metrics_path}")
            return
        with metrics_path.open("r", encoding="utf-8") as f:
            metrics_data = json.load(f)
        logger.info(f"Loaded existing metrics from {metrics_path}")
    else:
        if args.gen_dir is None or args.real_dir is None:
            logger.error("--gen_dir and --real_dir are required when computing metrics")
            return

        gen_dir = Path(args.gen_dir)
        real_dir = Path(args.real_dir)
        labels_path = Path(args.test_labels)

        if not gen_dir.exists() or not real_dir.exists():
            logger.error(f"Input dirs not found. gen_dir={gen_dir}, real_dir={real_dir}")
            return

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Device: {device}")

        clip_model, clip_preprocess = clip.load("ViT-L/14", device=device)
        clip_model.eval().requires_grad_(False)
        clip_preprocess_tensor = transforms.Compose([
            transforms.Resize(224, interpolation=InterpolationMode.BILINEAR),
            transforms.Normalize(mean=[0.48145466, 0.4578275, 0.40821073], std=[0.26862954, 0.26130258, 0.27577711]),
        ])

        pairs = load_pairs(gen_dir, real_dir)
        if not pairs:
            logger.error("No pairs to evaluate")
            return

        subject_groups = defaultdict(list)
        for stem, g, r in pairs:
            subject_groups[subject_from_stem(stem)].append((stem, g, r))

        class_labels = load_class_labels(labels_path)
        class_groups = defaultdict(list)
        for stem, g, r in pairs:
            cls_id = class_labels.get(stem)
            if cls_id is None:
                continue
            cls_name = id_to_caption.get(cls_id, str(cls_id))
            class_groups[cls_name].append((stem, g, r))

        results = {"subjects": {}, "classes": {}}

        logger.info("Evaluating per subject...")
        for subject, group_pairs in subject_groups.items():
            logger.info(f"Subject {subject}: {len(group_pairs)} samples")
            metrics = evaluate_group(group_pairs, args.image_resolution, device, clip_model, clip_preprocess_tensor)
            if metrics:
                results["subjects"][subject] = metrics

        logger.info("Evaluating per class...")
        for cls, group_pairs in class_groups.items():
            logger.info(f"Class {cls}: {len(group_pairs)} samples")
            metrics = evaluate_group(group_pairs, args.image_resolution, device, clip_model, clip_preprocess_tensor)
            if metrics:
                results["classes"][cls] = metrics

        with metrics_path.open("w", encoding="utf-8") as f:
            json.dump(results, f, indent=2)

        metrics_data = results
        logger.info(f"Saved grouped metrics to {metrics_path}")

    if metrics_data is None:
        logger.error("No metrics data available for analysis")
        return

    analysis = analyze_metrics(metrics_data)
    with open(args.analysis_output, "w", encoding="utf-8") as f:
        json.dump(analysis, f, indent=2)

    logger.info(f"Saved analysis to {args.analysis_output}")


if __name__ == "__main__":
    main()
