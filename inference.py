import os
import argparse
from types import SimpleNamespace

import numpy as np
from PIL import Image, ImageFile

import torch
from transformers import Mask2FormerForUniversalSegmentation, Mask2FormerImageProcessor

# Be robust to big / slightly broken TIFFs
Image.MAX_IMAGE_PIXELS = None
ImageFile.LOAD_TRUNCATED_IMAGES = True


def load_model(model_dir, device):
    print(f"Loading model from: {model_dir}")
    processor = Mask2FormerImageProcessor.from_pretrained(model_dir)
    model = Mask2FormerForUniversalSegmentation.from_pretrained(model_dir)
    model.to(device)
    model.eval()
    return processor, model


def overlay_instances(image, seg_maps, scores, score_threshold=0.5, max_instances=50):
    img_np = np.array(image).astype(np.float32)

    if scores is None or len(scores) == 0:
        print("No predictions returned.")
        return Image.fromarray(img_np.astype(np.uint8))

    scores = torch.as_tensor(scores)
    keep = scores > score_threshold
    if keep.numel() == 0:
        print(f"No instances above score threshold of {score_threshold}.")
        return Image.fromarray(img_np.astype(np.uint8))

    keep_idx = keep.nonzero(as_tuple=False).view(-1).tolist()
    seg_maps = [seg_maps[i] for i in keep_idx]
    scores = scores[keep]

    if len(seg_maps) == 0:
        print("No instances left after filtering.")
        return Image.fromarray(img_np.astype(np.uint8))

    if len(seg_maps) > max_instances:
        print(f"Limiting visualization to top {max_instances} instances.")
        seg_maps = seg_maps[:max_instances]
        scores = scores[:max_instances]

    H, W, _ = img_np.shape
    overlay = img_np.copy()
    rng = np.random.default_rng(0)
    colors = rng.integers(0, 255, size=(len(seg_maps), 3), dtype=np.uint8)
    alpha = 0.5

    for idx, mask in enumerate(seg_maps):
        mask_bool = np.asarray(mask).astype(bool)
        if mask_bool.shape != (H, W):
            import cv2
            mask_bool = cv2.resize(
                mask_bool.astype(np.uint8), (W, H), interpolation=cv2.INTER_NEAREST
            ).astype(bool)

        color = colors[idx]
        overlay[mask_bool] = (1 - alpha) * overlay[mask_bool] + alpha * color

    overlay = np.clip(overlay, 0, 255).astype(np.uint8)
    return Image.fromarray(overlay)


def run_inference(
    model_dir, image_path, output_path, score_threshold=0.5,
    max_instances=50, max_side=2048, device=None,
):
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    processor, model = load_model(model_dir, device)
    image = Image.open(image_path).convert("RGB")
    orig_w, orig_h = image.size
    print(f"Original image: {image_path} (W={orig_w}, H={orig_h})")

    long_side = max(orig_w, orig_h)
    if long_side > max_side:
        scale = max_side / float(long_side)
        new_w, new_h = int(orig_w * scale), int(orig_h * scale)
        print(f"Resizing for inference to (W={new_w}, H={new_h})")
        image = image.resize((new_w, new_h), resample=Image.BILINEAR)
    else:
        new_w, new_h = orig_w, orig_h
        print("No resize needed for inference.")

    print(f"Running inference on device: {device}")
    inputs = processor(images=image, return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model(**inputs)

    if hasattr(outputs, "to"): outputs = outputs.to("cpu")
    else: outputs = {k: v.cpu() for k, v in outputs.items()}
    torch.cuda.empty_cache()
    if isinstance(outputs, dict): outputs = SimpleNamespace(**outputs)

    processor.thing_ids = [1]
    processed = processor.post_process_instance_segmentation(
        outputs,
        target_sizes=[(new_h, new_w)],
    )[0]

    seg_maps, scores = [], []

    if "segments_info" in processed and len(processed["segments_info"]) > 0:
        segmentation_map = processed["segmentation"]
        segments_info = processed["segments_info"]
        
        for seg in segments_info:
            mask = (segmentation_map == seg["id"])
            seg_maps.append(mask)
            scores.append(seg["score"])
        print(f"Instance case: Found {len(seg_maps)} 'fallen_tree' instances.")
    else:
        print("No instances found after post-processing. Check model predictions and score threshold.")

    if len(scores) > 0:
        print("Top scores:", torch.tensor(scores)[:5].tolist())

    result_img = overlay_instances(
        image, seg_maps, scores, score_threshold=score_threshold, max_instances=max_instances
    )

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    result_img.save(output_path)
    print(f"Saved overlay to: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Run inference with fine-tuned Mask2Former.")
    parser.add_argument("--model_dir", type=str, required=True, help="Path to the fine-tuned model directory.")
    parser.add_argument("--image_path", type=str, required=True, help="Path to the input image.")
    parser.add_argument("--output_path", type=str, required=True, help="Path to save the overlay PNG.")
    parser.add_argument("--score_threshold", type=float, default=0.5, help="Score threshold for keeping instances.")
    parser.add_argument("--max_instances", type=int, default=100, help="Maximum number of instances to visualize.")
    parser.add_argument("--max_side", type=int, default=2048, help="Max image side for inference.")
    args = parser.parse_args()

    run_inference(
        model_dir=args.model_dir, image_path=args.image_path, output_path=args.output_path,
        score_threshold=args.score_threshold, max_instances=args.max_instances, max_side=args.max_side,
    )

if __name__ == "__main__":
    main()