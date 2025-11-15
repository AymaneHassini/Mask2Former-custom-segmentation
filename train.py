# train.py – Mask2Former fine-tuning on fallen trees (2 classes: background + fallen_tree)

import os
import json
import argparse
from pathlib import Path

import numpy as np
from PIL import Image, ImageFile

import torch
from torch.utils.data import Dataset

from transformers import (
    Mask2FormerForUniversalSegmentation,
    Mask2FormerImageProcessor,
    Trainer,
    TrainingArguments,
)
from pycocotools import mask as coco_mask
import albumentations as A  # heavy data augmentation

# Allow very large & slightly truncated TIFFs
Image.MAX_IMAGE_PIXELS = None
ImageFile.LOAD_TRUNCATED_IMAGES = True


class CocoInstanceSegmentationDataset(Dataset):
    """
    PyTorch Dataset for COCO-formatted instance segmentation,
    prepared for Mask2FormerImageProcessor using an instance map.
    """

    def __init__(self, image_dir, annotation_file, processor, augment=False, crop_size=(1024, 1024)):
        self.image_dir = Path(image_dir)
        self.processor = processor
        self.augment = augment
        self.crop_size = crop_size

        with open(annotation_file, "r") as f:
            coco_data = json.load(f)

        self.images = coco_data["images"]
        self.annotations = coco_data["annotations"]

        # Map image_id -> list[annotation]
        self.img_to_anns = {}
        for ann in self.annotations:
            img_id = ann["image_id"]
            self.img_to_anns.setdefault(img_id, []).append(ann)

        print(f"Loaded {len(self.images)} images from {annotation_file}")

        #  Augmentation pipeline (Albumentations) 
        if augment:
            crop_h, crop_w = self.crop_size
            self.transform = A.Compose(
                [
                    # Random crop & rescale to crop_size
                    A.RandomResizedCrop(
                        size=(crop_h, crop_w),  
                        scale=(0.5, 1.0),
                        ratio=(0.75, 1.33),
                        p=1.0,
                    ),

                    # Geometric jitter
                    A.HorizontalFlip(p=0.5),
                    A.VerticalFlip(p=0.5),
                    A.RandomRotate90(p=0.5),
                    A.ShiftScaleRotate(
                        shift_limit=0.05,
                        scale_limit=0.1,
                        rotate_limit=10,
                        border_mode=0,  
                        p=0.5,
                    ),

                    # Photometric (image only)
                    A.OneOf(
                        [
                            A.RandomBrightnessContrast(
                                brightness_limit=0.3,
                                contrast_limit=0.3,
                                p=1.0,
                            ),
                            A.CLAHE(
                                clip_limit=2.0,
                                tile_grid_size=(8, 8),
                                p=1.0,
                            ),
                            A.RGBShift(
                                r_shift_limit=10,
                                g_shift_limit=10,
                                b_shift_limit=10,
                                p=1.0,
                            ),
                        ],
                        p=0.7,
                    ),

                    # Noise / blur
                    A.GaussNoise(p=0.3),
                    A.MotionBlur(blur_limit=5, p=0.2),
                ]
            )
        else:
            self.transform = None

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image_info = self.images[idx]
        img_id = image_info["id"]
        image_path = self.image_dir / image_info["file_name"]

        # Load image and get real size from PIL (robust to truncated TIFFs)
        image = Image.open(image_path).convert("RGB")
        width, height = image.size  # (W, H)

        anns = self.img_to_anns.get(img_id, [])

        # Build instance map: (H, W), 0 = background, 1..255 = instances
        instance_map = np.zeros((height, width), dtype=np.int32)

        # Mapping from instance id -> semantic id
        # 0 -> 0 (background), 1..K -> 1 ("fallen_tree")
        instance_id_to_semantic_id = {0: 0}

        current_inst = 0
        max_instances = 255  # 8-bit limit

        for ann in anns:
            if current_inst >= max_instances:
                # skip extra instances beyond 255
                continue

            seg = ann.get("segmentation", None)
            if not seg:
                continue

            rle = coco_mask.frPyObjects(seg, height, width)
            mask = coco_mask.decode(rle)
            if mask.ndim == 3:
                mask = np.any(mask, axis=2)

            if mask.max() == 0:
                continue

            current_inst += 1
            instance_map[mask.astype(bool)] = current_inst
            # semantic id 1 = fallen_tree
            instance_id_to_semantic_id[current_inst] = 1

        # Convert to numpy for Albumentations
        image_np = np.array(image).astype(np.uint8)
        instance_uint8 = instance_map.astype(np.uint8)

        # Apply Albumentations (joint image + mask) if enabled
        if self.transform is not None:
            transformed = self.transform(image=image_np, mask=instance_uint8)
            image_np = transformed["image"]
            instance_uint8 = transformed["mask"]

        # Back to PIL / uint8 for processor
        final_image = Image.fromarray(image_np)
        final_instance_map = instance_uint8

        inputs = self.processor(
            images=[final_image],
            segmentation_maps=[final_instance_map],
            instance_id_to_semantic_id=[instance_id_to_semantic_id],
            return_tensors="pt",
        )

        # Squeeze only tensors, unwrap lists for B=1
        out = {}
        for k, v in inputs.items():
            if isinstance(v, torch.Tensor):
                out[k] = v.squeeze(0)  
            elif isinstance(v, list):
                out[k] = v[0]        
            else:
                out[k] = v

        return out


def collate_fn(batch):
    pixel_values = torch.stack([b["pixel_values"] for b in batch])
    pixel_mask = torch.stack([b["pixel_mask"] for b in batch])
    mask_labels = [b["mask_labels"] for b in batch]
    class_labels = [b["class_labels"] for b in batch]
    return {
        "pixel_values": pixel_values,
        "pixel_mask": pixel_mask,
        "mask_labels": mask_labels,
        "class_labels": class_labels,
    }


def main(args):
    print("=== Fine-tuning Mask2Former on fallen trees ===")
    print(f"COCO path:   {args.coco_path}")
    print(f"Images path: {args.data_path}")
    print(f"Output dir:  {args.output_dir}")
    print(f"Epochs: {args.epochs} | Batch size: {args.batch_size} | LR: {args.learning_rate}")

    os.makedirs(args.output_dir, exist_ok=True)

    processor = Mask2FormerImageProcessor.from_pretrained(
        args.model_checkpoint,
        do_reduce_labels=False,
    )

    # 2 classes: 0 = background, 1 = fallen_tree
    id2label = {0: "background", 1: "fallen_tree"}
    label2id = {"background": 0, "fallen_tree": 1}

    model = Mask2FormerForUniversalSegmentation.from_pretrained(
        args.model_checkpoint,
        id2label=id2label,
        label2id=label2id,
        num_labels=len(id2label),  # 2
        ignore_mismatched_sizes=True,
    )

    # Freeze backbone 
    print("\n--- Freezing Backbone ---")
    for name, param in model.model.pixel_level_module.encoder.named_parameters():
        param.requires_grad = False
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Trainable parameters: {trainable_params:,} / {total_params:,}")
    print("--- Backbone Frozen ---\n")

    train_dataset = CocoInstanceSegmentationDataset(
        image_dir=args.data_path,
        annotation_file=os.path.join(args.coco_path, "train_coco.json"),
        processor=processor,
        augment=True,                 
        crop_size=(1024, 1024),
    )
    val_dataset = CocoInstanceSegmentationDataset(
        image_dir=args.data_path,
        annotation_file=os.path.join(args.coco_path, "val_coco.json"),
        processor=processor,
        augment=False,                
        crop_size=(1024, 1024),
    )

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.epochs,
        seed=42,
        learning_rate=args.learning_rate,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        weight_decay=0.05,
        eval_strategy="epoch",
        save_strategy="epoch",
        logging_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        report_to=[],  
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=collate_fn,
    )

    print("Starting training...")
    trainer.train()

    print("Training finished. Saving the best model...")
    final_model_path = os.path.join(args.output_dir, "best_model")
    trainer.save_model(final_model_path)
    processor.save_pretrained(final_model_path)
    print(f"Model saved to {final_model_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Fine-tune Mask2Former for fallen tree instance segmentation."
    )
    parser.add_argument(
        "--model_checkpoint",
        type=str,
        default="facebook/mask2former-swin-base-coco-instance",
    )
    parser.add_argument(
        "--coco_path",
        type=str,
        default="./data/coco_formatted",
    )
    parser.add_argument(
        "--data_path",
        type=str,
        default="./custom-dataset",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./mask2former-fallen-trees-finetuned-2class-aug",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=5,
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=1,
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=1e-5,
    )
    args = parser.parse_args()
    main(args)
