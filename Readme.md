# Mask2Former for Fallen Tree Instance Segmentation

This repository contains a complete pipeline to fine-tune a Mask2Former model on a custom dataset of fallen trees in **aerial imagery**. The project demonstrates the full end-to-end process, including data preparation, model training with data augmentation, and an inference script to visualize the results.

---
## Setup

### 1. Clone the Repository
```bash
git clone https://github.com/AymaneHassini/Mask2Former-custom-segmentation.git
cd Mask2Former-custom-segmentation
```

### 2. Set up the Conda Environment

# Create a new conda environment named 'mask2former' with Python 3.10
```bash
conda create --name mask2former python=3.10 -y
```
# Activate the newly created environment
```bash
conda activate mask2former
```
# Install all required packages from the requirements file
```bash
pip install -r requirements.txt
```

## Pipeline Execution

The pipeline consists of three main steps, which should be run in order:
1.  **Convert Data:** Use `convert_to_coco.py` to prepare the annotations.
2.  **Train Model:** Use `train.py` to fine-tune the model.
3.  **Run Inference:** Use `inference.py` to generate predictions with the trained model.

### **Step A: Convert Annotations to COCO Format**

This script reads the original dataset (in LabelMe format) and converts the annotations into the standard COCO format required for training. It will create two files: `train_coco.json` and `val_coco.json`.

**Command:**

Run the following command in your terminal, replacing the placeholder paths with the actual paths on your system.

```bash
python convert_to_coco.py \
  --data_dir <PATH_TO_ORIGINAL_DATASET> \
  --output_dir <PATH_TO_SAVE_COCO_FILES>
```

### **Step B: Train the Model**

This script fine-tunes the Mask2Former model using the prepared COCO annotations and the original images. It incorporates a strong data augmentation pipeline to improve generalization and freezes the model's backbone to prevent overfitting on the small dataset.

The script will save model checkpoints after each epoch and will automatically keep the best-performing one based on the validation loss.

**Command:**

Run the following command, providing the paths from the previous step and specifying a directory to save the trained model.

```bash
python train.py \
  --data_path <PATH_TO_YOUR_ORIGINAL_DATASET> \
  --coco_path <PATH_TO_YOUR_COCO_FILES> \
  --output_dir <PATH_TO_SAVE_MODEL_CHECKPOINTS> \
  --epochs 5 \
  --batch_size 1 \
  --learning_rate 1e-5
```
---

### **Step C: Run Inference**

This script loads the best-trained model from the previous step and runs it on a single image to generate an instance segmentation overlay.

**Command:**

Run the following command, providing the path to your saved model, the path to an input image, and a path for the output visualization.

```bash
python inference.py \
  --model_dir <PATH_TO_YOUR_SAVED_MODEL>/best_model \
  --image_path <PATH_TO_AN_INPUT_IMAGE> \
  --output_path <PATH_TO_SAVE_THE_OUTPUT_PNG>```
```
**Optional Arguments**

- **`--score_threshold`**  
  Confidence threshold for displaying detected instances.  
  **Default:** `0.5`  
  **Example:** `--score_threshold 0.3` (shows more low-confidence predictions)

- **`--max_side`**  
  Maximum size (in pixels) for the longer side of the input image.  
  **Default:** `2048`  
  **Example:** `--max_side 1024` (useful for limited GPU memory)

- **`--max_instances`**  
  Maximum number of detected instances to visualize.  
  **Default:** `100`
