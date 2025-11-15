import os
import json
import argparse
from pathlib import Path
from sklearn.model_selection import train_test_split
from PIL import Image

def convert_labelme_to_coco(data_dir: str, output_dir: str, test_size: float = 0.2):
    """
    Converts a dataset from LabelMe format to COCO instance segmentation format.

    Args:
        data_dir (str): Path to the directory containing .tif and .json files.
        output_dir (str): Path to save the output COCO json files.
        test_size (float): Proportion of the dataset to include in the validation split.
    """
    
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # coco structure
    coco_train = {
        "images": [],
        "annotations": [],
        "categories": [{"id": 1, "name": "fallen_tree", "supercategory": "tree"}]
    }
    coco_val = {
        "images": [],
        "annotations": [],
        "categories": [{"id": 1, "name": "fallen_tree", "supercategory": "tree"}]
    }

    #  get file lists and split
    all_files = sorted([Path(f).stem for f in os.listdir(data_dir) if f.endswith('.json')])
    train_files, val_files = train_test_split(all_files, test_size=test_size, random_state=42)

    print(f"Found {len(all_files)} total images.")
    print(f"Splitting into {len(train_files)} training and {len(val_files)} validation images.")

    # process files for each split 
    annotation_id_counter = 1
    for split, files, coco_dict in [("train", train_files, coco_train), ("val", val_files, coco_val)]:
        print(f"\nProcessing {split} split...")
        
        for image_id, filename in enumerate(files, 1):
            json_path = Path(data_dir) / f"{filename}.json"
            image_path = Path(data_dir) / f"{filename}.tif"

            # ---  read image to get dimensions ---
            try:
                with Image.open(image_path) as img:
                    width, height = img.size
            except Exception as e:
                print(f"Warning: Could not read image {image_path}. Skipping. Error: {e}")
                continue

            #  rdd image info to coco dict
            image_info = {
                "id": image_id,
                "file_name": f"{filename}.tif",
                "width": width,
                "height": height
            }
            coco_dict["images"].append(image_info)

            # read labelme json and add annotations 
            with open(json_path, 'r') as f:
                labelme_data = json.load(f)

            for shape in labelme_data["shapes"]:
                raw_label = shape.get("label", "")
                label_norm = raw_label.strip().lower()

                if label_norm == "fallen_tree":
                    points = shape["points"]
                    if len(points) < 3:
                        # skip degenerate polygons
                        continue
                    # coco segmentation format: [x1, y1, x2, y2, ...]
                    segmentation = [coord for point in points for coord in point]
                    
                    # coco bounding box format: [xmin, ymin, width, height]
                    x_coords = [p[0] for p in points]
                    y_coords = [p[1] for p in points]

                    # bbox
                    xmin = min(x_coords)
                    ymin = min(y_coords)
                    xmax = max(x_coords)
                    ymax = max(y_coords)
                    bbox = [xmin, ymin, xmax - xmin, ymax - ymin]

                    # polygon area using shoelace formula
                    area = 0.0
                    n = len(points)
                    for i in range(n):
                        j = (i + 1) % n
                        area += x_coords[i] * y_coords[j] - x_coords[j] * y_coords[i]
                    area = abs(area) * 0.5


                    annotation_info = {
                        "id": annotation_id_counter,
                        "image_id": image_id,
                        "category_id": 1, 
                        "segmentation": [segmentation],
                        "bbox": bbox,
                        "area": area,
                        "iscrowd": 0
                    }
                    coco_dict["annotations"].append(annotation_info)
                    annotation_id_counter += 1
                    
        
        # save the coco json file 
        output_json_path = output_path / f"{split}_coco.json"
        with open(output_json_path, 'w') as f:
            json.dump(coco_dict, f, indent=4)
        print(f"Successfully created {output_json_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert LabelMe annotations to COCO format.")
    parser.add_argument(
        "--data_dir",
        type=str,
        default="/workspace/bloom/custom-dataset",
        help="Path to the directory with LabelMe .json and image files.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="/workspace/bloom/data/coco_formatted",
        help="Directory to save the output COCO .json files.",
    )
    args = parser.parse_args()
    convert_labelme_to_coco(args.data_dir, args.output_dir)