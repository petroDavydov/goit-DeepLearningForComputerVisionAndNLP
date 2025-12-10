# need some corrections
# -----------------------
# -*- coding: utf-8 -*-
"""
module_7_object_detection_conspect_info.py
Converted from Colab notebook for VSCode
"""

import matplotlib.image as mpimg
import glob
import torch
from ultralytics import YOLO  # type: ignore
import cv2
from PIL import Image
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import yaml
import random
import os
import warnings
warnings.filterwarnings("ignore")

# -----------------------------------------
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f'Using device: {device}')

# ---------------- CONFIG ----------------


class CFG:
    DEBUG = False
    FRACTION = 0.05 if DEBUG else 1.0
    SEED = 42

    CLASSES = ['Hardhat', 'Mask', 'NO-Hardhat', 'NO-Mask',
               'NO-Safety Vest', 'Person', 'Safety Cone',
               'Safety Vest', 'machinery', 'vehicle']
    NUM_CLASSES_TO_TRAIN = len(CLASSES)

    EPOCHS = 3 if DEBUG else 70
    BATCH_SIZE = 8

    BASE_MODEL = 'yolov9e'
    BASE_MODEL_WEIGHTS = f'{BASE_MODEL}.pt'
    EXP_NAME = f'ppe_css_{EPOCHS}_epochs'

    CUSTOM_DATASET_DIR = './dataObjectDetect/css-data-object-detect/'
    OUTPUT_DIR = './'


# ---------------- YAML ----------------
dict_file = {
    'train': os.path.join(CFG.CUSTOM_DATASET_DIR, 'train'),
    'val': os.path.join(CFG.CUSTOM_DATASET_DIR, 'valid'),
    'test': os.path.join(CFG.CUSTOM_DATASET_DIR, 'test'),
    'nc': CFG.NUM_CLASSES_TO_TRAIN,
    'names': CFG.CLASSES
}
with open(os.path.join(CFG.OUTPUT_DIR, 'data.yaml'), 'w+') as file:
    yaml.dump(dict_file, file)

# ---------------- IMAGE UTILS ----------------


def display_image(image, print_info=True, hide_axis=False):
    if isinstance(image, str):
        img = Image.open(image)
        plt.imshow(img)
    elif isinstance(image, np.ndarray):
        image = image[..., ::-1]  # BGR to RGB
        img = Image.fromarray(image)
        plt.imshow(img)
    else:
        raise ValueError("Unsupported image format")

    if print_info:
        print('Type: ', type(img))
        print('Shape: ', np.array(img).shape)

    if hide_axis:
        plt.axis('off')
    plt.tight_layout()
    plt.pause(1)
    plt.close()


def plot_random_images_from_folder(folder_path, num_images=20, seed=CFG.SEED):
    random.seed(seed)

    # Get a list of image files in the folder
    image_files = [f for f in os.listdir(folder_path) if f.endswith(
        ('.jpg', '.png', '.jpeg', '.gif'))]

    # Ensure that we have at least num_images files to choose from
    if len(image_files) < num_images:
        raise ValueError("Not enough images in the folder")

    if len(image_files) < num_images:
        num_images = len(image_files)

    # Randomly select num_images image files
    selected_files = random.sample(image_files, num_images)

    # Create a subplot grid
    num_cols = 5
    num_rows = (num_images + num_cols - 1) // num_cols
    fig, axes = plt.subplots(num_rows, num_cols, figsize=(12, 8))

    for i, file_name in enumerate(selected_files):
        # Open and display the image using PIL
        img = Image.open(os.path.join(folder_path, file_name))

        if num_rows == 1:
            ax = axes[i % num_cols]
        else:
            ax = axes[i // num_cols, i % num_cols]

        ax.imshow(img)
        ax.axis('off')
        # ax.set_title(file_name)

    # Remove empty subplots
    for i in range(num_images, num_rows * num_cols):
        if num_rows == 1:
            fig.delaxes(axes[i % num_cols])
        else:
            fig.delaxes(axes[i // num_cols, i % num_cols])

    plt.tight_layout()
    plt.pause(1)
    plt.close()


def get_image_properties(image_path):
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError("Could not read image file")
    return {
        "width": img.shape[1],
        "height": img.shape[0],
        "channels": img.shape[2] if len(img.shape) == 3 else 1,
        "dtype": img.dtype,
    }


# ---------------- EXAMPLE ----------------
example_image_path = os.path.join(
    CFG.CUSTOM_DATASET_DIR, 'train/images/-2297-_png_jpg.rf.9fff3740d864fbec9cda50d783ad805e.jpg')
display_image(example_image_path)

img_properties = get_image_properties(example_image_path)
print(img_properties)


# ---------------- SHOW 20 RANDOM TRAIN IMAGES ----------------
train_images_folder = os.path.join(CFG.CUSTOM_DATASET_DIR, 'train', 'images')
print("Folder:", train_images_folder, "| files:",
      len(os.listdir(train_images_folder)))
plot_random_images_from_folder(train_images_folder, num_images=20)


# ---------------- DATASET STATS ----------------
class_idx = {str(i): CFG.CLASSES[i] for i in range(CFG.NUM_CLASSES_TO_TRAIN)}
class_info = []

for mode in ['train', 'valid', 'test']:
    class_count = {CFG.CLASSES[i]: 0 for i in range(CFG.NUM_CLASSES_TO_TRAIN)}
    path = os.path.join(CFG.CUSTOM_DATASET_DIR, mode, 'labels')

    for file in os.listdir(path):
        if not file.endswith(".txt"):
            continue
        with open(os.path.join(path, file)) as f:
            for line in f:
                cls_id = line.split()[0]
                class_count[class_idx[cls_id]] += 1

    data_volume = sum(1 for f in os.listdir(path) if f.endswith(".txt"))
    for cls, cnt in class_count.items():
        class_info.append({
            'Mode': mode,
            'Class': cls,
            'Count': cnt,
            'Data_Volume': data_volume
        })

dataset_stats_df = pd.DataFrame(class_info)
with pd.option_context('display.max_columns', None):
    print(dataset_stats_df.head())

# побудова графіків статистики
fig, axes = plt.subplots(1, 3, figsize=(15, 5))
for i, mode in enumerate(['train', 'valid', 'test']):
    subset = dataset_stats_df[dataset_stats_df['Mode'] == mode]
    sns.barplot(
        data=subset,
        x="Class", y="Count", ax=axes[i], palette="Set2"
    )
    axes[i].set_title(f"{mode.capitalize()} Class Statistics")
    axes[i].tick_params(axis="x", rotation=90)

    for p in axes[i].patches:
        height = p.get_height()
        axes[i].annotate(f"{int(height)}",
                         (p.get_x() + p.get_width()/2., height),
                         ha="center", va="center", fontsize=8, color="black",
                         xytext=(0, 5), textcoords="offset points")
plt.tight_layout()
plt.pause(1)
plt.close()

# ---------------- YOLO ----------------
device = 0 if torch.cuda.is_available() else "cpu"
print("Використовую пристрій:", device)
model = YOLO(CFG.BASE_MODEL_WEIGHTS)

results = model.predict(
    source=example_image_path,
    classes=[0],
    conf=0.30,
    device=device,
    imgsz=(img_properties['height'], img_properties['width']),
    save=True, save_txt=True, save_conf=True, exist_ok=True,
)

predicted_images = glob.glob(
    "runs/detect/predict/*.jpg") + glob.glob("runs/detect/predict/*.png")
for img_path in predicted_images:
    display_image(img_path, hide_axis=True)

metrics_plots = glob.glob("runs/detect/exp*/results*.png")
for plot in metrics_plots:
    img = mpimg.imread(plot)
    plt.figure(figsize=(8, 6))
    plt.imshow(img)
    plt.axis("off")
    plt.tight_layout()
    plt.pause(1)
    plt.close()

model.export(
    format="onnx",
    imgsz=(img_properties['height'], img_properties['width']),
    half=False, int8=False, simplify=False, nms=False,
)

if __name__ == "__main__":
    # ---------------- TRAINING ----------------
    torch.multiprocessing.set_start_method('spawn', force=True)
    # Запуск навчання моделі на твоєму датасеті
    model.export(
        # шлях до yaml з описом датасету
        data=os.path.join(CFG.OUTPUT_DIR, 'data.yaml'),
        epochs=CFG.EPOCHS,                               # кількість епох
        batch=CFG.BATCH_SIZE,                            # розмір батчу
        imgsz=640,                                       # розмір зображень
        device=device,                                   # GPU або CPU
        project="runs/train",                            # куди зберігати результати
        name=CFG.EXP_NAME                                # назва експерименту
    )
