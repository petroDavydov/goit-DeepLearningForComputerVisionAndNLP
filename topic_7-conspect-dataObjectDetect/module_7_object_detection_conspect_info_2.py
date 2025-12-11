# working on 3 epoches
# ----------------------
import torch
import matplotlib
from ultralytics import YOLO  # type: ignore
import cv2
from PIL import Image
import seaborn as sns
from matplotlib.patches import Rectangle
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import yaml
import random
import glob
import re
import os
import warnings
warnings.filterwarnings("ignore")


# import IPython.display as display
# -----------------------

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f'Using device: {device}')

# ---------------------


class CFG:
    DEBUG = True  # Set to True to make quick experiments
    # Specifies the fraction of the dataset to use for training. Allows for training on a subset of the full dataset, useful for experiments or when resources are limited.
    FRACTION = 0.05 if DEBUG else 1.0
    SEED = 42

    # classes
    CLASSES = ['Hardhat', 'Mask', 'NO-Hardhat', 'NO-Mask',
               'NO-Safety Vest', 'Person', 'Safety Cone',
               'Safety Vest', 'machinery', 'vehicle']
    NUM_CLASSES_TO_TRAIN = len(CLASSES)

    # training
    EPOCHS = 3 if DEBUG else 70  # 100
    BATCH_SIZE = 8  # 16

    BASE_MODEL = 'yolov9e'  # yolov8n, yolov8s, yolov8m, yolov8l, yolov8x, yolov9c, yolov9e
    BASE_MODEL_WEIGHTS = f'{BASE_MODEL}.pt'
    EXP_NAME = f'ppe_css_{EPOCHS}_epochs'

    OPTIMIZER = 'auto'  # SGD, Adam, Adamax, AdamW, NAdam, RAdam, RMSProp, auto
    LR = 1e-3
    # Final learning rate as a fraction of the initial rate = (lr0 * lrf), used in conjunction with schedulers to adjust the learning rate over time.
    LR_FACTOR = 0.01
    # L2 regularization term, penalizing large weights to prevent overfitting.
    WEIGHT_DECAY = 5e-4
    DROPOUT = 0.0
    PATIENCE = 20
    PROFILE = False
    LABEL_SMOOTHING = 0.0

    # paths
    CUSTOM_DATASET_DIR = './dataObjectDetect/css-data-object-detect/'
    OUTPUT_DIR = './'

    # -----------------------------


dict_file = {
    'train': os.path.join(CFG.CUSTOM_DATASET_DIR, 'train'),
    'val': os.path.join(CFG.CUSTOM_DATASET_DIR, 'valid'),
    'test': os.path.join(CFG.CUSTOM_DATASET_DIR, 'test'),
    'nc': CFG.NUM_CLASSES_TO_TRAIN,
    'names': CFG.CLASSES
}

with open(os.path.join(CFG.OUTPUT_DIR, 'data.yaml'), 'w+') as file:
    yaml.dump(dict_file, file)

# ---------------------------

# read yaml file created


def read_yaml_file(file_path=CFG.CUSTOM_DATASET_DIR):
    with open(file_path, 'r') as file:
        try:
            data = yaml.safe_load(file)
            return data
        except yaml.YAMLError as e:
            print("Error reading YAML:", e)
            return None

# print it with newlines


def print_yaml_data(data):
    formatted_yaml = yaml.dump(data, default_style=None)
    print(formatted_yaml)


file_path = os.path.join(CFG.OUTPUT_DIR, 'data.yaml')
yaml_data = read_yaml_file(file_path)

if yaml_data:
    print_yaml_data(yaml_data)


# ----------------------

def display_image(image, print_info=True, hide_axis=False):
    if isinstance(image, str):  # Check if it's a file path
        img = Image.open(image)
        plt.imshow(img)
    elif isinstance(image, np.ndarray):  # Check if it's a NumPy array
        image = image[..., ::-1]  # BGR to RGB
        img = Image.fromarray(image)
        plt.imshow(img)
    else:
        raise ValueError("Unsupported image format")

    if print_info:
        print('Type: ', type(img), '\\n')
        print('Shape: ', np.array(img).shape, '\\n')

    if hide_axis:
        plt.axis('off')

    plt.tight_layout()
    plt.pause(1)
    plt.close()


example_image_path = CFG.CUSTOM_DATASET_DIR + \
    'train/images/-2297-_png_jpg.rf.9fff3740d864fbec9cda50d783ad805e.jpg'
display_image(example_image_path, print_info=True, hide_axis=False)

# ------------------------------


def plot_random_images_from_folder(folder_path, num_images=20, seed=CFG.SEED):

    random.seed(seed)

    # Get a list of image files in the folder
    image_files = [f for f in os.listdir(folder_path) if f.endswith(
        ('.jpg', '.png', '.jpeg', '.gif'))]

    # Ensure that we have at least num_images files to choose from
    if len(image_files) < num_images:
        raise ValueError("Not enough images in the folder")

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

# ------------------------------------


def get_image_properties(image_path):
    # Read the image file
    img = cv2.imread(image_path)

    # Check if the image file is read successfully
    if img is None:
        raise ValueError("Could not read image file")

    # Get image properties
    properties = {
        "width": img.shape[1],
        "height": img.shape[0],
        "channels": img.shape[2] if len(img.shape) == 3 else 1,
        "dtype": img.dtype,
    }

    return properties


img_properties = get_image_properties(example_image_path)
print(img_properties)

# ----------------------------------------

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
                cls_id = line.split()[0]   # правильний ID класу
                class_count[class_idx[cls_id]] += 1

    data_volume = len(os.listdir(os.path.join(
        CFG.CUSTOM_DATASET_DIR, mode, 'images')))
    class_info.append({'Mode': mode, **class_count,
                      'Data_Volume': data_volume})

dataset_stats_df = pd.DataFrame(class_info)
print(dataset_stats_df)

# -----------------------------

# Create subplots with 1 row and 3 columns
fig, axes = plt.subplots(1, 3, figsize=(15, 5))

# Plot vertical bar plots for each mode in subplots
for i, mode in enumerate(['train', 'valid', 'test']):
    sns.barplot(
        data=dataset_stats_df[dataset_stats_df['Mode']
                              == mode].drop(columns='Mode'),
        orient='v',
        ax=axes[i],
        palette='Set2'
    )

    axes[i].set_title(f'{mode.capitalize()} Class Statistics')
    axes[i].set_xlabel('Classes')
    axes[i].set_ylabel('Count')
    axes[i].tick_params(axis='x', rotation=90)

    # Add annotations on top of each bar
    for p in axes[i].patches:
        axes[i].annotate(f"{int(p.get_height())}", (p.get_x() + p.get_width() / 2., p.get_height()),
                         ha='center', va='center', fontsize=8, color='black', xytext=(0, 5),
                         textcoords='offset points')

plt.tight_layout()
plt.pause(1)
plt.close()

# ---------------------------------------------------

model = YOLO(CFG.BASE_MODEL_WEIGHTS)

results = model.predict(
    source=example_image_path,
    classes=[0],
    conf=0.30,
    #     device = [0,1], # inference with dual GPU
    device=0,  # inference with CPU
    imgsz=(img_properties['height'], img_properties['width']),
    save=True,
    save_txt=True,
    save_conf=True,
    exist_ok=True,
)

example_image_inference_output = example_image_path.split('/')[-1]
display_image(f'runs/detect/predict/{example_image_inference_output}')
# -------------------------------------


model = YOLO(CFG.BASE_MODEL_WEIGHTS)
# -----------------------------------------


if __name__ == '__main__':
    import torch.multiprocessing
    torch.multiprocessing.set_start_method('spawn', force=True)

    model.train(
        data=os.path.join(CFG.OUTPUT_DIR, 'data.yaml'),

        task='detect',

        imgsz=(img_properties['height'], img_properties['width']),

        epochs=CFG.EPOCHS,
        batch=CFG.BATCH_SIZE,
        optimizer=CFG.OPTIMIZER,
        lr0=CFG.LR,
        lrf=CFG.LR_FACTOR,
        weight_decay=CFG.WEIGHT_DECAY,
        dropout=CFG.DROPOUT,
        fraction=CFG.FRACTION,
        patience=CFG.PATIENCE,
        profile=CFG.PROFILE,
        label_smoothing=CFG.LABEL_SMOOTHING,

        name=f'{CFG.BASE_MODEL}_{CFG.EXP_NAME}',
        seed=CFG.SEED,

        val=True,
        amp=True,
        exist_ok=True,
        resume=False,
        device=device,  # [0,1]
        #     device = None, # CPU run
        verbose=False,
    )

# -----------------------------------------------

    # Export the model
    model.export(
        format='onnx',  # openvino, onnx, engine, tflite
        imgsz=(img_properties['height'], img_properties['width']),
        half=False,
        int8=False,
        simplify=False,
        nms=False,
    )
