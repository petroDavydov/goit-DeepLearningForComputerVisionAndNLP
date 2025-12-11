import glob
import random
import os
from pathlib import Path

# Якщо CFG у твоєму основному файлі — імпортуємо
from module_7_object_detection_conspect_info_2 import CFG, display_image

# ============================
# ANALYSIS OF YOLO RESULTS
# ============================

exp_dir = f"{CFG.OUTPUT_DIR}runs/detect/{CFG.BASE_MODEL}_{CFG.EXP_NAME}"

print("\n=== SEARCHING IN DIRECTORY ===")
print(exp_dir)

# 1. Збираємо всі результати (графіки)
results_paths = [
    i for i in
    glob.glob(f'{exp_dir}/*.png') +
    glob.glob(f'{exp_dir}/*.jpg')
    if 'batch' not in i
]

print("\n=== RESULT IMAGES ===")
for p in results_paths:
    print(p)

# 2. Вибираємо випадковий val_batch для перегляду
validation_results_paths = [
    i for i in
    glob.glob(f'{exp_dir}/*.png') +
    glob.glob(f'{exp_dir}/*.jpg')
    if 'val_batch' in i
]

if len(validation_results_paths) >= 1:
    val_img_path = random.choice(validation_results_paths)
    print("\n=== RANDOM VAL BATCH IMAGE ===")
    print(val_img_path)
    display_image(val_img_path, print_info=False, hide_axis=True)
else:
    print("\nNo val_batch images found.")
