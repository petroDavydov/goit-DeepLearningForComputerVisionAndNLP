from ultralytics import YOLO  # type: ignore
import cv2
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import yaml
import random
import os
import warnings
warnings.filterwarnings("ignore")


# -----------------------------
# Клас CFG для зберігання налаштувань
class CFG:
    DATASET_DIR = './indoreObjectDetection/'   # локальний шлях
    OUTPUT_DIR = './runs_indoor/'           # директорія для результатів

    EXP_NAME = 'IndoorObjects_YOLOv9s'

    MODEL = 'yolov9s.pt'
    IMG_SIZE = 640

    DEBUG = False

    EPOCHS = 1 if DEBUG else 70
    BATCH_SIZE = 16
    OPTIMIZER = 'auto'

    CLASSES = ['door', 'cabinetDoor', 'refrigeratorDoor',
               'window', 'chair', 'table', 'cabinet', 'couch',
               'openedDoor', 'pole']
    NUM_CLASSES = len(CLASSES)

    SEED = 42


print(f"Загальна кількість класів:\n {CFG.NUM_CLASSES}")


random.seed(CFG.SEED)
np.random.seed(CFG.SEED)

# -----------------------------
# Підготовка data.yaml
yaml_config = {
    'train': 'train/images',
    'val':  'valid/images',
    'test': 'test/images',
    'nc': CFG.NUM_CLASSES,
    'names': CFG.CLASSES
}

yaml_file_path = os.path.join(CFG.DATASET_DIR, 'data.yaml')

with open(yaml_file_path, 'w') as file:
    yaml.dump(yaml_config, file, default_flow_style=False)

print("Вміст data.yaml:")
with open(yaml_file_path, 'r') as f:
    print(f.read())


# -----------------------------
# Візуалізація випадкових зображень
def visualize_random_images(dataset_path, num_images=9, seed=CFG.SEED):
    random.seed(seed)
    image_dir = os.path.join(dataset_path, 'images')
    label_dir = os.path.join(dataset_path, 'labels')

    image_files = os.listdir(image_dir)
    random_images = random.sample(image_files, num_images)

    plt.figure(figsize=(12, 12))

    for i, img_name in enumerate(random_images):
        img_path = os.path.join(image_dir, img_name)
        label_path = os.path.join(label_dir, Path(img_name).stem + '.txt')

        image = cv2.imread(img_path)
        if image is None:
            print(f"Файл по вказаному шляху не відкрився: {img_path}")
            continue

        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        h, w, _ = image.shape

        if os.path.exists(label_path):
            with open(label_path, 'r') as f:
                for line in f.readlines():
                    class_id, x_center, y_center, width, height = map(
                        float, line.split())

                    x1 = int((x_center - width/2) * w)
                    y1 = int((y_center - height/2) * h)
                    x2 = int((x_center + width/2) * w)
                    y2 = int((y_center + height/2) * h)

                    color = (random.randint(0, 255), random.randint(
                        0, 255), random.randint(0, 255))
                    cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)

                    class_name = CFG.CLASSES[int(class_id)]
                    cv2.putText(image, class_name, (x1, y1 - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

        plt.subplot(3, 3, i + 1)
        plt.imshow(image)
        plt.axis('off')
        plt.title(img_name)

    plt.tight_layout()
    plt.pause(0.5)
    plt.close()


# ----------IF_NAME == MAIN-------------------
if __name__ == "__main__":
    print("Приклади зображень з навчальної вибірки:")
    visualize_random_images(os.path.join(CFG.DATASET_DIR, 'train'))

    print(f"Навчання на {CFG.EPOCHS} епохах, batch={CFG.BATCH_SIZE}, img_size={CFG.IMG_SIZE}")


    # Ініціалізація моделі
    model = YOLO(CFG.MODEL)

    # Навчання
    model.train(
        data=yaml_file_path,
        epochs=CFG.EPOCHS,
        batch=CFG.BATCH_SIZE,
        imgsz=CFG.IMG_SIZE,
        optimizer=CFG.OPTIMIZER,
        name=CFG.EXP_NAME,
        project=CFG.OUTPUT_DIR,
        seed=CFG.SEED
    )

    # правильний доступ до директорії результатів
    results_path = Path(CFG.OUTPUT_DIR) / CFG.EXP_NAME
    print(f"Результати збережено:\n {results_path}")

    # Завантаження найкращої моделі
    best_model_path = results_path / 'weights/best.pt'
    model = YOLO(best_model_path)

    # Прогнози на валідаційних зображеннях
    print("\n Прогнозуємо на валідаційній вибірці:\n")
    val_image_dir = os.path.join(CFG.DATASET_DIR, 'valid/images')
    image_files = os.listdir(val_image_dir)
    random_images = random.sample(image_files, 9)

    plt.figure(figsize=(15, 15))

    for i, img_name in enumerate(random_images):
        img_path = os.path.join(val_image_dir, img_name)
        res = model.predict(source=img_path, conf=0.4)

        im_array = res[0].plot()
        im = Image.fromarray(im_array[..., ::-1])

        plt.subplot(3, 3, i + 1)
        plt.imshow(im)
        plt.axis('off')
        plt.title(img_name)

    plt.tight_layout()
    plt.pause(0.5)
    plt.close()
