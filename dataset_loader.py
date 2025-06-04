import os
import numpy as np
from PIL import Image

def load_dataset(dataset_path, image_size=(100, 100)):
    images = []
    labels = []
    filenames = []

    for root, _, files in os.walk(dataset_path):
        for file in files:
            if file.lower().endswith(('jpg', 'jpeg', 'png')):
                file_path = os.path.join(root, file)
                try:
                    img = Image.open(file_path).convert('L')  
                    img = img.resize(image_size)
                    img_data = np.asarray(img, dtype=np.float64).flatten()

                    images.append(img_data)
                    labels.append(os.path.basename(root)) 
                    filenames.append(file)
                except Exception as e:
                    print(f"Gagal memproses {file_path}: {e}")

    return np.array(images), labels, filenames