import numpy as np
from PIL import Image
import io

def read_upimg(uploaded_file, image_size=(100, 100)):
    try:
        image = Image.open(uploaded_file).convert('L')
        image = image.resize(image_size)
        image_vector = np.asarray(image, dtype=np.float64).flatten()
        return image, image_vector
    except Exception as e:
        print(f"Gagal membaca gambar: {e}")
        return None, None
