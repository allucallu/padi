from flask import Flask, render_template, request
import os
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import requests

app = Flask(__name__)

MODEL_DIR = "model"
MODEL_PATH = os.path.join(MODEL_DIR, "best_model.h5")

# Ganti URL ini dengan link direct dari Hugging Face
MODEL_URL = "https://huggingface.co/spaces/calluu/Klasifikasi-penyakit-tanaman-padi/blob/main/best_model.h5"

# Kelas sesuai urutan folder pada training
class_names = ['Blast', 'Blight', 'Brownspot']

# Unduh model dari Hugging Face jika belum ada
def download_model():
    if not os.path.exists(MODEL_PATH):
        os.makedirs(MODEL_DIR, exist_ok=True)
        print("Mengunduh model dari Hugging Face...")
        response = requests.get(MODEL_URL)
        with open(MODEL_PATH, 'wb') as f:
            f.write(response.content)
        print("Model berhasil diunduh.")

# Load model
download_model()
model = load_model(MODEL_PATH)

@app.route('/', methods=['GET', 'POST'])
def index():
    prediction = None
    if request.method == 'POST':
        if 'image' not in request.files:
            return render_template('index.html', prediction="Tidak ada gambar.")
        file = request.files['image']
        if file.filename == '':
            return render_template('index.html', prediction="Pilih gambar terlebih dahulu.")
        
        # Simpan gambar sementara
        img_path = os.path.join("static", "uploads", file.filename)
        os.makedirs(os.path.dirname(img_path), exist_ok=True)
        file.save(img_path)

        # Preprocessing
        img = image.load_img(img_path, target_size=(224, 224))  # ukuran sesuai model Paduka
        img_array = image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0) / 255.0  # normalisasi

        # Prediksi
        pred = model.predict(img_array)
        pred_class = class_names[np.argmax(pred)]

        return render_template('index.html', prediction=pred_class, image_path=img_path)

    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)
