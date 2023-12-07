"""
	Contoh Deloyment untuk Domain Computer Vision (CV)
	Orbit Future Academy - AI Mastery - KM Batch 3
	Tim Deployment
	2022
"""
import os
from keras.models import load_model

# =[Modules dan Packages]========================

from flask import Flask, render_template, request, jsonify
from werkzeug.utils import secure_filename
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (
    Conv2D,
    MaxPooling2D,
    Flatten,
    Dense,
    Activation,
    Dropout,
    LeakyReLU,
)
from PIL import Image

# =[Variabel Global]=============================

app = Flask(__name__, static_url_path="/static")

app.config["MAX_CONTENT_LENGTH"] = 1024 * 1024
app.config["UPLOAD_EXTENSIONS"] = [".jpg", ".jpeg", ".png", ".JPG", "JPEG", "PNG"]
app.config["UPLOAD_PATH"] = "./static/images/uploads/"

model = None

NUM_CLASSES = 2
cifar10_classes = ["fire", "no fire"]

# =[Routing]=====================================


# [Routing untuk Halaman Utama atau Home]
@app.route("/", methods=["GET", "POST"])
def beranda():
    if request.method == "POST":
        model.save()
    else:
        return render_template("index.html")


# [Routing untuk API]
@app.route("/api/deteksi", methods=["POST"])
def apiDeteksi():
    # Set nilai default untuk hasil prediksi dan gambar yang diprediksi

    # Get File Gambar yg telah diupload pengguna
    uploaded_file = request.files["file"]
    filename = secure_filename(uploaded_file.filename)

    # Periksa apakah ada file yg dipilih untuk diupload
    if filename != "":
        # Set/mendapatkan extension dan path dari file yg diupload
        file_ext = os.path.splitext(filename)[1]
        gambar_prediksi = "/static/images/uploads/" + filename

        # Periksa apakah extension file yg diupload sesuai (jpg)
        if file_ext in app.config["UPLOAD_EXTENSIONS"]:
            print("File yang diupload: ", filename)
            # Simpan Gambar
            uploaded_file.save(os.path.join(app.config["UPLOAD_PATH"], filename))

            # Memuat Gambar
            test_image = Image.open("." + gambar_prediksi)

            # Mengubah Ukuran Gambar
            test_image_resized = test_image.resize((256, 256))
            print("Ukuran Gambar: ", test_image_resized.size)

            # Konversi Gambar ke Array
            image_array = image.img_to_array(test_image_resized)
            img_array = tf.expand_dims(image_array, 0)
            img_array = img_array / 255.0
            print("Dimensi Gambar: ", img_array)

            # Fit
            model.compile(
                optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"]
            )

            # Melakukan prediksi pada gambar uji
            print("Melakukan prediksi pada gambar uji...")
            predictions = model.predict(img_array)
            print("Hasil prediksi: ", predictions)

            # Menafsirkan hasil prediksi
            if predictions[0][0] > 0.5:
                prediction_result = "🌳Tidak Terbakar🌳"
            else:
                prediction_result = "🔥Terbakar🔥"

            # Return hasil
            # prediksi dengan format JSON
            return jsonify(
                {"prediksi": prediction_result, "gambar_prediksi": gambar_prediksi}
            )
        else:
            # Return hasil prediksi dengan format JSON
            gambar_prediksi = "(none)"
            prediction_result = "Tidak Ada"
            return jsonify(
                {"prediksi": prediction_result, "gambar_prediksi": gambar_prediksi}
            )


# =[Main]========================================

if __name__ == "__main__":
    # Load model yang telah ditraining
    # model = make_model()
    model = load_model("fire_detection_model.h5", compile=False)

    # Run Flask di localhost
    app.run(host="localhost", port=5000, debug=True)
