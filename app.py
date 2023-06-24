# Import general libraries
import os
import numpy as np

# Import Tensorflow & Keras
import tensorflow as tf
from tensorflow.keras.utils import load_img
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image

# Import Flask
from flask import Flask, render_template, request, redirect, url_for, session
from werkzeug.utils import secure_filename


# Konfigurasi Flask
app = Flask(__name__)
app.config["UPLOAD_FOLDER"] = os.path.join("static", "uploads")
# app.config[
#     "UPLOAD_FOLDER"
# ] = "C:\\Users\\ASUS\\Downloads\\punya zalfa\\deyegnose-web\\static\\uploads"
app.config["ALLOWED_EXTENSIONS"] = {"png", "jpg", "jpeg"}
app.config["MAX_CONTENT_LENGTH"] = 5 * 1024 * 1024  # 5MB
app.secret_key = "deyegnose-web"


# Fungsi bantu untuk memeriksa ekstensi file yang diizinkan
def allowed_file(filename):
    return (
        "." in filename
        and filename.rsplit(".", 1)[1].lower() in app.config["ALLOWED_EXTENSIONS"]
    )


# Halaman utama
@app.route("/")
def landing():
    return render_template("landing.html")


# Halaman About
@app.route("/about")
def about():
    return render_template("about.html")


# Halaman Try
@app.route("/try", methods=["GET", "POST"])
def try_it():
    if request.method == "POST":
        # Ambil data form
        nama = request.form["nama"]
        email = request.form["email"]
        gambar = request.files["image"]
        usia = request.form["usia"]
        jenis_pekerjaan = request.form["jenis-pekerjaan"]
        riwayat_penyakit = request.form["riwayat-penyakit"]
        riwayat_luka = request.form["riwayat-luka"]

        # Verifikasi file gambar yang diunggah
        if gambar and allowed_file(gambar.filename):
            filename = secure_filename(gambar.filename)
            filepath = os.path.join(app.config["UPLOAD_FOLDER"], filename)
            gambar.save(os.path.join(app.config["UPLOAD_FOLDER"], filename))
            session["uploaded_img_file_path"] = os.path.join(
                app.config["UPLOAD_FOLDER"], filename
            )

            print("filename:", filename)
            print("\nfilepath:", filepath)

            # Lakukan prediksi dengan model TensorFlow disini
            prediction, confidence_score = predict_image(
                os.path.join(app.config["UPLOAD_FOLDER"], filename)
            )

            # Kirim data ke halaman Result
            return redirect(
                url_for(
                    "result",
                    nama=nama,
                    email=email,
                    usia=usia,
                    image_url=filepath,
                    filename=filename,
                    jenis_pekerjaan=jenis_pekerjaan,
                    riwayat_penyakit=riwayat_penyakit,
                    riwayat_luka=riwayat_luka,
                    prediction=prediction,
                    confidence_score=confidence_score,
                )
            )

    return render_template("try.html")


# Halaman Result
@app.route("/result")
def result():
    nama = request.args.get("nama")
    email = request.args.get("email")
    usia = request.args.get("usia")
    image_url = request.args.get("image_url")
    image_path = session.get("uploaded_img_file_path", None)
    filename = request.args.get("filename")
    jenis_pekerjaan = request.args.get("jenis_pekerjaan")
    riwayat_penyakit = request.args.get("riwayat_penyakit")
    riwayat_luka = request.args.get("riwayat_luka")
    prediction = request.args.get("prediction")
    confidence_score = request.args.get("confidence_score")

    return render_template(
        "result.html",
        nama=nama,
        email=email,
        usia=usia,
        image_url=image_url,
        image_path=image_path,
        filename=filename,
        jenis_pekerjaan=jenis_pekerjaan,
        riwayat_penyakit=riwayat_penyakit,
        riwayat_luka=riwayat_luka,
        prediction=prediction,
        confidence_score=confidence_score,
    )


# Fungsi untuk melakukan prediksi gambar
def predict_image(img_path):
    # Implementasikan model TensorFlow Anda disini
    # Mengembalikan hasil prediksi (katarak atau normal)
    class_names = ["Cataract", "Normal"]
    final_model = load_model("./model/final_model_50epoch_2_class.h5", compile=True)
    img = load_img(img_path, target_size=(100, 100))
    img_array = image.img_to_array(img)
    img_array = tf.expand_dims(img_array, axis=0)

    print("img_array shape:", img_array.shape)
    print("img_array type:", type(img_array))

    prediction_result = final_model.predict(img_array)

    print("prediction_result:", prediction_result)
    print("prediction_result[0][0]:", prediction_result[0][0])
    print("prediction_result shape:", prediction_result.shape)
    print("prediction_result type:", type(prediction_result))

    confidence_score = 0
    if prediction_result[0][0] > 0.5:
        predicted_class = "Mata Normal"
        confidence_score = prediction_result[0][0] * 100
    else:
        predicted_class = "Katarak"
        confidence_score = (1 - prediction_result[0][0]) * 100

    print("Prediction class:", predicted_class)
    print("Confidence score:", confidence_score)

    return predicted_class, confidence_score


if __name__ == "__main__":
    app.run(debug=True)
