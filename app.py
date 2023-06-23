import os
from flask import Flask, render_template, request, redirect, url_for
from werkzeug.utils import secure_filename
# Import general libraries
import numpy as np

# Import Tensorflow & Keras
import tensorflow as tf
from tensorflow.keras.utils import load_img
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image

# Konfigurasi Flask
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/uploads'
app.config['ALLOWED_EXTENSIONS'] = {'png', 'jpg', 'jpeg'}
app.config['MAX_CONTENT_LENGTH'] = 5 * 1024 * 1024  # 5MB

# Fungsi bantu untuk memeriksa ekstensi file yang diizinkan
def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

# Halaman utama
@app.route('/')
def landing():
    return render_template('landing.html')

# Halaman About
@app.route('/about')
def about():
    return render_template('about.html')

# Halaman Try
@app.route('/try', methods=['GET', 'POST'])
def try_it():
    if request.method == 'POST':
        # Ambil data form
        name = request.form['name']
        email = request.form['email']
        image = request.files['image']
        dropdown_value = request.form['dropdown']

        # Verifikasi file gambar yang diunggah
        if image and allowed_file(image.filename):
            filename = secure_filename(image.filename)
            image.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
            # Lakukan prediksi dengan model TensorFlow disini
            prediction = predict_image(os.path.join(app.config['UPLOAD_FOLDER'], filename))
            # Kirim data ke halaman Result
            return redirect(url_for('result', name=name, email=email, prediction=prediction))

    return render_template('try.html')

# Halaman Result
@app.route('/result')
def result():
    name = request.args.get('name')
    email = request.args.get('email')
    prediction = request.args.get('prediction')
    return render_template('result.html', name=name, email=email, prediction=prediction)

# Fungsi bantu untuk melakukan prediksi gambar
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
    print("prediction_result shape:", prediction_result.shape)
    print("prediction_result type:", type(prediction_result))

    if prediction_result[0] > 0.5:
        predicted_class = class_names[1]
    else:
        predicted_class = class_names[0]

    print("Prediction class:", predicted_class)

    return predicted_class

if __name__ == '__main__':
    app.run(debug=True)
