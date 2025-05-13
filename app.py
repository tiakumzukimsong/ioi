from flask import Flask, render_template, request
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
import os
from werkzeug.utils import secure_filename

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/uploads'

model = load_model('screen_classifier_model.h5')

def predict_image(img_path):
    img = image.load_img(img_path, target_size=(150, 150))
    img_array = image.img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    pred = model.predict(img_array)[0][0]
    print("pred->",pred)
    label = "Screen Image" if pred > 0.2 else "Non-Screen Image"
    confidence = pred if pred > 0.5 else 1 - pred
    print(confidence)
    return label, confidence * 100

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        uploaded_file = request.files['image']
        if uploaded_file.filename != '':
            filename = secure_filename(uploaded_file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            uploaded_file.save(filepath)

            label, confidence = predict_image(filepath)
            return render_template('index.html', label=label, confidence=confidence, image_url=filepath)

    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)
