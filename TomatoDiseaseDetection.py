from flask import Flask, render_template, request
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import image

from PIL import Image

app = Flask(__name__)

model = tf.keras.models.load_model(r"TomatoDiseaseDetection_optimizedvscode2nd.h5")

CLASS_NAMES = ["Tomato___Bacterial_spot", "Tomato___Early_blight", "Tomato___Late_blight",
               "Tomato___Leaf_Mold", "Tomato___Septoria_leaf_spot", "Tomato___Spider_mites Two-spotted_spider_mite",
               "Tomato___Target_Spot", "Tomato___Tomato_Yellow_Leaf_Curl_Virus", "Tomato___Tomato_mosaic_virus",
               "Tomato___healthy"]

@app.route("/", methods=["GET"])
def ind():
    return render_template("index.html", result=None)

@app.route("/predict", methods=["POST"])
def home():
    if "file" not in request.files:
        return render_template("index.html", result="No file uploaded")

    file = request.files["file"]
    if file.filename == "":
        return render_template("index.html", result="No selected file")

    print(f"name: {file.filename}")
    img = Image.open(file.stream).convert("RGB")
    img = img.resize((224, 224))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = img_array / 255.0

    predictions = model.predict(img_array)
    predicted_class = CLASS_NAMES[np.argmax(predictions)]
    confidence = float(np.max(predictions))

    result = f"{predicted_class} (Confidence: {confidence:.2f}) (fileName: {file.filename})"
    return render_template("index.html", result=result)

if __name__ == "__main__":
    app.run(debug=True)