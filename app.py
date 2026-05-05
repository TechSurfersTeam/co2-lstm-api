from flask import Flask, request, jsonify
import numpy as np
import tensorflow as tf
import pickle

app = Flask(__name__)

# Load model
model = tf.keras.models.model_from_json(open("config.json").read())
model.load_weights("model.weights.h5")

scaler_X = pickle.load(open("scaler_X_v3.pkl", "rb"))
scaler_y = pickle.load(open("scaler_y_v3.pkl", "rb"))

@app.route("/")
def home():
    return "CO2 Model API Running"

@app.route("/predict", methods=["POST"])
def predict():
    data = request.json

    co2 = np.array(data["co2"])
    occ = np.array(data["occupancy"])

    combined = np.column_stack((co2, occ))
    scaled = scaler_X.transform(combined).reshape(1,60,2)

    pred_scaled = model.predict(scaled)
    pred = scaler_y.inverse_transform(pred_scaled)

    return jsonify({"prediction": float(pred[0][0])})
