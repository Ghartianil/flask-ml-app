from flask import Flask, request, render_template
import pandas as pd
import numpy as np
import pickle
from tensorflow.keras.models import load_model

app = Flask(__name__)

# Load model and scalers once
scaler_X = pickle.load(open("G:/data analysis/scaler_X.pkl", "rb"))
scaler_y = pickle.load(open("G:/data analysis/scaler_y.pkl", "rb"))
model = load_model("G:/data analysis/lstm_model.h5", compile=False)

@app.route("/", methods=["GET", "POST"])
def index():
    prediction = None
    error = None

    if request.method == "POST":
        try:
            timestamp = request.form["timestamp"]
            temps = [float(request.form[f"temp{i}"]) for i in range(24)]

            # Time processing
            dt = pd.to_datetime(timestamp)
            times = [dt - pd.Timedelta(hours=i) for i in range(23, -1, -1)]

            features = [
                [t.dayofweek, t.month, temps[i], t.hour]
                for i, t in enumerate(times)
            ]
            features = np.array(features)

            # Scale & Predict
            X_scaled = scaler_X.transform(features).reshape(1, 24, 4)
            y_scaled = model.predict(X_scaled)
            y_pred = scaler_y.inverse_transform(y_scaled)

            prediction = round(float(y_pred[0][0]), 2)
        except Exception as e:
            error = str(e)

    return render_template("index.html", prediction=prediction, error=error)

if __name__ == "__main__":
    app.run(debug=True)