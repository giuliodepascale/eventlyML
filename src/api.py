import os
import pandas as pd
from flask import Flask, request, jsonify
from flask_cors import CORS
from src.models.predict_model import load_model_pickle, load_model_joblib

# === Variabile globale ===
model = None

# Inizializza l'app Flask
app = Flask(__name__)

# Abilita CORS per i domini frontend
CORS(app, resources={r"/*": {"origins": [
    "https://evently-se-4-ai.vercel.app",
    "http://localhost:3000"
]}}, supports_credentials=True)

# Carica il modello
def load_model():
    global model
    try:
        model = load_model_pickle()  # o load_model_joblib()
        print("Modello caricato con successo!")
    except Exception as e:
        print(f"Errore durante il caricamento del modello: {e}")
        model = None

# Carica il modello prima di ogni richiesta
@app.before_request
def before_request():
    global model
    if model is None:
        load_model()

# Root route per evitare errori GET /
@app.route('/', methods=['GET'])
def index():
    return jsonify({"message": "API Evently attiva"})

# Health check
@app.route('/health', methods=['GET'])
def health_check():
    global model
    if model is not None:
        return jsonify({"status": "ok", "message": "API funzionante e modello caricato"})
    else:
        return jsonify({"status": "error", "message": "Modello non caricato"}), 500

# Singola predizione
@app.route('/predict', methods=['POST', 'OPTIONS'])
def predict():
    global model
    if model is None:
        try:
            load_model()
        except Exception as e:
            return jsonify({"error": f"Impossibile caricare il modello: {str(e)}"}), 500

    data = request.get_json()
    required_features = ['region_match', 'user_likes_for_category', 'event_popularity']

    if not all(feature in data for feature in required_features):
        missing = [f for f in required_features if f not in data]
        return jsonify({"error": f"Mancano le seguenti feature: {missing}"}), 400

    try:
        input_data = pd.DataFrame([data])[required_features]
        prediction = model.predict(input_data)[0]
        return jsonify({
            "prediction": float(prediction),
            "input_data": data
        })
    except Exception as e:
        return jsonify({"error": f"Errore durante la predizione: {str(e)}"}), 500

# Batch predizioni
@app.route('/batch-predict', methods=['POST', 'OPTIONS'])
def batch_predict():
    global model
    if model is None:
        try:
            load_model()
        except Exception as e:
            return jsonify({"error": f"Impossibile caricare il modello: {str(e)}"}), 500

    data = request.get_json()

    if not isinstance(data, list):
        return jsonify({"error": "I dati devono essere una lista di oggetti"}), 400

    required_features = ['region_match', 'user_likes_for_category', 'event_popularity']
    for item in data:
        if not all(feature in item for feature in required_features):
            missing = [f for f in required_features if f not in item]
            return jsonify({"error": f"Mancano le seguenti feature nell'elemento {item}: {missing}"}), 400

    try:
        input_data = pd.DataFrame(data)[required_features]
        predictions = model.predict(input_data).tolist()

        results = [
            {
                "prediction": float(pred),
                "input_data": data[i]
            }
            for i, pred in enumerate(predictions)
        ]

        return jsonify(results)
    except Exception as e:
        return jsonify({"error": f"Errore durante la predizione batch: {str(e)}"}), 500

# Avvio in produzione
if __name__ == '__main__':
    load_model()
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)
