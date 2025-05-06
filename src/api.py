import os
import pandas as pd
import numpy as np
from pathlib import Path
from flask import Flask, request, jsonify
from src.models.predict_model import load_model_pickle, load_model_joblib

# Inizializza l'app Flask
app = Flask(__name__)

# Carica il modello all'avvio dell'applicazione
model = None

def load_model():
    """Carica il modello prima della prima richiesta"""
    global model
    try:
        model = load_model_pickle()  # oppure load_model_joblib()
        print("Modello caricato con successo!")
    except Exception as e:
        print(f"Errore durante il caricamento del modello: {e}")

# Utilizza un decoratore alternativo per caricare il modello
@app.before_request
def before_request():
    if model is None:
        load_model()

@app.route('/health', methods=['GET'])
def health_check():
    """Endpoint per verificare lo stato dell'API"""
    if model is not None:
        return jsonify({"status": "ok", "message": "API funzionante e modello caricato"})
    else:
        return jsonify({"status": "error", "message": "Modello non caricato"}), 500

@app.route('/predict', methods=['POST'])
def predict():
    """Endpoint per fare predizioni"""
    # Verifica che il modello sia caricato
    if model is None:
        try:
            load_model()
        except Exception as e:
            return jsonify({"error": f"Impossibile caricare il modello: {str(e)}"}), 500
    
    # Ottieni i dati dalla richiesta JSON
    data = request.get_json()
    
    # Verifica che i dati contengano tutte le feature necessarie
    required_features = ['region_match', 'user_likes_for_category', 'event_popularity']
    
    if not all(feature in data for feature in required_features):
        missing = [f for f in required_features if f not in data]
        return jsonify({"error": f"Mancano le seguenti feature: {missing}"}), 400
    
    try:
        # Prepara i dati per la predizione
        input_data = pd.DataFrame([data])
        
        # Assicurati che ci siano solo le feature necessarie e nell'ordine corretto
        input_data = input_data[required_features]
        
        # Effettua la predizione
        prediction = model.predict(input_data)[0]
        
        # Restituisci il risultato
        return jsonify({
            "prediction": float(prediction),
            "input_data": data
        })
    
    except Exception as e:
        return jsonify({"error": f"Errore durante la predizione: {str(e)}"}), 500

@app.route('/batch-predict', methods=['POST'])
def batch_predict():
    """Endpoint per fare predizioni su un batch di dati"""
    # Verifica che il modello sia caricato
    if model is None:
        try:
            load_model()
        except Exception as e:
            return jsonify({"error": f"Impossibile caricare il modello: {str(e)}"}), 500
    
    # Ottieni i dati dalla richiesta JSON
    data = request.get_json()
    
    if not isinstance(data, list):
        return jsonify({"error": "I dati devono essere una lista di oggetti"}), 400
    
    # Verifica che i dati contengano tutte le feature necessarie
    required_features = ['region_match', 'user_likes_for_category', 'event_popularity']
    
    for item in data:
        if not all(feature in item for feature in required_features):
            missing = [f for f in required_features if f not in item]
            return jsonify({"error": f"Mancano le seguenti feature nell'elemento {item}: {missing}"}), 400
    
    try:
        # Prepara i dati per la predizione
        input_data = pd.DataFrame(data)
        
        # Assicurati che ci siano solo le feature necessarie e nell'ordine corretto
        input_data = input_data[required_features]
        
        # Effettua la predizione
        predictions = model.predict(input_data).tolist()
        
        # Restituisci il risultato
        results = []
        for i, pred in enumerate(predictions):
            results.append({
                "prediction": float(pred),
                "input_data": data[i]
            })
        
        return jsonify(results)
    
    except Exception as e:
        return jsonify({"error": f"Errore durante la predizione batch: {str(e)}"}), 500


if __name__ == '__main__':
    # Carica il modello all'avvio
    load_model()
    # Avvia l'app Flask
    app.run(debug=True, host='0.0.0.0', port=5000)