import pandas as pd
import numpy as np
import pickle
import joblib
from pathlib import Path
from sklearn.metrics import mean_squared_error, r2_score
import os


def load_model_pickle():
    """Carica il modello salvato con pickle"""
    model_path = Path('model-linear-regression') / 'linear_regression_model.pkl'
    with open(model_path, 'rb') as f:
        model = pickle.load(f)
    return model


def load_model_joblib():
    """Carica il modello salvato con joblib"""
    model_path = Path('model-linear-regression') / 'linear_regression_model.joblib'
    model = joblib.load(model_path)
    return model


def evaluate_predictions(y_true, y_pred):
    """Valuta le predizioni utilizzando MSE e R2"""
    mse = mean_squared_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    
    print(f"Mean Squared Error: {mse}")
    print(f"R2 Score: {r2}")
    
    return {
        'mse': mse,
        'r2': r2
    }


def predict_and_save(input_data_path=None, output_path=None, evaluate=False):
    """Carica il modello, fa predizioni sui dati di input e salva i risultati"""
    # Carica il modello (puoi scegliere tra pickle o joblib)
    model = load_model_pickle()  # oppure load_model_joblib()
    
    # Se non è specificato un percorso di input, usa il percorso predefinito
    if input_data_path is None:
        input_data_path = Path('data/processed/user_event_similarity.csv')
    
    # Carica i dati di input
    df = pd.read_csv(input_data_path)
    
    # Seleziona le feature (le stesse usate durante l'addestramento)
    X = df[['region_match', 'user_likes_for_category', 'event_popularity']]
    
    # Effettua la predizione
    predictions = model.predict(X)
    
    # Aggiungi le predizioni al dataframe
    df['predicted_score'] = predictions
    
    # Se è richiesta la valutazione, calcola le metriche
    metrics = None
    if evaluate and 'score' in df.columns:
        metrics = evaluate_predictions(df['score'], predictions)
    
    # Se non è specificato un percorso di output, usa il percorso predefinito
    if output_path is None:
        output_path = Path('data/processed/predictions.csv')
    
    # Crea la directory di output se non esiste
    output_path.parent.mkdir(exist_ok=True, parents=True)
    
    # Salva i risultati
    df.to_csv(output_path, index=False)
    print(f"Predizioni salvate in: {output_path}")
    
    # Salva anche le metriche se disponibili
    if metrics is not None:
        metrics_path = Path('reports') / 'metrics.csv'
        metrics_path.parent.mkdir(exist_ok=True, parents=True)
        pd.DataFrame([metrics]).to_csv(metrics_path, index=False)
        print(f"Metriche salvate in: {metrics_path}")
    
    return df, metrics


def main():
    """Funzione principale"""
    # Carica i dati, fa predizioni e salva i risultati
    df, metrics = predict_and_save(evaluate=True)
    
    # Mostra alcune statistiche sulle predizioni
    print("\nStatistiche sulle predizioni:")
    print(f"Numero di predizioni: {len(df)}")
    print(f"Media delle predizioni: {df['predicted_score'].mean():.4f}")
    print(f"Deviazione standard: {df['predicted_score'].std():.4f}")
    print(f"Min: {df['predicted_score'].min():.4f}")
    print(f"Max: {df['predicted_score'].max():.4f}")
    
    # Se sono disponibili le metriche, mostra un riepilogo
    if metrics is not None:
        print("\nRiepilogo delle metriche:")
        print(f"MSE: {metrics['mse']:.6f}")
        print(f"R²: {metrics['r2']:.6f}")


if __name__ == "__main__":
    main()