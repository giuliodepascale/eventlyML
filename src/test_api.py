import requests
import json
import pandas as pd
import numpy as np

# URL di base dell'API (modifica se necessario)
BASE_URL = "http://localhost:5000"

def test_health_check():
    """Verifica lo stato dell'API"""
    response = requests.get(f"{BASE_URL}/health")
    print("\nTest Health Check:")
    print(f"Status Code: {response.status_code}")
    print(f"Response: {response.json()}")
    return response.status_code == 200

def test_single_prediction():
    """Test di una singola predizione"""
    # Dati di esempio per la predizione
    data = {
        "region_match": 1,
        "user_likes_for_category": 3,
        "event_popularity": 15
    }
    
    # Invia la richiesta POST
    response = requests.post(
        f"{BASE_URL}/predict",
        json=data,
        headers={"Content-Type": "application/json"}
    )
    
    print("\nTest Predizione Singola:")
    print(f"Status Code: {response.status_code}")
    print(f"Response: {json.dumps(response.json(), indent=2)}")
    return response.status_code == 200

def test_batch_prediction():
    """Test di predizioni batch"""
    # Dati di esempio per la predizione batch
    data = [
        {
            "region_match": 1,
            "user_likes_for_category": 3,
            "event_popularity": 15
        },
        {
            "region_match": 0,
            "user_likes_for_category": 1,
            "event_popularity": 5
        },
        {
            "region_match": 1,
            "user_likes_for_category": 5,
            "event_popularity": 20
        }
    ]
    
    # Invia la richiesta POST
    response = requests.post(
        f"{BASE_URL}/batch-predict",
        json=data,
        headers={"Content-Type": "application/json"}
    )
    
    print("\nTest Predizione Batch:")
    print(f"Status Code: {response.status_code}")
    print(f"Response: {json.dumps(response.json(), indent=2)}")
    return response.status_code == 200

def main():
    print("=== Test dell'API EventlyML ===")
    
    # Esegui i test
    health_ok = test_health_check()
    if not health_ok:
        print("\nERRORE: L'API non è disponibile. Assicurati che il server sia in esecuzione.")
        return
    
    single_ok = test_single_prediction()
    batch_ok = test_batch_prediction()
    
    # Riepilogo
    print("\n=== Riepilogo dei Test ===")
    print(f"Health Check: {'OK' if health_ok else 'FALLITO'}")
    print(f"Predizione Singola: {'OK' if single_ok else 'FALLITO'}")
    print(f"Predizione Batch: {'OK' if batch_ok else 'FALLITO'}")
    
    if health_ok and single_ok and batch_ok:
        print("\nTutti i test sono stati completati con successo!")
    else:
        print("\nAlcuni test sono falliti. Controlla i messaggi di errore sopra.")

if __name__ == "__main__":
    main()