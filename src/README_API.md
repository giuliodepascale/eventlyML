# API per EventlyML

Questo documento descrive come utilizzare l'API REST per fare inferenze con il modello di machine learning di EventlyML.

## Requisiti

Per eseguire l'API, assicurati di avere installato tutte le dipendenze necessarie:

```bash
pip install -r requirements.txt
```

Assicurati che il file `requirements.txt` contenga le seguenti dipendenze:
- flask
- pandas
- numpy
- scikit-learn
- joblib

## Avvio del server API

Per avviare il server API, esegui il seguente comando dalla directory principale del progetto:

```bash
python -m src.api
```

Il server sarà disponibile all'indirizzo `http://localhost:5000`.

## Endpoint disponibili

### Controllo dello stato dell'API

```
GET /health
```

Risposta di esempio:
```json
{
  "status": "ok",
  "message": "API funzionante e modello caricato"
}
```

### Predizione singola

```
POST /predict
```

Corpo della richiesta (JSON):
```json
{
  "region_match": 1,
  "user_likes_for_category": 3,
  "event_popularity": 15
}
```

Risposta di esempio:
```json
{
  "prediction": 0.75,
  "input_data": {
    "region_match": 1,
    "user_likes_for_category": 3,
    "event_popularity": 15
  }
}
```

### Predizione batch

```
POST /batch-predict
```

Corpo della richiesta (JSON):
```json
[
  {
    "region_match": 1,
    "user_likes_for_category": 3,
    "event_popularity": 15
  },
  {
    "region_match": 0,
    "user_likes_for_category": 1,
    "event_popularity": 5
  }
]
```

Risposta di esempio:
```json
[
  {
    "prediction": 0.75,
    "input_data": {
      "region_match": 1,
      "user_likes_for_category": 3,
      "event_popularity": 15
    }
  },
  {
    "prediction": 0.25,
    "input_data": {
      "region_match": 0,
      "user_likes_for_category": 1,
      "event_popularity": 5
    }
  }
]
```

## Esempi di utilizzo con curl

### Controllo dello stato

```bash
curl -X GET http://localhost:5000/health
```

### Predizione singola

```bash
curl -X POST http://localhost:5000/predict \
  -H "Content-Type: application/json" \
  -d '{"region_match": 1, "user_likes_for_category": 3, "event_popularity": 15}'
```

### Predizione batch

```bash
curl -X POST http://localhost:5000/batch-predict \
  -H "Content-Type: application/json" \
  -d '[{"region_match": 1, "user_likes_for_category": 3, "event_popularity": 15}, {"region_match": 0, "user_likes_for_category": 1, "event_popularity": 5}]'
```

## Deployment in produzione

Per il deployment in produzione, si consiglia di utilizzare un server WSGI come Gunicorn:

```bash
pip install gunicorn
gunicorn -w 4 -b 0.0.0.0:5000 src.api:app
```

Per un deployment più robusto, considera l'utilizzo di Docker o servizi cloud come AWS, Google Cloud o Azure.