import mlflow
import os
import pandas as pd
import ast
from dotenv import load_dotenv
load_dotenv()

mlflow.set_tracking_uri("https://dagshub.com/giuliodepascale/eventlyML.mlflow")
os.environ["MLFLOW_TRACKING_USERNAME"] = os.getenv('USERNAME')
os.environ["MLFLOW_TRACKING_PASSWORD"] = os.getenv('PASSWORD')

def preprocess_data():
    users_df = pd.read_csv('data/raw/final_synthetic_users_with_region.csv')
    events_df = pd.read_csv('data/raw/final_synthetic_events.csv')
    events_df = events_df.set_index('id')
    try:
        users_df['favoriteIds'] = users_df['favoriteIds'].apply(ast.literal_eval)
    except Exception as e:
        raise ValueError(f"Errore nella conversione delle stringhe in liste: {e}")

    # Pre-calcola la categoria di ogni evento
    event_category = events_df['category'].to_dict()

    # Prepara lista per le righe del nuovo dataset
    rows = []

    for _, user in users_df.iterrows():
        user_id = user['id']
        user_region = user['regione']
        user_favorites = user['favoriteIds']
        # Conta i like dell'utente per categoria
        liked_categories = {}
        for fav_id in user_favorites:
            if fav_id in event_category:
                cat = event_category[fav_id]
                liked_categories[cat] = liked_categories.get(cat, 0) + 1

        for event_id, event in events_df.iterrows():
            event_region = event['regione']
            event_cat = event['category']
            event_fav_count = event['favoriteCount']

            # Feature 1: match regione (1 se coincide, 0 altrimenti)
            region_match = int(user_region == event_region)
            # Feature 2: numero di like dell'utente per la categoria dell'evento
            user_likes_for_cat = liked_categories.get(event_cat, 0)
            # Feature 3: popolarità evento
            popularity = event_fav_count

            # Score: puoi personalizzare la formula, qui una semplice somma pesata
            score = 0.5 * region_match + 0.3 * (user_likes_for_cat > 0) + 0.2 * (popularity / (1 + popularity))

            rows.append({
                'user_id': user_id,
                'event_id': event_id,
                'region_match': region_match,
                'user_likes_for_category': user_likes_for_cat,
                'event_popularity': popularity,
                'score': score
            })

    df = pd.DataFrame(rows)
    return df

if __name__ == "__main__":
    mlflow.set_experiment("user-features-experiment")
    with mlflow.start_run():
        df = preprocess_data()
        output_path = 'data/processed/user_event_similarity.csv'
        df.to_csv(output_path, index=False, encoding='utf-8')
        mlflow.log_artifact(output_path)
        print("Dataset utente-evento creato e tracciato su DagsHub tramite MLflow.")