import pandas as pd
import ast

def preprocess_data():
    users_df = pd.read_csv('data/raw/final_synthetic_users.csv')
    events_df = pd.read_csv('data/raw/final_synthetic_events.csv').set_index('id')  # Imposta l'indice sugli ID evento
    
    # Converti le stringhe degli ID preferiti in liste
    try:
        users_df['favoriteIds'] = users_df['favoriteIds'].apply(ast.literal_eval)
    except Exception as e:
        raise ValueError(f"Errore nella conversione delle stringhe in liste: {e}")


    # Creazione di un DataFrame per le caratteristiche
    user_features = []
    # Correggi il ciclo per iterare su ogni utente e i suoi preferiti
    for user_id, favorites in zip(users_df['id'], users_df['favoriteIds']):
        regions = []
        categories = []
        for event_id in favorites:
            if event_id in events_df.index:
                event = events_df.loc[event_id]
                regions.append(event['regione'])
                categories.append(event['category'])
            else:
                print(f"Attenzione: l'evento con ID {event_id} non esiste.")
        user_features.append({
            'user_id': user_id,
            'regions': regions,
            'categories': categories
        })
    
    user_features_df = pd.DataFrame(user_features)
    return user_features_df

if __name__ == "__main__":
    user_features_df = preprocess_data()
    user_features_df.to_csv('data/processed/user_features.csv', index=False, encoding='utf-8')
    print("Dati salvati correttamente in data/processed/user_features.csv")
    print(user_features_df)