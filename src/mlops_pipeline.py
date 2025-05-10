import subprocess

def run_command(command):
    """Esegue un comando di shell e stampa l'output."""
    result = subprocess.run(command, shell=True, capture_output=True, text=True)
    print(result.stdout)
    if result.stderr:
        print(result.stderr)

def main():
    # Definisci i percorsi dei file come variabili
    make_dataset_script = "c:/Users/depas/Documents/eventlyML/src/data/make_dataset.py"
    train_model_script = "c:/Users/depas/Documents/eventlyML/src/models/train_model.py"
    predict_model_script = "c:/Users/depas/Documents/eventlyML/src/models/predict_model.py"

    # Step 1: Preprocess the data
    print("Preprocessing data...")
    run_command(f"python {make_dataset_script}")

    # Step 2: Train the model
    print("Training model...")
    run_command(f"python {train_model_script}")

    # Step 3: Validate the model
    print("Validating model...")
    run_command(f"python {predict_model_script}")

if __name__ == "__main__":
    main()