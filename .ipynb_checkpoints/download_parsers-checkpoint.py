import nltk
import spacy
import benepar
import os

def download_spacy_model(model_name="en_core_web_sm"):
    """Checks if a SpaCy model is installed and downloads it if not."""
    try:
        spacy.load(model_name)
        print(f"SpaCy model '{model_name}' already installed.")
    except OSError:
        print(f"SpaCy model '{model_name}' not found. Downloading...")
        try:
            spacy.cli.download(model_name)
            print(f"Successfully downloaded SpaCy model '{model_name}'.")
        except Exception as e:
            print(f"Error downloading SpaCy model '{model_name}': {e}")
            print("Please try: python -m spacy download {model_name}")

def download_nltk_resource(resource_name, resource_path_check):
    """Checks if an NLTK resource is available and downloads it if not."""
    try:
        nltk.data.find(resource_path_check)
        print(f"NLTK resource '{resource_name}' already available.")
    except LookupError:
        print(f"NLTK resource '{resource_name}' not found. Downloading...")
        try:
            nltk.download(resource_name, quiet=False) # Set quiet=False to see download progress
            print(f"Successfully downloaded NLTK resource '{resource_name}'.")
            # Verify after download
            nltk.data.find(resource_path_check)
            print(f"NLTK resource '{resource_name}' verified after download.")
        except Exception as e:
            print(f"Error downloading NLTK resource '{resource_name}': {e}")
            print(f"Please try manually in Python: import nltk; nltk.download('{resource_name}')")


def download_benepar_model(model_name="benepar_en3"):
    """Checks if a Benepar model is available and downloads it if not."""
    # Benepar usually downloads its models into the NLTK data path.
    # A simple way to check is to try initializing it; it often handles its own download.
    try:
        _ = benepar.Parser(model_name) # This might trigger download if not present
        print(f"Benepar model '{model_name}' is available or was auto-downloaded.")
    except Exception as e: # Catch broader exceptions as benepar's internals might vary
        print(f"Benepar model '{model_name}' not immediately available (Error: {e}). Attempting explicit download...")
        try:
            benepar.download(model_name)
            print(f"Successfully downloaded Benepar model '{model_name}'.")
            _ = benepar.Parser(model_name) # Verify load after download
            print(f"Benepar model '{model_name}' successfully loaded after download.")
        except Exception as download_e:
            print(f"Error downloading or loading Benepar model '{model_name}': {download_e}")
            print(f"Please try manually in Python: import benepar; benepar.download('{model_name}')")

if __name__ == "__main__":
    print("--- Starting Parser Dependency Download Script ---")

    # 1. SpaCy model
    print("\n--- SpaCy Model Check ---")
    download_spacy_model("en_core_web_sm")

    # 2. NLTK 'punkt' tokenizer (needed by Benepar)
    print("\n--- NLTK 'punkt' Tokenizer Check ---")
    download_nltk_resource(resource_name="punkt", resource_path_check="tokenizers/punkt")

    # 3. Benepar model
    print("\n--- Benepar Model Check ---")
    download_benepar_model("benepar_en3") # You can change to "benepar_en3_large" if preferred

    print("\n--- Dependency check complete. ---")
    print("If any downloads failed, please try the manual commands suggested above.")