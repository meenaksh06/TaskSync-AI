from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification

def download_models():
    # Pre-download the zero-shot classifier used in app_enhanced.py
    print("Downloading typeform/distilbert-base-uncased-mnli...")
    pipeline("zero-shot-classification", model="typeform/distilbert-base-uncased-mnli")
    
    # Pre-download the base distilbert model (if used elsewhere)
    print("Downloading distilbert-base-uncased...")
    AutoTokenizer.from_pretrained("distilbert-base-uncased")
    AutoModelForSequenceClassification.from_pretrained("distilbert-base-uncased")
    
    # Download spaCy model
    print("Downloading en_core_web_sm...")
    import spacy
    try:
        spacy.load("en_core_web_sm")
    except OSError:
        from spacy.cli import download
        download("en_core_web_sm")

if __name__ == "__main__":
    download_models()
