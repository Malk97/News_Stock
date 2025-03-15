from Data_preprocess import preprocess_text
from transformers import AutoModelForSequenceClassification, AutoTokenizer

def load_model_and_tokenizer(model_path: str):
    """
    Load the model and tokenizer from the given path.
    """
    model = AutoModelForSequenceClassification.from_pretrained(model_path)
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    return model, tokenizer

def predict_class(text: str, model, tokenizer):
    """
    Preprocess the text, tokenize it, and predict the class using the model.
    """
    # Preprocess text (this will be handled by the preprocess_text function from text_processing.py)
    text = preprocess_text(text)

    # Tokenize the preprocessed text
    inputs = tokenizer(text, return_tensors='pt')

    # Get model outputs
    outputs = model(**inputs)

    # Make prediction based on the model output
    predictions = outputs.logits.argmax(dim=-1)
    return "Positive" if predictions.item() == 1 else "Negative"
