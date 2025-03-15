from Data_preprocess import preprocess_text
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import boto3
BUCKET_NAME = "data-storage-bucket-123" 
MODEL_PATH_IN_S3 = "Model/" 

def load_model_from_s3(bucket_name, model_path_in_s3):
    model_file = io.BytesIO()
    tokenizer_file = io.BytesIO()
    
    s3_client.download_fileobj(bucket_name, model_path_in_s3 + 'model.safetensors', model_file)
    
    s3_client.download_fileobj(bucket_name, model_path_in_s3 + 'tokenizer.json', tokenizer_file)
    
    model_file.seek(0)
    model = AutoModelForSequenceClassification.from_pretrained(model_file)
    
    tokenizer_file.seek(0)
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_file)
    
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
