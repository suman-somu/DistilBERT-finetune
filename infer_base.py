import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
base_model = AutoModelForSequenceClassification.from_pretrained("distilbert-base-uncased")

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
base_model.to(device)

def run_base_inference(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding="max_length").to(device)
    
    with torch.no_grad():
        outputs = base_model(**inputs)
    
    predictions = outputs.logits.argmax(dim=-1)
    return predictions.item()

texts = [
    "This movie was fantastic! I loved every minute of it.",
    "I didn't like this movie. It was really boring and poorly made.",
    "The acting was great, but the story was weak.",
    "What an amazing film! Will definitely watch it again.",
    "I wish I could get my time back. This movie was terrible.",
    "The plot was intriguing, but the pacing felt off.",
    "This is a must-watch movie for all cinema lovers.",
    "I couldn't finish this movie. It was too long and uninteresting.",
    "The cinematography was beautiful, but the script was lacking.",
    "I loved the movie, it was so heartwarming and emotional."
]

for text in texts:
    prediction = run_base_inference(text)
    sentiment = "Positive" if prediction == 1 else "Negative"
    print(f"Input: {text}\nPredicted Sentiment: {sentiment}\n")