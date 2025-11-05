import torch
import joblib
import re
import emoji
from torch import nn
from sklearn.feature_extraction.text import TfidfVectorizer

# === Model Class ===
# Same as the model used in the notebook

class SoftmaxClassifier(nn.Module):
    def __init__(self, input_dim, output_dim):
        """
        Initialize the softmax classifier with Batch Normalization.

        Args:
            input_dim: Input dimension (number of TF-IDF features)
            output_dim: Number of classes (3 for negative, neutral, positive)
        """
        super(SoftmaxClassifier, self).__init__()

        self.model = nn.Sequential(

            nn.Linear(input_dim, 64),
            nn.BatchNorm1d(64),  # Batch Normalization to stabilize training
            nn.ReLU(),
            nn.Dropout(0.4),  # Higher dropout for TF-IDF to avoid overfitting
            
            
            nn.Linear(64, output_dim)
        )

    def forward(self, x):
        return self.model(x)




# === Text cleaning function ===
# Same as the function used in the notebook

def clean_text(text, keep_emojis=True):
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r'http\S+|www\S+|https\S+', '', text)
    text = re.sub(r'@\w+|#\w+', '', text)
    text = re.sub(r'(.)\1{2,}', r'\1\1', text)
    text = re.sub(r'[^\w\s]', ' ', text)
    text = re.sub(r'\b(?![ia]\b)\w\b', '', text)

    if keep_emojis:
        text = emoji.demojize(text, language='en')
        text = re.sub(r':', ' ', text)
        text = re.sub(r'[^\w\s]', '', text)
    else:
        text = emoji.replace_emoji(text, replace='')
        text = re.sub(r'[^\w\s]', '', text)

    text = re.sub(r'\s+', ' ', text).strip()
    return text




# === Loading saved objects ===
vectorizer = joblib.load("vectorizer.pkl")
label_encoder = joblib.load("label_encoder.pkl")




# Initialize and load the model
input_dim = len(vectorizer.get_feature_names_out())
output_dim = len(label_encoder.classes_)
model = SoftmaxClassifier(input_dim, output_dim)
model.load_state_dict(torch.load("model.pth"))
model.eval()




# === Start ===
print("== Sentiment Classifier CLI ==")
print("Enter a sentence in English to classify (CTRL+C to exit).")

while True:
    try:
        text = input("\n>> ")
        cleaned = clean_text(text)
        tfidf_vec = vectorizer.transform([cleaned])
        tensor = torch.FloatTensor(tfidf_vec.toarray())

        with torch.no_grad():
            output = model(tensor)
            probs = torch.softmax(output, dim=1)[0]
            pred_idx = probs.argmax().item()
            label = label_encoder.inverse_transform([pred_idx])[0]

        print(f"[Sentiment]: {label} ({probs[pred_idx]*100:.2f}%)")
        print("Probability for each class:")
        for i, class_name in enumerate(label_encoder.classes_):
            print(f" - {class_name}: {probs[i]*100:.2f}%")

    except KeyboardInterrupt:
        print("\nExiting the program.")
        break
