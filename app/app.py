import os
import pickle
from flask import Flask, request, render_template

app = Flask(__name__)

# Base path setup to load pickle files reliably regardless of working directory
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "model.pkl")
VECTORIZER_PATH = os.path.join(BASE_DIR, "vectorizer.pkl")

model = None
vectorizer = None

# Attempt to load the pre-trained NLP model and TF-IDF vectorizer
try:
    with open(MODEL_PATH, "rb") as f:
        model = pickle.load(f)
    print("Model loaded successfully.")
    
    with open(VECTORIZER_PATH, "rb") as f:
        vectorizer = pickle.load(f)
    print("Vectorizer loaded successfully.")
except Exception as e:
    print(f"Warning: Could not load model or vectorizer. Did you train the model first? Error: {e}")

@app.route("/", methods=["GET"])
def home():
    """Render the main French HTML page"""
    return render_template("index.html")

@app.route("/ar", methods=["GET"])
def home_ar():
    """Render the Arabic HTML page"""
    return render_template("index_ar.html")

@app.route("/predict", methods=["POST"])
def predict():
    """
    Accept POST request, vectorize text, and use model to return prediction.
    Supports ?lang=ar or ?lang=fr in the URL.
    """
    lang = request.args.get("lang", "fr")
    template = "index_ar.html" if lang == "ar" else "index.html"
    
    # Check if model has been loaded
    if not model or not vectorizer:
        error_msg = "لم يتم تحميل النموذج. يرجى تدريب النموذج أولاً." if lang == "ar" else "Le modèle n'est pas chargé. Veuillez l'entraîner d'abord."
        return render_template(template, error=error_msg)
    
    # Get text from the form
    news_text = request.form.get("news_text", "").strip()
    if not news_text:
        error_msg = "الرجاء إدخال نص الخبر للمراجعة." if lang == "ar" else "Veuillez entrer du texte à analyser."
        return render_template(template, error=error_msg)
    
    try:
        # Transform text using TF-IDF vectorizer
        transformed_text = vectorizer.transform([news_text])
        
        # Predict (0 = Fake, 1 = Real)
        prediction = model.predict(transformed_text)[0] 
        
        # Determine result label
        if prediction == 1:
            result = "حقيقي ✅" if lang == "ar" else "Vrai ✅"
            is_fake = False
        else:
            result = "كاذب ❌" if lang == "ar" else "Faux ❌"
            is_fake = True
            
        return render_template(template, result=result, news_text=news_text, is_fake=is_fake)
        
    except Exception as e:
        return render_template(template, error=str(e), news_text=news_text)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
