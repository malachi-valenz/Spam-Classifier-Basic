import pickle
import string
from flask import Flask, request, jsonify
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

#Load the saved model and vectorizer
with open('model.pkl', 'rb') as f:
    model = pickle.load(f)

with open('vectorizer.pkl', 'rb') as f:
    vectorizer = pickle.load(f)

def clean_text(text):
    return ''.join([char for char in text if char not in string.punctuation]).lower()

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    text = clean_text(data['text'])
    vectorized = vectorizer.transform([text])
    prediction = model.predict(vectorized)[0]
    return jsonify({'result': prediction})

if __name__ == '__main__':
    app.run(debug=True)