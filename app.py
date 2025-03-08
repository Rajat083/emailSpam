from flask import Flask, request, render_template
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer as ps
import string
import pickle 

nltk.download('stopwords')
nltk.download('punkt')

def text_preprocess(text):
    text = text.lower()
    text = nltk.word_tokenize(text)

    y=[x for x in text if x.isalnum()]
    text = y[:]
    y.clear()
    y = [word for word in text if word not in stopwords.words('english') and word not in string.punctuation]

    text = y[:]
    y.clear()

    y = [ps().stem(word) for word in text]

    return " ".join(y)

app = Flask(__name__)


model_bnb = pickle.load(open('spam_model.pkl', 'rb'))
model_rfc = pickle.load(open('spam.pkl', 'rb'))
model_ann = pickle.load(open('ann.pkl', 'rb'))

tfidf = pickle.load(open('spam_vectorizer.pkl', 'rb'))
predictions_meaning = {1 : 'Spam', 0 : 'Not Spam'}

def colorAdder(str, result):
    if result == 'Spam':
        return f'<span style="color: red;">{str}</span>'
    else:
        return f'<span style="color: green;">{str}</span>'

@app.route('/')
def home():
    return render_template("index.html")

@app.route('/predict', methods=['POST'])
def test():
    text = request.form['input']
    preprocessed_text = text_preprocess(text)
    vectorized_text = tfidf.transform([preprocessed_text])

    predictions = [model_bnb.predict(vectorized_text), model_rfc.predict(vectorized_text)]
    model_pred = [predictions_meaning[predictions[0][0]],predictions_meaning[predictions[1][0]], predictions_meaning[int(model_ann.predict(vectorized_text)[0][0] > 0.5)]]
    bnb_text = f'Naive Bayes Classifier Prediction: {model_pred[0]}\n'
    rfc_text = f'Random Forest Classifier Prediction: {model_pred[1]}\n'
    ann_text = f'Artificial Neural Network Prediction: {model_pred[2]}\n'
    
    return render_template('index.html', 
                        bnb=bnb_text or "No prediction", 
                        rfc=rfc_text or "No prediction", 
                        ann=ann_text or "No prediction",
                        bnb_result=predictions[0][0] if predictions else 0, 
                        rfc_result=predictions[1][0] if predictions else 0,
                        ann_result=model_ann.predict(vectorized_text)[0] if predictions else 0,
                        )

if __name__ == '__main__':
    app.run(debug=True)