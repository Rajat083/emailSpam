from flask import Flask, request, render_template
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer as ps
import string
import pickle 

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
# model_svc = pickle.load(open('spam_svc.pkl', 'rb'))
# text_preprocess = pickle.load(open('spam_preprocess.pkl', 'rb'))
tfidf = pickle.load(open('spam_vectorizer.pkl', 'rb'))
predictions_meanig = {1 : 'Spam', 0 : 'Not Spam'}

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
    model_pred = [predictions_meanig[predictions[0][0]],predictions_meanig[predictions[1][0]]]
    bnb_text = f'Naive Bayes Classifier Prediction: {model_pred[0]}\n'
    rfc_text = f'Random Forest Classifier Prediction: {model_pred[1]}'
    
    return render_template('index.html', bnb=bnb_text, rfc=rfc_text, bnb_result=predictions[0][0], rfc_result=predictions[1][0])


# st.title('E-mail Spam Classifier')
# input = st.text_input('Enter the e-mail text')

# if st.button('Predict'):
#     preprocessed_text = text_preprocess(input)

#     vectorized_text = tfidf.transform([preprocessed_text])

#     predictions_meanig = {1 : 'Spam', 0 : 'Not Spam'}
#     predictions = [model_bnb.predict(vectorized_text), model_svc.predict(vectorized_text)]

#     st.write(f'Naive Bayes Classifier Prediction: {predictions_meanig[predictions[0][0]]}')
#     st.write(f'Support Vector Classifier Prediction: {predictions_meanig[predictions[1][0]]}')

if __name__ == '__main__':
    app.run(debug=True)
