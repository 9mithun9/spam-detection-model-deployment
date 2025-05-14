from flask import Flask, render_template, request, jsonify
import pickle 

tokenizer = pickle.load(open("models/cv.pkl", "rb"))
model = pickle.load(open('models/clf.pkl','rb'))

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=["POST"])
def predcit():
    email_text = request.form.get('email-content')
    tokenized_email = tokenizer.transform([email_text])
    prediction = model.predict(tokenized_email)
    prediction = 1 if prediction == 1 else -1
    return render_template('index.html', text=email_text, prediction = prediction )



@app.route('/api/predict', methods=['POST'])
def predict_api():
    data = request.get_json(force=True)  # Get data posted as a json
    email_text = data['email-content']
    tokenized_email = tokenizer.transform([email_text]) # X 
    prediction = model.predict(tokenized_email)
    # If the email is spam prediction should be 1
    prediction = 1 if prediction == 1 else -1
    return jsonify({prediction: prediction})



if __name__ == '__main__':
    app.run(host='0.0.0.0', debug=True)