from flask import Flask, request, render_template
import pickle
import numpy as np

app = Flask(__name__)

# load the model

with open("model.pkl", "rb") as f:
    model, scaler = pickle.load(f)

@app.route('/', methods=['GET', 'POST'])
def home():
    if request.method == 'POST':
        data = [float(x) for x in request.form.values()]
        scaled_data = scaler.transform([data])
        prediction = model.predict(scaled_data)[0]
        result = "Customer is likely to churn" if prediction == 1 else "Customer will stay"
        return render_template('index.html', prediction_text=result)
    return render_template('index.html')

@app.route('/analyze', methods=['POST'])
def analyze():
    # your code here
    pass   
    

if __name__ =='__main__':
    app.run(debug=True)
