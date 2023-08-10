import os
from flask import Flask, render_template, request
from model import load_model, classify_image

app = Flask(__name__)
model = load_model()

@app.route('/', methods=['GET', 'POST'])
def index():
    input_data = []
    if request.method == 'POST':
        input_data = [
            request.form.get('input_text_petal_length'), 
            request.form.get('input_text_petal_width'), 
            request.form.get('input_text_sepal_length'), 
            request.form.get('input_text_sepal_width')]

        if input_data:
            prediction = classify_image(model, [int(numeric_Input) for numeric_Input in input_data])
            return render_template('index.html', prediction=prediction,input_data = input_data)

    return render_template('index.html',input_data = input_data)

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=4000)