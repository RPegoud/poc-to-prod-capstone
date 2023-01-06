from flask import Flask, render_template, request
from run import TextPredictionModel

app = Flask(__name__)

model_dir = "train/data/artefacts/2023-01-03-10-43-31"

@app.route('/', methods=["GET"])
def index():
    return render_template('index.html')

@app.route('/', methods=["POST"])
def predict():
    model = TextPredictionModel.from_artefacts(model_dir)
    model_input = request.form['model_input']
    preds = model.predict([model_input], top_k=3)

    return preds

if __name__ == '__main__':
    app.run(debug=True)