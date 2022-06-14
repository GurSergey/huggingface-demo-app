from flask import Flask, request, jsonify
from transformers import pipeline

debug_mode = False

app = Flask(__name__)


@app.route('/translation', methods=['POST'])
def get_answer():
    request_data = request.json
    model, task, text = \
        request_data['model'], request_data['task'],\
        request_data['text']
    ppl = pipeline(task, model)
    return str(ppl(text))


@app.route('/translation', methods=['POST'])
def get_answer():
    request_data = request.json
    model, task, text = \
        request_data['model'], request_data['task'],\
        request_data['text']
    ppl = pipeline(task, model)
    return str(ppl(text))

app.debug = False
app.run(host='0.0.0.0', port=5005)
