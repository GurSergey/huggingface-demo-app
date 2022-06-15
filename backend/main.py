import json

from flask import Flask, request, jsonify
from transformers import pipeline
from sentence_transformers import SentenceTransformer, util
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
import numpy as np

debug_mode = False

app = Flask(__name__)
app.config['JSON_AS_ASCII'] = False

class CacheModels:
    def __init__(self, size=2):
        self.max_size = size
        self.models = dict()
        self.list_models = list()

    def contains(self, model_name, task):
        return model_name + '_' + task in self.models.keys()

    def put(self, model_name, task, model):
        if len(self.models) == self.max_size:
            del self.models[self.list_models[0]]
            self.list_models = self.list_models[1:]
        self.models[model_name+'_'+task] = model
        self.list_models.append(model_name+'_'+task)

    def get(self, model_name, task):
        if self.contains(model_name, task):
            return self.models[model_name+'_'+task]
        else:
            return None


class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            # 👇️ alternatively use str()
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)


cache = CacheModels()


@app.route('/classification', methods=['POST'])
def get_answer():
    request_data = request.json
    model, text = \
        request_data['model'],\
        request_data['text']
    if cache.contains("text-classification", model):
        ppl = cache.get("text-classification", model)
    else:
        ppl = pipeline("text-classification", model)
        cache.put("text-classification", model, ppl)
    return json.dumps(ppl(text))


@app.route('/ner', methods=['POST'])
def get_ner():
    request_data = request.json
    model, text = \
        request_data['model'],\
        request_data['text']
    if cache.contains("ner", model):
        ner_pipeline = cache.get("ner", model)
    else:
        ner_pipeline = pipeline("ner", model, aggregation_strategy="first")
        cache.put("ner", model, ner_pipeline)
    return json.dumps(ner_pipeline(text), cls=NpEncoder)


@app.route('/sentence_sim', methods=['POST'])
def get_sentence_sim():
    request_data = request.json
    model, sentences = \
        request_data['model'], \
        request_data['sentences']
    if cache.contains("sentence_sim", model):
        model = cache.get("sentence_sim", model)
    else:
        model = SentenceTransformer(model)
        cache.put("sentence_sim", model, model)

    cos_sim = dict()
    for sentence in sentences[1:]:
        cos_sim[sentence] = util.pytorch_cos_sim(model.encode(sentences[0], convert_to_tensor=True),
                                                 model.encode(sentence, convert_to_tensor=True)).item()
    return cos_sim


@app.route('/qa', methods=['POST'])
def get_qa():
    request_data = request.json
    model, question, context = \
        request_data['model'],\
        request_data['question'], request_data['context']
    qa = pipeline("question-answering", model)
    return json.dumps(qa(question, context), cls=NpEncoder)


@app.route('/generation', methods=['POST'])
def generate():
    request_data = request.json
    model, sentences = \
        request_data['model'], \
        request_data['text']
    generator = pipeline('text-generation', model=model)
    return json.dumps(generator(sentences))


@app.route('/seq2seq', methods=['POST'])
def get_seq2seq():
    request_data = request.json
    model, sentences = \
        request_data['model'], \
        request_data['text']
    seq2seq = pipeline('text2text-generation', model=model)
    return json.dumps(seq2seq(sentences))


@app.route('/translation', methods=['POST'])
def get_translation():
    request_data = request.json
    model, from_lang, to_lang, text = \
        request_data['model'], request_data['from'], request_data['to'],\
        request_data['text']
    ppl = pipeline("translation_" + from_lang + "_to_" + to_lang, model)
    return json.dumps(ppl(text))


@app.route('/summarization', methods=['POST'])
def get_summarization():
    request_data = request.json
    model, text = \
        request_data['model'],\
        request_data['text']
    ppl = pipeline("summarization", model)
    return json.dumps(ppl(text))


@app.route('/closed-book', methods=['POST'])
def get_qa_closed_book():
    request_data = request.json
    model, text = \
        request_data['model'], \
        request_data['text']

    qa_model = AutoModelForSeq2SeqLM.from_pretrained(model)
    tok = AutoTokenizer.from_pretrained(model)
    input_ids = tok(text, return_tensors="pt").input_ids
    gen_output = qa_model.generate(input_ids)[0]
    return json.dumps(tok.decode(gen_output, skip_special_tokens=True))


app.debug = False
app.run(host='0.0.0.0', port=5005)
