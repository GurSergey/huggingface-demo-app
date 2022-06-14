from flask import Flask, request, jsonify
from transformers import pipeline
from sentence_transformers import SentenceTransformer, util
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

debug_mode = False

app = Flask(__name__)


@app.route('/classification', methods=['POST'])
def get_answer():
    request_data = request.json
    model, text = \
        request_data['model'],\
        request_data['text']
    ppl = pipeline("text-classification", model)
    return str(ppl(text))


@app.route('/ner', methods=['POST'])
def get_ner():
    request_data = request.json
    model, text = \
        request_data['model'],\
        request_data['text']
    ner_pipeline = pipeline("ner", model, aggregation_strategy="first")
    return str(ner_pipeline(text))


@app.route('/sentence_sim', methods=['POST'])
def get_sentence_sim():
    request_data = request.json
    model, sentences = \
        request_data['model'], \
        request_data['text']

    model = SentenceTransformer(model)
    embeddings = []
    for sentence in sentences:
        embeddings.append(model.encode(sentence, convert_to_tensor=True))
    cos_sim = dict()
    for emb in embeddings[1:]:
        cos_sim[emb] = util.pytorch_cos_sim(embeddings[0], emb)
    return cos_sim


@app.route('/qa', methods=['POST'])
def get_qa():
    request_data = request.json
    model, question, context = \
        request_data['model'],\
        request_data['question'], request_data['context']
    qa = pipeline("question-answering", model)
    return str(qa(question, context))


@app.route('/generation', methods=['POST'])
def generate():
    request_data = request.json
    model, sentences = \
        request_data['model'], \
        request_data['text']
    generator = pipeline('text-generation', model=model)
    return generator(sentences)


@app.route('/seq2seq', methods=['POST'])
def get_seq2seq():
    request_data = request.json
    model, sentences = \
        request_data['model'], \
        request_data['text']
    seq2seq = pipeline('text2text-generation', model=model)
    return seq2seq(sentences)


@app.route('/translation', methods=['POST'])
def get_translation():
    request_data = request.json
    model, from_lang, to_lang, text = \
        request_data['model'], request_data['from'], request_data['to'],\
        request_data['text']
    ppl = pipeline("translation_" + from_lang + "_to_" + to_lang, model)
    return str(ppl(text))


@app.route('/summarization', methods=['POST'])
def get_summarization():
    request_data = request.json
    model, task, text = \
        request_data['model'], request_data['task'],\
        request_data['text']
    ppl = pipeline(task, model)
    return str(ppl(text))


@app.route('/closed-book', methods=['POST'])
def get_qa_closed_book():
    t5_qa_model = AutoModelForSeq2SeqLM.from_pretrained("google/t5-small-ssm-nq")
    t5_tok = AutoTokenizer.from_pretrained("google/t5-small-ssm-nq")
    input_ids = t5_tok("When was Franklin D. Roosevelt born?", return_tensors="pt").input_ids
    gen_output = t5_qa_model.generate(input_ids)[0]
    print(t5_tok.decode(gen_output, skip_special_tokens=True))


app.debug = False
app.run(host='0.0.0.0', port=5005)
