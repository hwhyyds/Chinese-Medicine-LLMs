from flask import Flask, request, jsonify
from ChatGLM3.finetune_demo.inference_hf import load_model_and_tokenizer
from ChatGLM3.finetune_demo.RAG.get_similar_text import search_similar_texts, get_dic_data
import json
import os

app = Flask(__name__)
model, tokenizer = load_model_and_tokenizer("./ChatGLM3/finetune_demo/output/checkpoint-30000")

@app.route('/', methods=['GET'])
def predict():
    question = request.args.get('question')
    print(question)
    id = request.args.get('id')

    if str(id) == '2':
        path="./ChatGLM3/finetune_demo/data/train.json"
    
        with open('./ChatGLM3/finetune_demo/data/train.json', 'r') as json_file:
            json_result = json.load(json_file)
        train_values = []
        for i in json_result:
            train_values.append(str(i['conversations']))
    
        distances, indices = search_similar_texts(question, k=1, path=path)
        rag_value = train_values[indices[0][0]]
        train_data_dic = get_dic_data(path=path)
        Q = list(train_data_dic.keys())[indices[0][0]]
        A = list(train_data_dic.values())[indices[0][0]]
        new_prompt = f'这是一份资料Q:{Q}\n:{A}'+ '\n请根据上面这份资料，再结合下文患者的病案诊断下文患者的身体状况，并给出治法和方剂建议。' + question
        response, _ = model.chat(tokenizer, new_prompt)
        send_data = {
            'answer': response,
            'is_rag': True,
            'rag': f'Q:{Q}\nA:{A}'
        }
    else:
        print(question)
        response, _ = model.chat(tokenizer, question) #禁用RAG
    
        send_data = {
            'answer': response,
            'is_rag': False,
        }
    
    return send_data

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
