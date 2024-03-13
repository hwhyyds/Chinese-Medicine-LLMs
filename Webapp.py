from flask import Flask, request, jsonify
from ChatGLM3.finetune_demo.inference_hf import load_model_and_tokenizer
from ChatGLM3.finetune_demo.RAG.get_similar_text import search_similar_texts, get_dic_data
import json
import os

app = Flask(__name__)

@app.route('/', methods=['GET'])
def predict():
    question = request.form.get('question')
    id = request.form.get('id')
    model, tokenizer = load_model_and_tokenizer("./ChatGLM3/finetune_demo/output/checkpoint-30000")
    if id==2:
        path="./ChatGLM3/finetune_demo/data/train.json"
    
        with open('./ChatGLM3/finetune_demo/data/train.json', 'r') as json_file:
            json_result = json.load(json_file)
        train_values = []
        for i in json_result:
            train_values.append(str(i['conversations']))
    
        distances, indices = search_similar_texts(question, k=1, path=path)
        rag_value = train_values[indices[0][0]]
        print(os.getcwd())
        train_data_dic = get_dic_data(path=path)
        Q = list(train_data_dic.keys())[indices[0][0]]
        A = list(train_data_dic.values())[indices[0][0]]
        new_prompt = question + f'\n你可以参考这个病案诊断进行回答 Q:{Q}\n:{A}'
        response, _ = model.chat(tokenizer, new_prompt)
    else:
        response, _ = model.chat(tokenizer, question) #禁用RAG
    
    send_data = {
        'answer': response,
        'id': id,
        # 'is_rag': True,
        # 'rag': f'{rag_value}'
    }
    
    return jsonify(send_data)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
