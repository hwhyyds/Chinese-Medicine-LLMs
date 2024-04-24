import torch
from transformers import BertTokenizer, BertModel
import faiss
import os
import json
os.environ['TRANSFORMERS_CACHE'] = '/home/hk_cache'
tokenizer = BertTokenizer.from_pretrained('bert-base-chinese')
model = BertModel.from_pretrained('bert-base-chinese')

def bert_vectorize(texts):
    # 编码文本
    encoded_input = tokenizer(texts, padding=True, truncation=True, return_tensors='pt')
    # 获取BERT模型的输出
    with torch.no_grad():
        model_output = model(**encoded_input)
    # 提取[CLS]标记的输出作为句子的向量表示
    sentence_embeddings = model_output.last_hidden_state[:, 0, :].numpy()
    return sentence_embeddings

def get_dic_data(path):
    with open(path, 'r') as json_file:
        json_result = json.load(json_file)
    dic = {}
    for i in json_result:
        dic[i['conversations'][1]['content']] = i['conversations'][2]['content']
    return dic

def get_fassi_index(path):
    dic = get_dic_data(path)
    dic_input_texts = list(dic.keys())
    vectorized_texts = bert_vectorize(dic_input_texts) # 构建索引库
    dimension = vectorized_texts.shape[1]  # 获取向量的维度
    index = faiss.IndexFlatL2(dimension)  # 使用L2距离创建Faiss索引
    # 将数据转换为Faiss期望的格式
    vectorized_texts = vectorized_texts.astype('float32')
    # 将向量添加到Faiss索引
    index.add(vectorized_texts)
    return index

def search_similar_texts(query_text, k=2, path='data/train.json'):
    index = get_fassi_index(path)
    query_vector = bert_vectorize([query_text])
    query_vector = query_vector.astype('float32')
    distances, indices = index.search(query_vector, k)
    return distances, indices

if __name__ == '__main__':
    distances, indices = search_similar_texts('脊某，男，5岁。初诊：1979年4月2日。主诉及病史：周岁即患哮嗤，每逢冬春发作。现在复发两天。诊查：昨起稍有喘咳，今日喉中哮鸣如曳锯，多汗神萎，唇绀，舌苔腻，脉滑。', k=1)
    print(distances)
    print(indices)