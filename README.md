# Chinese-Medicine-LLMs
以ChatGLM-6B作为基础模型，添加RAG功能完成模型，使模型能够基于患者的医案描述给出合适的诊疗

# 使用流程
首先clone chatglm3-6b模型https://github.com/THUDM/ChatGLM3

# data
将train.json、dev.json放在ChatGLM3的finetune-demo/data文件夹下

# RAG
将get_similar_text.py文件放在ChatGLM3的finetune-demo/RAG文件夹下

# Web接口
将Webapp.py文件放在和ChatGLM3文件夹同目录下

# 使用
在终端中启动python Webapp.py
然后可以通过requests.get("http://localhost:5000", data={"question": question,"id":2})访问应用(其中question代表向模型提出的问题，id代表是否使用RAG)
