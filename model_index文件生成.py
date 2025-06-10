import json
import os

# 定义模型组件的目录路径
model_directory = "/home/qiqiutang/qqt/AUDIT_my/checkpoint/model_checkpoints/pipeline/"

# 获取模型组件的列表
components = os.listdir(model_directory)

# 创建一个字典来存储模型组件的信息
model_index = {
    "components": [
        {"name": component, "path": os.path.join(model_directory, component)} for component in components
    ]
}

# 将字典写入JSON文件
with open(os.path.join(model_directory, "model_index.json"), "w", encoding="utf-8") as f:
    json.dump(model_index, f, ensure_ascii=False, indent=4)

print(f"model_index.json 已创建在 {model_directory}")