# import torch
# import torch.nn as nn
# import torch.optim as optim
# from torch.utils.data import DataLoader, TensorDataset

# # 1. 伪造数据
# num_samples = 100  # 样本数
# num_features = 10  # 特征数
# num_labels = 5  # 标签数

# # 随机生成数据
# X = torch.randn(num_samples, num_features)  # 100个样本，每个样本10个特征
# y = torch.randint(0, 2, (num_samples, num_labels)).float()  # 100个样本，每个样本5个标签，0或1

# # 创建数据集和数据加载器
# dataset = TensorDataset(X, y)
# dataloader = DataLoader(dataset, batch_size=16, shuffle=True)

# # 2. 定义神经网络
# class MultiLabelNN(nn.Module):
#     def __init__(self, input_size, hidden_size, output_size):
#         super(MultiLabelNN, self).__init__()
#         self.fc1 = nn.Linear(input_size, hidden_size)
#         self.fc2 = nn.Linear(hidden_size, output_size)
#         self.sigmoid = nn.Sigmoid()  # 输出概率

#     def forward(self, x):
#         x = torch.relu(self.fc1(x))  # 第一层，ReLU 激活
#         x = self.fc2(x)  # 第二层
#         x = self.sigmoid(x)  # 每个标签的概率
#         return x

# # 创建网络实例
# input_size = num_features
# hidden_size = 64  # 隐藏层大小
# output_size = num_labels  # 标签数
# model = MultiLabelNN(input_size, hidden_size, output_size)

# # 3. 设置损失函数和优化器
# criterion = nn.BCELoss()  # 二进制交叉熵损失
# optimizer = optim.Adam(model.parameters(), lr=0.001)

# # 4. 训练模型
# num_epochs = 10

# for epoch in range(num_epochs):
#     model.train()
#     running_loss = 0.0

#     for inputs, labels in dataloader:
#         # 前向传播
#         outputs = model(inputs)
#         loss = criterion(outputs, labels)  # 计算损失

#         # 反向传播和优化
#         optimizer.zero_grad()  # 清空梯度
#         loss.backward()  # 反向传播
#         optimizer.step()  # 更新参数

#         running_loss += loss.item()

#     print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {running_loss/len(dataloader):.4f}")

# # 5. 模型评估
# model.eval()  # 设为评估模式
# with torch.no_grad():
#     # 获取一些测试数据
#     sample_input = X[0].unsqueeze(0)  # 选择第一个样本
#     output = model(sample_input)
#     predicted_labels = (output > 0.5).float()  # 使用阈值0.5来决定标签
#     print(f"Predicted Labels: {predicted_labels}")
import torch
from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments
from datasets import load_dataset, Dataset
import numpy as np
from sklearn.model_selection import train_test_split

# 1. 数据准备：示例数据（你可以根据自己的需求处理数据集）
data = [
    {"text": "这家餐厅的食物很美味，但服务太差了。", "aspects": ["食物", "服务"], "labels": ["正面", "负面"]},
    {"text": "这部电影太好看了，演员的演技非常棒。", "aspects": ["演员", "电影"], "labels": ["正面", "正面"]},
    {"text": "这个手机的电池续航很差，屏幕显示还行。", "aspects": ["电池", "屏幕"], "labels": ["负面", "中性"]}
]

# 2. 处理数据：将文本和标签转换成适合训练的数据格式
def process_data(data):
    texts = []
    aspect_labels = []
    
    for item in data:
        text = item['text']
        for i, aspect in enumerate(item['aspects']):
            label = item['labels'][i]
            texts.append(f"{aspect} {text}")  # 拼接方面和文本
            aspect_labels.append(label)
    
    return texts, aspect_labels

texts, aspect_labels = process_data(data)

# 3. 使用BERT进行文本分类
tokenizer = BertTokenizer.from_pretrained('/opt/data/private/huggingface/models--bert-base-chinese/snapshots/c30a6ed22ab4564dc1e3b2ecbf6e766b0611a33f')

# 将数据转换为BERT的输入格式
def tokenize_function(texts):
    return tokenizer(texts, padding=True, truncation=True, max_length=128)

# 处理数据集
train_texts, val_texts, train_labels, val_labels = train_test_split(texts, aspect_labels, test_size=0.2)

train_encodings = tokenize_function(train_texts)
val_encodings = tokenize_function(val_texts)

# 将标签转换为数值（正面=0, 中性=1, 负面=2）
label_map = {"正面": 0, "中性": 1, "负面": 2}
train_labels = [label_map[label] for label in train_labels]
val_labels = [label_map[label] for label in val_labels]

# 4. 创建数据集
train_dataset = Dataset.from_dict({**train_encodings, "labels": train_labels})
val_dataset = Dataset.from_dict({**val_encodings, "labels": val_labels})

# 5. 定义模型
model = BertForSequenceClassification.from_pretrained('/opt/data/private/huggingface/models--bert-base-chinese/snapshots/c30a6ed22ab4564dc1e3b2ecbf6e766b0611a33f', num_labels=3)

# 6. 设置训练参数
training_args = TrainingArguments(
    output_dir='./results',
    num_train_epochs=3,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    warmup_steps=500,
    weight_decay=0.01,
    logging_dir='./logs',
    logging_steps=10,
)

# 7. 训练模型
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
)

trainer.train()

# 8. 模型评估
results = trainer.evaluate()

print(results)

# 9. 模型推理：预测
def predict(text):
    encoding = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=128)
    with torch.no_grad():
        logits = model(**encoding).logits
    predicted_class = torch.argmax(logits, dim=-1).item()
    for label, idx in label_map.items():
        if predicted_class == idx:
            return label

# 测试预测
test_text = "这家餐厅的食物很美味，但服务太差了。"
aspect = "食物"
print(f"Aspect: {aspect}, Sentiment: {predict(f'{aspect} {test_text}')}")


