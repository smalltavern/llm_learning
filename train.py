import torch
from transformers import BertTokenizer, BertForSequenceClassification, get_linear_schedule_with_warmup, Trainer, TrainingArguments, BertModel, AdamW
from datasets import load_dataset, Dataset
import numpy as np
import tqdm
import torch.nn as nn
from sklearn.model_selection import train_test_split
from sklearn import metrics


class Config(object):

    """配置参数"""
    def __init__(self, dataset):
        self.model_name = 'bert'
        self.train_path = dataset + '/homework/train.txt'                                # 训练集                                  # 验证集
        self.test_path = dataset + '/homework/test.txt'                                  # 测试集
        self.class_list = [x.strip() for x in open(
            dataset + '/homework/class.txt').readlines()]                                # 类别名单
        self.save_path = dataset + '/saved_dict/' + self.model_name       # 模型训练结果
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')   # 设备

        self.require_improvement = 1000                                 # 若超过1000batch效果还没提升，则提前结束训练
        self.num_classes = len(self.class_list)                         # 类别数
        self.num_epochs = 50                                        # epoch数
        self.batch_size = 32                                           # mini-batch大小
        self.pad_size = 64                                              # 每句话处理成的长度(短填长切)
        self.learning_rate = 5e-5                                       # 学习率
        self.bert_path = '/opt/data/private/huggingface/models--bert-base-chinese/snapshots/c30a6ed22ab4564dc1e3b2ecbf6e766b0611a33f'
        self.tokenizer = BertTokenizer.from_pretrained(self.bert_path)
        self.hidden_size = 768
        
        
class BModel(nn.Module):

    def __init__(self, config):
        super(BModel, self).__init__()
        self.bert = BertModel.from_pretrained(config.bert_path)
        for param in self.bert.parameters():
            param.requires_grad = True
        self.fc = nn.Linear(config.hidden_size, config.num_classes)
        self.sigmod = nn.Sigmoid()

    def forward(self, x):
        context = x[0]  # 输入的句子
        mask = x[2]  # 对padding部分进行mask，和句子一个size，padding部分用0表示，如：[1, 1, 1, 1, 0, 0]
        out = self.bert(context, attention_mask=mask)
        out = out.last_hidden_state[:, 0]
        out = self.fc(out)
        out = self.sigmod(out)
        return out
    
    
dataset = 'data'
config = Config(dataset)
PAD, CLS = '[PAD]', '[CLS]'
def build_dataset(config):
    def load_dataset(path, pad_size=64):
        contents = []
        with open(path, 'r', encoding='UTF-8') as f:
            for line in f:
                lin = line.strip()
                if not lin:
                    continue
                content = lin.split('\t')[0]
                label_list = [w.split('#')[0] for w in lin.split('\t')[1:]]
                label = [0] * config.num_classes
                for l in label_list:
                    label[config.class_list.index(l)] = 1
                token = config.tokenizer.tokenize(content)
                token = [CLS] + token
                seq_len = len(token)
                mask = []
                token_ids = config.tokenizer.convert_tokens_to_ids(token)

                if pad_size:
                    if len(token) < pad_size:
                        mask = [1] * len(token_ids) + [0] * (pad_size - len(token))
                        token_ids += ([0] * (pad_size - len(token)))
                    else:
                        mask = [1] * pad_size
                        token_ids = token_ids[:pad_size]
                        seq_len = pad_size
                contents.append((token_ids, label, seq_len, mask))
        return contents
    train = load_dataset(config.train_path, config.pad_size)
    test = load_dataset(config.test_path, config.pad_size)
    return train, test
train_data, test_data = build_dataset(config)

class DatasetIterater(object):
    def __init__(self, batches, batch_size, device):
        self.batch_size = batch_size
        self.batches = batches
        self.n_batches = len(batches) // batch_size
        self.residue = False  # 记录batch数量是否为整数
        if len(batches) % self.n_batches != 0:
            self.residue = True
        self.index = 0
        self.device = device

    def _to_tensor(self, datas):
        x = torch.LongTensor([_[0] for _ in datas]).to(self.device)
        y = torch.LongTensor([_[1] for _ in datas]).to(self.device)

        # pad前的长度(超过pad_size的设为pad_size)
        seq_len = torch.LongTensor([_[2] for _ in datas]).to(self.device)
        mask = torch.LongTensor([_[3] for _ in datas]).to(self.device)
        return (x, seq_len, mask), y

    def __next__(self):
        if self.residue and self.index == self.n_batches:
            batches = self.batches[self.index * self.batch_size: len(self.batches)]
            self.index += 1
            batches = self._to_tensor(batches)
            return batches

        elif self.index >= self.n_batches:
            self.index = 0
            raise StopIteration
        else:
            batches = self.batches[self.index * self.batch_size: (self.index + 1) * self.batch_size]
            self.index += 1
            batches = self._to_tensor(batches)
            return batches

    def __iter__(self):
        return self

    def __len__(self):
        if self.residue:
            return self.n_batches + 1
        else:
            return self.n_batches
        
        
# train_iter = DatasetIterater(train_data, config.batch_size, config.device)
# test_iter = DatasetIterater(test_data, config.batch_size, config.device)
# print(len(train_iter))


def evaluate(config, model, data_iter, test=False):
    model.eval()
    loss_total = 0
    predict_all = []
    labels_all = []
    criterion = nn.BCELoss()
    with torch.no_grad():
        for texts, labels in data_iter:
            outputs = model(texts)
            loss = criterion(outputs, labels.float())
            loss_total += loss
            labels = labels.data.cpu().numpy()
            predic = (outputs.data > 0.5).int().cpu().numpy()
            assert len(labels) == len(predic)
            for i in range(len(labels)):
                labels_all.append(labels[i])
                predict_all.append(predic[i])

    acc = metrics.accuracy_score(labels_all, predict_all)
    if test:
        report = metrics.classification_report(np.array(labels_all), np.array(predict_all), target_names=config.class_list, digits=4)
        return acc, loss_total / len(data_iter), report
    return acc, loss_total / len(data_iter)

def test(config, model, test_iter):
    # test
    model.load_state_dict(torch.load(config.save_path + '_{}.ckpt'.format(config.num_epochs - 1)))
    model.eval()
    test_acc, test_loss, test_report, test_confusion = evaluate(config, model, test_iter, test=True)
    msg = 'Test Loss: {0:>3.2},  Test Acc: {1:>3.2%}'
    print(msg.format(test_loss, test_acc))
    print("Precision, Recall and F1-Score...")
    print(test_report)

def train(config, model, train_iter, test_iter):
    model.train()
    param_optimizer = list(model.named_parameters())
    no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
        {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}]
    optimizer = AdamW(optimizer_grouped_parameters,
                         lr=config.learning_rate)
    criterion = nn.BCELoss()
    
    total_batch = 0  # 记录进行到多少batch
    flag = False  # 记录是否很久没有效果提升
    train_best_loss = float('inf')
    model.train()
    for epoch in range(config.num_epochs):
        print('Epoch [{}/{}]'.format(epoch + 1, config.num_epochs))
        for i, (trains, labels) in enumerate(train_iter):
            outputs = model(trains)
            model.zero_grad()
            loss = criterion(outputs, labels.float())
            loss.backward()
            optimizer.step()
            if total_batch % 10 == 0:
                # 每多少轮输出在训练集和验证集上的效果
                true = labels.data.cpu()
                predic = (outputs.data > 0.5).int().cpu()
                train_acc = metrics.accuracy_score(true, predic)
                msg = 'Iter: {0:>3},    Train Loss: {1:>3.2},     Train Acc: {2:>3.2%}'
                print(msg.format(total_batch, loss.item(), train_acc))
            total_batch += 1
        torch.save(model.state_dict(), config.save_path+ '_{}.ckpt'.format(epoch))
    test(config, model, test_iter)
            
            
# model = BModel(config).to(config.device)
# train(config, model, train_iter, test_iter)


def build_ADBSdataset(config):
    def load_dataset(path, pad_size=64):
        contents = []
        with open(path, 'r', encoding='UTF-8') as f:
            for line in f:
                lin = line.strip()
                if not lin:
                    continue
                content = lin.split('\t')[0]
                aspect_list = [w.split('#')[0] for w in lin.split('\t')[1:]]
                label = [int(w.split('#')[1]) + 1 for w in lin.split('\t')[1:]]
                for i in range(len(aspect_list)): 
                    token = config.tokenizer.tokenize(aspect_list[i] + content)
                    token = [CLS] + token
                    seq_len = len(token)
                    mask = []
                    token_ids = config.tokenizer.convert_tokens_to_ids(token)

                    if pad_size:
                        if len(token) < pad_size:
                            mask = [1] * len(token_ids) + [0] * (pad_size - len(token))
                            token_ids += ([0] * (pad_size - len(token)))
                        else:
                            mask = [1] * pad_size
                            token_ids = token_ids[:pad_size]
                            seq_len = pad_size
                    contents.append([token_ids, np.eye(3)[label[i]], seq_len, mask])
        return contents
    train = load_dataset(config.train_path, config.pad_size)
    test = load_dataset(config.test_path, config.pad_size)
    return train, test

train_ADBSdata, test_ADBSdata = build_ADBSdataset(config)

from torch.utils.data import Dataset, DataLoader
class ABSADataset(Dataset):
    def __init__(self, data, device):
        super(ABSADataset, self).__init__()
        self.dataset = data
        self.device = device

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        text = torch.LongTensor(self.dataset[idx][0]).to(self.device)
        label = torch.LongTensor(self.dataset[idx][1]).to(self.device)
        seq_len = torch.LongTensor([self.dataset[idx][2]]).to(self.device)
        mask = torch.LongTensor(self.dataset[idx][3]).to(self.device)

        return text, seq_len, mask, label
    
train_ADBSdataset = ABSADataset(train_ADBSdata, config.device)
train_loader = DataLoader(train_ADBSdataset, batch_size=config.batch_size, shuffle=True)
test_ADBSdataset = ABSADataset(test_ADBSdata, config.device)
test_loader = DataLoader(test_ADBSdataset, batch_size=config.batch_size, shuffle=True)

model = BertForSequenceClassification.from_pretrained(config.bert_path, num_labels=3)
model.to(config.device)
optimizer = AdamW(model.parameters(), lr=5e-5)
total_steps = len(train_loader) * config.num_epochs
scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=total_steps)
print(len(train_loader))


for epoch in range(1):
    model.train()
    for step, (input_ids, seq_len, attention_mask, labels) in enumerate(train_loader):
        input_ids = input_ids
        attention_mask = attention_mask
        labels = labels.float()
        
        outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs.loss
        loss.backward()

        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()

        if step % 10 == 0:
            print(f'Epoch: {epoch}, Step: {step}, Loss: {loss.item()}')
            
            
all_predictions = []
all_labels = []

with torch.no_grad():
    for step, (input_ids, seq_len, attention_mask, labels) in enumerate(test_loader):
        input_ids = input_ids
        attention_mask = attention_mask
        labels = labels.float()

        outputs = model(input_ids, attention_mask=attention_mask)
        logits = outputs.logits

        # 将预测结果从张量转换为列表
        predictions = logits.squeeze(-1).cpu().numpy().tolist()
        labels = labels.cpu().numpy().tolist()
        print(len(predictions[0]))
        all_predictions.extend(predictions)
        all_labels.extend(labels)
print(all_predictions)