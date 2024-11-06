import os
import math
import torch
from torch import nn
from tqdm import tqdm
from torch.utils.data import Dataset,DataLoader

def read_data(path,num=None):
    with open(path,encoding='utf-8') as f:
        all_data = f.read().split('\n\n')
    if num is not None:
        return all_data[:-1][:num]
    return all_data[:-1]

def build_word_2_index(path):
    with open(path, encoding='utf-8') as f:
        index_2_word = f.read().split('\n')
    word_2_index = {k:v for v,k in enumerate(index_2_word)}
    return word_2_index,index_2_word

class G_dataset(Dataset):
    def __init__(self,all_data,word_2_index,max_seq_len):
        self.all_data = all_data
        self.word_2_index = word_2_index
        self.max_seq_len = max_seq_len
        
    def __getitem__(self, x):
        data = self.all_data[x].split('\n')
        text_idx = []
        for d in data:
            text_idx.extend([word_2_index.get(i,1) for i in d])
            text_idx.append(2)
        text_idx = text_idx[:self.max_seq_len-2]
        input_idx = text_idx[:-1]
        label_idx = text_idx[1:]

        assert len(input_idx) == len(label_idx) ,'sb.长度不一样'
        return input_idx,label_idx,len(input_idx)
    
    def process_data(self,data):
        batch_input,batch_label,batch_len = zip(*data)
        batch_max_len = max(batch_len)
        batch_new_input,batch_new_label = [],[]
        for input_idx,label_idx in zip(batch_input,batch_label):
            batch_new_input.append(input_idx+[word_2_index["<pad>"]]*(batch_max_len-len(input_idx)))
            batch_new_label.append(label_idx+[word_2_index["<pad>"]]*(batch_max_len-len(label_idx)))

        return torch.tensor(batch_new_input),torch.tensor(batch_new_label)
    
    def __len__(self):
        return len(self.all_data)

class EmbeddingLayer(nn.Module):
    def __init__(self):
        super().__init__()
        self.pos_emb = nn.Embedding(max_seq_len,d_model)
        self.token_emb = nn.Embedding(vocab_len,d_model)
        
    def forward(self,x):
        position = torch.arange(0, x.shape[1], device=x.device).reshape(1, -1)
        position = position.expand_as(x)
        x = self.token_emb(x) + self.pos_emb(position)

        return x

class Feed_Forward(nn.Module):
    def __init__(self):
        super().__init__()
        self.lin1 = nn.Linear(d_model,d_ff)
        self.relu = nn.ReLU()
        self.lin2 = nn.Linear(d_ff,d_model)
        self.layer_norm = nn.LayerNorm(768)
        
    def forward(self, x):
        x_copy = x.clone()
        x = self.lin1(x)
        x = self.relu(x)
        x = self.lin2(x)

        x = x + x_copy
        x = self.layer_norm(x)
        return x

class MultiHeadAttention(nn.Module):
    def __init__(self,nhead):
        super().__init__()
        assert 768%nhead==0
        self.sqrt_d_k = math.sqrt(768//nhead)
        self.Q = nn.Linear(768,768,bias=False)
        self.K = nn.Linear(768,768,bias=False)
        self.V = nn.Linear(768,768,bias=False)
        self.nhead = nhead
        self.softmax = nn.Softmax(dim=2)
        self.layer_norm = nn.LayerNorm(768)
        
    def forward(self, x,mask):
        batch,seq_len,emb_num = x.shape
        x_copy = x.clone()

        #x = x.reshape()
        q = self.Q(x)
        q = q.reshape(batch,seq_len,self.nhead,-1).transpose(1,2)

        k = self.V(x)
        k = k.reshape(batch, seq_len, self.nhead, -1).transpose(1, 2)

        v = self.V(x)
        v = v.reshape(batch, seq_len, self.nhead, -1).transpose(1, 2)

        weight =  q @ k.transpose(-1,-2)/self.sqrt_d_k

        weight.masked_fill_(mask,-1e9)
        score = self.softmax(weight)
        x = score @ v

        x = x.transpose(1,2).reshape(batch,seq_len,emb_num)
        x = x+x_copy
        x = self.layer_norm(x)
        return x

class DecoderBlock(nn.Module):
    def __init__(self,nhead):
        super().__init__()
        self.attention_block1 = MultiHeadAttention(nhead=nhead)
        self.attention_block2 = MultiHeadAttention(nhead=nhead)

        self.feed = Feed_Forward()
        
    def forward(self,x,mask):
        x = self.attention_block1(x,mask)
        mask = torch.zeros_like(mask,device=x.device,requires_grad=False)
        x = self.attention_block2(x,mask)
        x = self.feed(x)

        return x

def get_attention_mask(x):
    attention_mask = x==0
    attention_mask = attention_mask.unsqueeze(-1)
    return attention_mask

class Decoder(nn.Module):
    def __init__(self,nhead):
        super().__init__()
        self.embedding = EmbeddingLayer()
        self.layers = nn.ModuleList([DecoderBlock(nhead = nhead) for i in range(3)])
        self.nhead = nhead

    def forward(self,x):
        attention_mask = get_attention_mask(x)

        seq_len = x.shape[1]
        # attention_mask
        attention_mask = attention_mask.unsqueeze(1).repeat(1, 1, 1, seq_len)
        attention_mask = attention_mask.repeat(1, self.nhead, 1, 1)
        # look_ahead_mask
        look_ahead_mask = torch.triu(torch.ones_like(attention_mask), 1)

        mask = (attention_mask | look_ahead_mask).to(x.device)
        x = self.embedding(x)
        for layer in self.layers:
            x = layer(x,mask)
        return x

class GPT_Model(nn.Module):
    def __init__(self,vocab_len,nhead=4):
        super().__init__()
        self.decoder = Decoder(nhead=nhead)
        self.cls = nn.Linear(768,vocab_len)
        self.loss_fn = nn.CrossEntropyLoss(ignore_index=0)

    def forward(self,x,y=None):
        x =self.decoder(x)
        x = self.cls(x)

        if y is not None:
            loss = self.loss_fn(x.reshape(-1,x.shape[-1]),y.reshape(-1))
            return loss
        return x
    
    def answer1(self,input_text):
        input_idx = [word_2_index.get(i,1) if i != '\n' else word_2_index['<sep>'] for i in input_text]
        input_idx = torch.tensor([input_idx],device=device)


        while True:
            pre = self.forward(input_idx)
            pre = torch.argmax(pre,dim=-1)
            pre = int(pre[0][-1])
            input_idx = torch.cat((input_idx, torch.tensor([[pre]],device=device)), dim=1)

            if pre==2:
                break
        return input_idx.tolist()[0]
    
    def answer2(self,input_text):
        input_idx = [word_2_index.get(i,1) if i != '\n' else word_2_index['<sep>'] for i in input_text]
        input_idx = torch.tensor([input_idx],device=device)


        while True:
            pre = self.forward(input_idx)

            pre = torch.sort(pre)[1][0][-1].tolist()
            pre = int(pre[0][-1])
            input_idx = torch.cat((input_idx, torch.tensor([[pre]],device=device)), dim=1)

            if pre==2:
                break
        return input_idx.tolist()[0]


if __name__ == '__main__':
    all_data = read_data(os.path.join('data','train.txt'))
    word_2_index,index_2_word = build_word_2_index(os.path.join('data','vocab.txt'))

    vocab_len = len(word_2_index)
    batch_size = 50
    epoch = 5
    max_seq_len = 128
    lr = 0.0002
    nhead = 8
    d_ff = 2048
    d_model = 768
    is_train = True
    CLIP = 1
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_dataset = G_dataset(all_data, word_2_index, max_seq_len)
    train_dataloader = DataLoader(train_dataset, shuffle=True, batch_size=batch_size, collate_fn=train_dataset.process_data)

    model = GPT_Model(vocab_len,nhead=nhead).to(device)
    opt = torch.optim.Adam(model.parameters(),lr = lr)

    if is_train:
        for e in range(epoch):
            for bi,(x,y) in enumerate(tqdm(train_dataloader)):
                x = x.to(device)
                y = y.to(device)

                opt.zero_grad()
                loss = model(x,y)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), CLIP)
                opt.step()
            torch.save(model.state_dict(),f'model_{e}.pt')

    model.eval()
    #test
    while True:
        input_text = input('请输入:') +'\n'
        pre = model.answer1(input_text)
        pre = [index_2_word[i] for i in pre]
        print(pre)
