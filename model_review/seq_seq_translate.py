import torch
import torch.nn as nn
import pandas as pd
import pickle
from torch.utils.data import Dataset, DataLoader


def get_datas(file="datas/translate.csv", nums=None):
    all_datas = pd.read_csv(file)
    en_datas = list(all_datas["english"])
    ch_datas = list(all_datas["chinese"])
    if nums == None:
        return en_datas, ch_datas
    else:
        return en_datas[:nums], ch_datas[:nums]
    
    
def translate(sentence):
    global en_word_2_index,model,device,ch_word_2_index,ch_index_2_word
    en_index = torch.tensor([[en_word_2_index[i] for i in sentence]],device=device)

    result = []
    encoder_hidden = model.encoder(en_index)
    decoder_input = torch.tensor([[ch_word_2_index["<BOS>"]]],device=device)

    decoder_hidden = encoder_hidden
    while True:
        decoder_output,decoder_hidden = model.decoder(decoder_input,decoder_hidden)
        pre = model.classifier(decoder_output)

        w_index = int(torch.argmax(pre,dim=-1))
        word = ch_index_2_word[w_index]

        if word == "<EOS>" or len(result) > 50:
            break

        result.append(word)
        decoder_input = torch.tensor([[w_index]],device=device)

    print("译文: ","".join(result))
    
class MyDataset(Dataset):
    def __init__(self, en_datas, ch_datas, en_word_2_index, ch_word_2_index):
        self.en_datas = en_datas
        self.ch_datas = ch_datas
        self.en_word_2_index = en_word_2_index
        self.ch_word_2_index = ch_word_2_index
        
    
    def __len__(self):
        assert len(self.en_datas) == len(self.ch_datas)
        return len(self.en_datas)
    
    def __getitem__(self, index):
        en = self.en_datas[index]
        ch = self.ch_datas[index]
        
        en_index = [self.en_word_2_index[i] for i in en]
        ch_index = [self.ch_word_2_index[i] for i in ch]
        return en_index, ch_index
    
    def batch_data_process(self, batch_data):
        en_index, ch_index = [], []
        en_len, ch_len = [], []
        for en, ch in batch_data:
            en_index.append(en)
            ch_index.append(ch)
            en_len.append(len(en))
            ch_len.append(len(ch))
        max_en_len = max(en_len)
        max_ch_len = max(ch_len)
        en_index = [ i + [self.en_word_2_index["<PAD>"]] * (max_en_len - len(i))   for i in en_index]
        ch_index = [[self.ch_word_2_index["<BOS>"]]+ i + [self.ch_word_2_index["<EOS>"]] + [self.ch_word_2_index["<PAD>"]] * (max_ch_len - len(i))   for i in ch_index]
        en_index = torch.tensor(en_index,device = device)
        ch_index = torch.tensor(ch_index,device = device)
        
        return en_index, ch_index
    
    
class Encoder(nn.Module):
    def __init__(self, encoder_embeding_num, encoder_hidden_num, en_corpus_num):
        super().__init__()
        self.embeding = nn.Embedding(en_corpus_num, encoder_embeding_num)
        self.lstm = nn.LSTM(encoder_embeding_num, encoder_hidden_num, batch_first=True)
        
    def forward(self, en_index):
        en_embedding = self.embeding(en_index)
        _, encoder_hidden = self.lstm(en_embedding)
        
        return encoder_hidden
    

class Decoder(nn.Module):
    def __init__(self, decoder_embeding_num, decoder_hidden_num, ch_corpus_num):
        super().__init__()
        self.embeding = nn.Embedding(ch_corpus_num, decoder_embeding_num)
        self.lstm = nn.LSTM(decoder_embeding_num, decoder_hidden_num, batch_first=True)
    
    def forward(self, decoder_input, hidden):
        embedding = self.embeding(decoder_input)
        dec_out, dec_hidden = self.lstm(embedding, hidden)
        return dec_out, dec_hidden
    
    
class Seq2Seq(nn.Module):
    def __init__(self, encoder_embeding_num, encoder_hidden_num, en_corpus_num, decoder_embeding_num, decoder_hidden_num, ch_corpus_num):
        super().__init__()
        self.encode = Encoder(encoder_embeding_num, encoder_hidden_num, en_corpus_num)
        self.decode = Decoder(decoder_embeding_num, decoder_hidden_num, ch_corpus_num)
        self.classifier = nn.Linear(decoder_hidden_num, ch_corpus_num)
        self.cross_loss = nn.CrossEntropyLoss()
        
        
    def forward(self, en_index, ch_index):
        decoder_input = ch_index[:, :-1]
        label = ch_index[:, 1:]
        
        encoder_hidden = self.encode(en_index)
        dec_out, _ = self.decode(decoder_input, encoder_hidden)
        
        pre = self.classifier(dec_out)
        loss = self.cross_loss(pre.reshape(-1, pre.shape[-1]), label.reshape(-1))
        
        return loss
        
    
    
    
if __name__ == "__main__":
    epoch = 1000
    lr = 0.001
    batch_size = 64
    encoder_embeding_num = 50
    encoder_hidden_num = 100
    decoder_embeding_num = 107
    decoder_hidden_num = 100
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    with open("datas/ch.vec", "rb") as f:
        _, ch_word_2_index, ch_index_2_word = pickle.load(f)
    with open("datas/en.vec", "rb") as f:
        _, en_word_2_index, en_index_2_word = pickle.load(f)
    en_datas, ch_datas = get_datas(nums=10000)
    
    ch_corpus_len, en_corpus_len = len(ch_word_2_index), len(en_word_2_index)
    ch_word_2_index.update({"<PAD>": ch_corpus_len, "<BOS>": ch_corpus_len+1, "<EOS>": ch_corpus_len+2})
    en_word_2_index.update({"<PAD>":en_corpus_len})
    ch_index_2_word += ["<PAD>", "<BOS>", "<EOS>"]
    en_index_2_word += ["<PAD>"]
    ch_corpus_len += 3
    en_corpus_len += 1
    
    
    dataset = MyDataset(en_datas, ch_datas, en_word_2_index, ch_word_2_index)
    dataLoader = DataLoader(dataset, batch_size=batch_size, shuffle=False,collate_fn=dataset.batch_data_process)
    
    model = Seq2Seq(encoder_embeding_num=encoder_embeding_num, 
                    encoder_hidden_num=encoder_hidden_num, 
                    decoder_embeding_num=decoder_embeding_num, 
                    decoder_hidden_num=decoder_hidden_num, 
                    ch_corpus_num=ch_corpus_len, 
                    en_corpus_num=en_corpus_len).to(device)
    
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    
    for e in range(epoch):
        for en_index, ch_index in dataLoader:
            loss = model(en_index, ch_index)
            loss.backward()
            opt.step()
            opt.zero_grad()
        
        print(f"epoch:{epoch},loss:{loss:.3f}")
        

    while True:
        s = input("请输入英文: ")
        translate(s)
    