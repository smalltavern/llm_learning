import math
import re
from random import *
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

# 数据预处理
def make_batch():
    batch = [] 
    positive = negative = 0  # 计数器 为了记录NSP任务中的正样本和负样本的个数，比例最好是在一个batch中接近1：1
    while positive != batch_size/2 or negative != batch_size/2:
        #  抽出来两句话 先随机sample两个index 再通过index找出样本
        tokens_a_index, tokens_b_index= randrange(len(sentences)), randrange(len(sentences))  # 比如tokens_a_index=3，tokens_b_index=1；从整个样本中抽取对应的样本；
        tokens_a, tokens_b= token_list[tokens_a_index], token_list[tokens_b_index]## 根据索引获取对应样本：tokens_a=[5, 23, 26, 20, 9, 13, 18] tokens_b=[27, 11, 23, 8, 17, 28, 12, 22, 16, 25]
        # 拼接
        input_ids = [word_dict['[CLS]']] + tokens_a + [word_dict['[SEP]']] + tokens_b + [word_dict['[SEP]']] ## 加上特殊符号，CLS符号是1，sep符号是2：[1, 5, 23, 26, 20, 9, 13, 18, 2, 27, 11, 23, 8, 17, 28, 12, 22, 16, 25, 2]
        segment_ids = [0] * (1 + len(tokens_a) + 1) + [1] * (len(tokens_b) + 1)##分割句子符号：[0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]

        # MASK LM
        n_pred = min(max_pred, max(1, int(round(len(input_ids) * 0.15))))  # n_pred=3；整个句子的15%的字符可以被mask掉，这里取和max_pred中的最小值，确保每次计算损失的时候没有那么多字符以及信息充足，有15%做控制就够了；其实可以不用加这个，单个句子少了，就要加上足够的训练样本
        # 不让特殊字符参与mask
        cand_maked_pos = [i for i, token in enumerate(input_ids)
                          if token != word_dict['[CLS]'] and token != word_dict['[SEP]']] ## cand_maked_pos=[1, 2, 3, 4, 5, 6, 7, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18]；整个句子input_ids中可以被mask的符号必须是非cls和sep符号的，要不然没意义
        shuffle(cand_maked_pos)## 打乱顺序：cand_maked_pos=[6, 5, 17, 3, 1, 13, 16, 10, 12, 2, 9, 7, 11, 18, 4, 14, 15]  其实取mask对应的位置有很多方法，这里只是一种使用shuffle的方式
        masked_tokens, masked_pos = [], []
        for pos in cand_maked_pos[:n_pred]:  # 取其中的三个；masked_pos=[6, 5, 17] 注意这里对应的是position信息；masked_tokens=[13, 9, 16] 注意这里是被mask的元素之前对应的原始单字数字；
            masked_pos.append(pos)
            masked_tokens.append(input_ids[pos])  # 回到ppt看一下
            if random() < 0.8:  # 80%
                input_ids[pos] = word_dict['[MASK]']  # make mask
            elif random() < 0.5:  # 10%
                index = randint(0, vocab_size - 1) # random index in vocabulary
                input_ids[pos] = word_dict[number_dict[index]] # replace

        # Zero Paddings
        n_pad = maxlen - len(input_ids)##maxlen=30；n_pad=10
        input_ids.extend([0] * n_pad)
        segment_ids.extend([0] * n_pad)# 这里有一个问题，0和之前的重了

        # Zero Padding (100% - 15%) tokens 是为了计算一个batch中句子的mlm损失的时候可以组成一个有效矩阵放进去；不然第一个句子预测5个字符，第二句子预测7个字符，第三个句子预测8个字符，组不成一个有效的矩阵；
        ## 这里非常重要，为什么是对masked_tokens是补零，而不是补其他的字符？？？？我补1可不可以？？
        if max_pred > n_pred:
            n_pad = max_pred - n_pred
            masked_tokens.extend([0] * n_pad)##  masked_tokens= [13, 9, 16, 0, 0] masked_tokens 对应的是被mask的元素的原始真实标签是啥，也就是groundtruth
            masked_pos.extend([0] * n_pad)## masked_pos= [6, 5, 17，0，0] masked_pos是记录哪些位置被mask了

        if tokens_a_index + 1 == tokens_b_index and positive < batch_size/2:
            batch.append([input_ids, segment_ids, masked_tokens, masked_pos, True]) # IsNext
            positive += 1
        elif tokens_a_index + 1 != tokens_b_index and negative < batch_size/2:
            batch.append([input_ids, segment_ids, masked_tokens, masked_pos, False]) # NotNext
            negative += 1
    return batch
# 符号矩阵
def get_attn_pad_mask(seq_q, seq_k): # 在自注意力层q k是一致的
    batch_size, len_q = seq_q.size()
    batch_size, len_k = seq_k.size()
    # eq(zero) is PAD token
    # eq(0)表示和0相等的返回True，不相等返回False。
    pad_attn_mask = seq_k.data.eq(0).unsqueeze(1)  # batch_size x 1 x len_k(=len_q), one is masking
    return pad_attn_mask.expand(batch_size, len_q, len_k)  # 重复了len_q次  batch_size x len_q x len_k 不懂可以看一下例子
def gelu(x):
    "Implementation of the gelu activation function by Hugging Face"
    return x * 0.5 * (1.0 + torch.erf(x / math.sqrt(2.0)))

#Embedding层
class Embedding(nn.Module):
    def __init__(self):
        super(Embedding, self).__init__()
        self.tok_embed = nn.Embedding(vocab_size, d_model)  # token embedding
        self.pos_embed = nn.Embedding(maxlen, d_model)  # position embedding
        self.seg_embed = nn.Embedding(n_segments, d_model)  # segment(token type) embedding
        self.norm = nn.LayerNorm(d_model)
    def forward(self, input_ids, segment_ids):# x对应input_ids, seg对应segment_ids
        seq_len = input_ids.size(1)
        pos = torch.arange(seq_len, dtype=torch.long)
        pos = pos.unsqueeze(0).expand_as(input_ids)  # (seq_len,) -> (batch_size, seq_len)
        embedding = self.tok_embed(input_ids) + self.pos_embed(pos) + self.seg_embed(segment_ids)
        return self.norm(embedding)

# 注意力打分函数
class ScaledDotProductAttention(nn.Module):
    def __init__(self):
        super(ScaledDotProductAttention, self).__init__()

    def forward(self, Q, K, V, attn_pad):
        ## 输入进来的维度分别是 [batch_size x n_heads x len_q x d_k]  K： [batch_size x n_heads x len_k x d_k]  V: [batch_size x n_heads x len_k x d_v]
        ##首先经过matmul函数得到的scores形状是 : [batch_size x n_heads x len_q x len_k]
        scores = torch.matmul(Q, K.transpose(-1, -2)) / np.sqrt(d_k) # scores : [batch_size x n_heads x len_q(=len_k) x len_k(=len_q)]
        ## 然后关键词地方来了，下面这个就是用到了我们之前重点讲的attn_pad，把被pad的地方置为无限小，softmax之后基本就是0，对q的单词不起作用
        scores.masked_fill_(attn_pad, -1e9) # Fills elements of self tensor with value where mask is one.
        attn = nn.Softmax(dim=-1)(scores)
        context = torch.matmul(attn, V)
        return context, attn

#多头注意力机制
class MultiHeadAttention(nn.Module):
    def __init__(self):
        super(MultiHeadAttention, self).__init__()
        ## 输入进来的QKV是相等的，使用映射linear做一个映射得到参数矩阵Wq, Wk,Wv
        self.W_Q = nn.Linear(d_model, d_k * n_heads)
        self.W_K = nn.Linear(d_model, d_k * n_heads)
        self.W_V = nn.Linear(d_model, d_v * n_heads)
    def forward(self, Q, K, V, attn_pad):
        ## 这个多头分为这几个步骤，首先映射分头，然后计算atten_scores，然后计算atten_value;
        ## 输入进来的数据形状： Q: [batch_size x len_q x d_model], K: [batch_size x len_k x d_model], V: [batch_size x len_k x d_model]
        # q: [batch_size x len_q x d_model], k: [batch_size x len_k x d_model], v: [batch_size x len_k x d_model]
        residual, batch_size = Q, Q.size(0)
        # (B, S, D) -proj-> (B, S, D) -split-> (B, S, H, W) -trans-> (B, H, S, W)
        ##下面这个就是先映射，后分头；一定要注意的是q和k分头之后维度是一致额，所以这里都是dk
        q_s = self.W_Q(Q).view(batch_size, -1, n_heads, d_k).transpose(1,2)  # q_s: [batch_size x n_heads x len_q x d_k]
        k_s = self.W_K(K).view(batch_size, -1, n_heads, d_k).transpose(1,2)  # k_s: [batch_size x n_heads x len_k x d_k]
        v_s = self.W_V(V).view(batch_size, -1, n_heads, d_v).transpose(1,2)  # v_s: [batch_size x n_heads x len_k x d_v]
        ## 输入进行的attn_pad形状是 batch_size x len_q x len_k，然后经过下面这个代码得到 新的attn_pad : [batch_size x n_heads x len_q x len_k]，就是把pad信息重复了n个头上
        attn_pad = attn_pad.unsqueeze(1).repeat(1, n_heads, 1, 1)    # repeat 对张量重复扩充
        # context: [batch_size x n_heads x len_q x d_v], attn: [batch_size x n_heads x len_q(=len_k) x len_k(=len_q)]
        context, attn = ScaledDotProductAttention()(q_s, k_s, v_s, attn_pad)
        context = context.transpose(1, 2).contiguous().view(batch_size, -1, n_heads * d_v) # context: [batch_size x len_q x n_heads * d_v]
        output = nn.Linear(n_heads * d_v, d_model)(context)
        return nn.LayerNorm(d_model)(output + residual), attn # output: [batch_size x len_q x d_model]

#基于位置的前馈神经网络
class PoswiseFeedForwardNet(nn.Module):
    def __init__(self): # 对每个字的增强语义向量再做两次线性变换，以增强整个模型的表达能力。
        super(PoswiseFeedForwardNet, self).__init__()
        self.fc1 = nn.Linear(d_model, d_ff)
        self.fc2 = nn.Linear(d_ff, d_model)

    def forward(self, x):
        # (batch_size, len_seq, d_model) -> (batch_size, len_seq, d_ff) -> (batch_size, len_seq, d_model)
        return self.fc2(gelu(self.fc1(x)))

#Encoder
class EncoderLayer(nn.Module):
    def __init__(self):
        super(EncoderLayer, self).__init__()
        self.enc_self_attn = MultiHeadAttention()
        self.pos_ffn = PoswiseFeedForwardNet()

    def forward(self, enc_inputs, enc_self_attn_pad):
        enc_outputs, attn = self.enc_self_attn(enc_inputs, enc_inputs, enc_inputs, enc_self_attn_pad) # enc_inputs to same Q,K,V enc_self_attn_mask是pad符号矩阵
        enc_outputs = self.pos_ffn(enc_outputs) # enc_outputs: [batch_size x len_q x d_model]
        return enc_outputs, attn

## 1. BERT模型整体架构
class BERT(nn.Module):
    def __init__(self):
        super(BERT, self).__init__()
        self.embedding = Embedding() ## 词向量层，构建词表矩阵
        self.layers = nn.ModuleList([EncoderLayer() for _ in range(n_layers)]) ## 把N个encoder堆叠起来，具体encoder实现一会看
        self.fc = nn.Linear(d_model, d_model) ## 前馈神经网络-cls
        self.activ1 = nn.Tanh() ## 激活函数-cls
        self.linear = nn.Linear(d_model, d_model)#-mlm
        self.activ2 = gelu ## 激活函数--mlm
        self.norm = nn.LayerNorm(d_model)
        self.classifier = nn.Linear(d_model, 2)## cls 这是一个分类层，维度是从d_model到2，对应我们架构图中就是这种：

        # decoder is shared with embedding layer
        embed_weight = self.embedding.tok_embed.weight
        n_vocab, n_dim = embed_weight.size()
        self.decoder = nn.Linear(n_dim, n_vocab, bias=False)
        self.decoder.weight = embed_weight
        self.decoder_bias = nn.Parameter(torch.zeros(n_vocab))

    def forward(self, input_ids, segment_ids, masked_pos):
        input = self.embedding(input_ids, segment_ids) # 将input_ids，segment_ids，pos_embed加和

        ##get_attn_pad_mask是为了得到句子中pad的位置信息，给到模型后面，在计算自注意力和交互注意力的时候去掉pad符号的影响，去看一下这个函数 4.
        enc_self_attn_pad = get_attn_pad_mask(input_ids, input_ids)
        for layer in self.layers:
            output, enc_self_attn = layer(input, enc_self_attn_pad) ## enc_self_attn这里是QK转置相乘之后softmax之后的矩阵值，代表的是每个单词和其他单词相关性；
        # output : [batch_size, len, d_model], attn : [batch_size, n_heads, d_mode, d_model]
        h_pooled = self.activ1(self.fc(output[:, 0])) # [batch_size, d_model] cls 对应的位置 可以看一下例子
        logits_clsf = self.classifier(h_pooled) # [batch_size, 2]

        masked_pos = masked_pos[:, :, None].expand(-1, -1, output.size(-1)) # [batch_size, max_pred, d_model]  其中一个 masked_pos= [6, 5, 17，0，0]
        # get masked position from final output of transformer.
        h_masked = torch.gather(output, 1, masked_pos) #在output取出一维对应masked_pos数据 masking position [batch_size, max_pred, d_model]
        h_masked = self.norm(self.activ2(self.linear(h_masked)))
        logits_lm = self.decoder(h_masked) + self.decoder_bias # [batch_size, max_pred, n_vocab]

        return logits_lm, logits_clsf
# 1.从整体到局部
# 2.数据流动形状（输入 输出）
if __name__ == '__main__':
    # BERT Parameters
    maxlen = 30 # 句子的最大长度
    batch_size = 6 # 每一组有多少个句子一起送进去模型
    max_pred = 5  # max tokens of prediction
    n_layers = 6 # number of Encoder of Encoder Layer
    n_heads = 12 # number of heads in Multi-Head Attention
    d_model = 768 # Embedding Size
    d_ff = 3072  # 4*d_model, FeedForward dimension
    d_k = d_v = 64  # dimension of K(=Q), V
    n_segments = 2

    text = (
        'Hello, how are you? I am Romeo.\n'
        'Hello, Romeo My name is Juliet. Nice to meet you.\n'
        'Nice meet you too. How are you today?\n'
        'Great. My baseball team won the competition.\n'
        'Oh Congratulations, Juliet\n'
        'Thanks you Romeo'
    )
    sentences = re.sub("[.,!?\\-]", '', text.lower()).split('\n')  # filter '.', ',', '?', '!'
    word_list = list(set(" ".join(sentences).split()))
    word_dict = {'[PAD]': 0, '[CLS]': 1, '[SEP]': 2, '[MASK]': 3}
    for i, w in enumerate(word_list):
        word_dict[w] = i + 4
    number_dict = {i: w for i, w in enumerate(word_dict)}
    vocab_size = len(word_dict)

    # 把文本转化成数字
    token_list = list()
    for sentence in sentences:
        arr = [word_dict[s] for s in sentence.split()]
        token_list.append(arr)

    batch = make_batch()  # 最重要的一部分  预训练任务的数据构建部分
    input_ids, segment_ids, masked_tokens, masked_pos, isNext = map(torch.LongTensor, zip(*batch))# map把函数依次作用在list中的每一个元素上，得到一个新的list并返回。注意，map不改变原list，而是返回一个新list。

    model = BERT()
    criterion = nn.CrossEntropyLoss(ignore_index=0) # 只计算mask位置的损失
    optimizer = optim.Adam(model.parameters(), lr=0.001)


    for epoch in range(100):
        optimizer.zero_grad()
        # logits_lm 语言词表的输出
        # logits_clsf 二分类的输出
        # logits_lm：[batch_size, max_pred, n_vocab]
        logits_lm, logits_clsf = model(input_ids, segment_ids, masked_pos)## logits_lm 【6，5，29】 bs*max_pred*voca  logits_clsf:[6*2]
        loss_lm = criterion(logits_lm.transpose(1, 2), masked_tokens) # for masked LM ;masked_tokens [6,5]
        loss_lm = (loss_lm.float()).mean()
        loss_clsf = criterion(logits_clsf, isNext) # for sentence classification
        loss = loss_lm + loss_clsf
        if (epoch + 1) % 10 == 0:
            print('Epoch:', '%04d' % (epoch + 1), 'cost =', '{:.6f}'.format(loss))
        loss.backward()
        optimizer.step()