# pytorch实现seq2seq+attention转换日期
***
使用keras实现加入注意力机制的seq2seq比较麻烦，所以这里我尝试使用机器翻译的seq2seq+attention模型实现人造日期对标准日期格式的转换。


所copy的代码来自[practical-pytorch教程](https://github.com/spro/practical-pytorch/tree/master/seq2seq-translation)，以及[pytorch-seq2seq教程](https://github.com/bentrevett/pytorch-seq2seq)


所用的数据来自[注意力机制keras实现](https://github.com/Choco31415/Attention_Network_With_Keras/tree/master/data)。   
python3   
pytorch版本 0.4.0  
可能需要GPU

```{.python .input  n=29}
import json
from matplotlib import ticker
from numpy import *
from collections import Counter
import matplotlib.pyplot as plt
import torch
from torch import nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
device
```

```{.json .output n=29}
[
 {
  "data": {
   "text/plain": "device(type='cuda')"
  },
  "execution_count": 29,
  "metadata": {},
  "output_type": "execute_result"
 }
]
```

## 预处理 
---
这里先生成字符和数字相互转换的字典，如果是句子也可以按照词为单位。我在字典的开头添加了4种表示。

```{.python .input  n=7}
def build_vocab(texts, n=None):
    counter = Counter(''.join(texts))  # char level
    char2index = {w: i for i, (w, c) in enumerate(counter.most_common(n), start=4)}
    char2index['~'] = 0  # pad  不足长度的文本在后边填充0
    char2index['^'] = 1  # sos  表示句子的开头
    char2index['$'] = 2  # eos  表示句子的结尾
    char2index['#'] = 3  # unk  表示句子中出现的字典中没有的未知词
    index2char = {i: w for w, i in char2index.items()}
    return char2index, index2char
```

先看一下数据的格式。

```{.python .input  n=8}
pairs = json.load(open('Time Dataset.json', 'rt', encoding='utf-8'))
print(pairs[:1])
```

```{.json .output n=8}
[
 {
  "name": "stdout",
  "output_type": "stream",
  "text": "[['six hours and fifty five am', '06:55']]\n"
 }
]
```

我们将目标文本和原文本分开，求出两边句子的最大长度，然后建立两边各自的字典。

```{.python .input  n=211}
data = array(pairs)
src_texts = data[:, 0]
trg_texts = data[:, 1]
src_c2ix, src_ix2c = build_vocab(src_texts)
trg_c2ix, trg_ix2c = build_vocab(trg_texts)
```

这里按批量跟新，定义一个随机批量生成的函数，它能够将文本转换成字典中的数字表示，并同时返回batch_size个样本和它们的长度，这些样本按照长度降序排序。pad的长度以batch中最长的为准。这主要是为了适应pack_padded_sequence这个函数，因为输入RNN的序列不需要将pad标志也输入RNN中计算，RNN只需要循环计算到其真实长度即可。

```{.python .input  n=213}
def indexes_from_text(text, char2index):
    return [1] + [char2index[c] for c in text] + [2]  # 手动添加开始结束标志
def pad_seq(seq, max_length):
    seq += [0 for _ in range(max_length - len(seq))]
    return seq

max_src_len = max(list(map(len, src_texts))) + 2
max_trg_len = max(list(map(len, trg_texts))) + 2
max_src_len, max_trg_len
```

```{.json .output n=213}
[
 {
  "data": {
   "text/plain": "(43, 7)"
  },
  "execution_count": 213,
  "metadata": {},
  "output_type": "execute_result"
 }
]
```

```{.python .input  n=214}
def random_batch(batch_size, pairs, src_c2ix, trg_c2ix):
    input_seqs, target_seqs = [], []

    for i in random.choice(len(pairs), batch_size):
        input_seqs.append(indexes_from_text(pairs[i][0], src_c2ix))
        target_seqs.append(indexes_from_text(pairs[i][1], trg_c2ix))

    seq_pairs = sorted(zip(input_seqs, target_seqs), key=lambda p: len(p[0]), reverse=True)
    input_seqs, target_seqs = zip(*seq_pairs)
    input_lengths = [len(s) for s in input_seqs]
    input_padded = [pad_seq(s, max(input_lengths)) for s in input_seqs]
    target_lengths = [len(s) for s in target_seqs]
    target_padded = [pad_seq(s, max(target_lengths)) for s in target_seqs]

    input_var = torch.LongTensor(input_padded).transpose(0, 1)  
    # seq_len x batch_size
    target_var = torch.LongTensor(target_padded).transpose(0, 1)
    input_var = input_var.to(device)
    target_var = target_var.to(device)

    return input_var, input_lengths, target_var, target_lengths
```

可以先打印一下，batch_size=3时的返回结果。注意这里batch经过了转置。

```{.python .input  n=215}
random_batch(3, data, src_c2ix, trg_c2ix)
```

```{.json .output n=215}
[
 {
  "data": {
   "text/plain": "(tensor([[  1,   1,   1],\n         [ 12,  23,   6],\n         [  7,   9,  18],\n         [ 27,  26,  21],\n         [ 10,  23,  23],\n         [  4,   4,  25],\n         [ 16,  17,   2],\n         [  7,   9,   0],\n         [ 27,  11,   0],\n         [ 10,   9,   0],\n         [ 19,   2,   0],\n         [  4,   0,   0],\n         [ 13,   0,   0],\n         [  8,   0,   0],\n         [ 32,   0,   0],\n         [  4,   0,   0],\n         [  6,   0,   0],\n         [ 31,   0,   0],\n         [  5,   0,   0],\n         [  8,   0,   0],\n         [  6,   0,   0],\n         [ 20,   0,   0],\n         [  4,   0,   0],\n         [ 12,   0,   0],\n         [ 14,   0,   0],\n         [ 28,   0,   0],\n         [  5,   0,   0],\n         [  4,   0,   0],\n         [ 13,   0,   0],\n         [ 12,   0,   0],\n         [  6,   0,   0],\n         [  5,   0,   0],\n         [ 10,   0,   0],\n         [  4,   0,   0],\n         [  8,   0,   0],\n         [  7,   0,   0],\n         [  7,   0,   0],\n         [  8,   0,   0],\n         [  2,   0,   0]], device='cuda:0'),\n [39, 11, 7],\n tensor([[  1,   1,   1],\n         [  6,   6,   5],\n         [ 13,   9,   7],\n         [  4,   4,   4],\n         [  7,   5,   8],\n         [  9,   8,  10],\n         [  2,   2,   2]], device='cuda:0'),\n [7, 7, 7])"
  },
  "execution_count": 215,
  "metadata": {},
  "output_type": "execute_result"
 }
]
```

## 模型定义
---
模型分为encoder和decoder两个部分，decoder部分比较简单，就是一层Embedding层加上两层GRU。之前处理的batch的格式主要是为了使用pack_padded_sequence和pad_packed_sequence这两个类对GRU输入输出批量处理。一定要注意各个变量的shape。

```{.python .input  n=216}
class Encoder(nn.Module):
    def __init__(self, input_dim, embedding_dim, hidden_dim, num_layers=2, dropout=0.2):
        super().__init__()

        self.input_dim = input_dim
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.dropout = dropout
        # input_dim = vocab_size + 1
        self.embedding = nn.Embedding(input_dim, embedding_dim)

        self.rnn = nn.GRU(embedding_dim, hidden_dim,
                          num_layers=num_layers, dropout=dropout)

        self.dropout = nn.Dropout(dropout)

    def forward(self, input_seqs, input_lengths, hidden=None):
        # src = [sent len, batch size]
        embedded = self.dropout(self.embedding(input_seqs))

        # embedded = [sent len, batch size, emb dim]
        packed = torch.nn.utils.rnn.pack_padded_sequence(embedded, input_lengths)

        outputs, hidden = self.rnn(packed, hidden)
        outputs, output_lengths = torch.nn.utils.rnn.pad_packed_sequence(outputs)

        # outputs, hidden = self.rnn(embedded, hidden)
        # outputs = [sent len, batch size, hid dim * n directions]
        # hidden = [n layers, batch size, hid dim]

        # outputs are always from the last layer
        return outputs, hidden
```

首先定义一下Attention层，这里主要是对encoder的输出进行attention操作，也可以直接对embedding层的输出进行attention。   
论文[Neural Machine Translation by Jointly Learning to Align and Translate](https://arxiv.org/abs/1409.0473)中定义了attention的计算公式。 

decoder的输出取决于decoder先前的输出和 $\mathbf x$, 这里 $\mathbf x$ 包括当前GRU输出的hidden state（这部分已经考虑了先前的输出） 以及attention（上下文向量，由encoder的输出求得）。 计算公式如下：函数 $g$ 非线性激活的全连接层，输入是 $y_{i-1}$, $s_i$, and $c_i$ 三者的拼接。

$$
p(y_i \mid \{y_1,...,y_{i-1}\},\mathbf{x}) = g(y_{i-1}, s_i, c_i)
$$

所谓的上下文向量就是对encoder的所有输出进行加权求和，$a_{ij}$ 表示输出的第 i 个词对encoder第 j 个输出 $h_j$ 的权重。

$$
c_i = \sum_{j=1}^{T_x} a_{ij} h_j
$$

每个 $a_{ij}$ 通过对所有 $e_{ij}$ 进行softmax，而每个 $e_{ij}$ 是decoder的上一个hidden state $s_{i-1}$ 和指定的encoder的输出 $h_j$ 经过某些线性操作 $a$ 计算得分。

$$
a_{ij} = \dfrac{exp(e_{ij})}{\sum_{k=1}^{T} exp(e_{ik})} 
\\
e_{ij} = a(s_{i-1}, h_j)
$$

此外，论文[Effective Approaches to Attention-based Neural Machine Translation](https://arxiv.org/abs/1508.04025)中提出了计算分值的不同方式。这里用到的是第三种。

$$
score(h_t, \bar h_s) =
\begin{cases}
h_t ^\top \bar h_s & dot \\
h_t ^\top \textbf{W}_a \bar h_s & general \\
v_a ^\top \textbf{W}_a [ h_t ; \bar h_s ] & concat
\end{cases}
$$

```{.python .input  n=217}
class Attention(nn.Module):
    def __init__(self, hidden_dim):
        super(Attention, self).__init__()
        self.hidden_dim = hidden_dim
        self.attn = nn.Linear(self.hidden_dim * 2, hidden_dim)
        self.v = nn.Parameter(torch.rand(hidden_dim))
        self.v.data.normal_(mean=0, std=1. / np.sqrt(self.v.size(0)))

    def forward(self, hidden, encoder_outputs):
        #  encoder_outputs:(seq_len, batch_size, hidden_size)
        #  hidden:(num_layers * num_directions, batch_size, hidden_size)
        max_len = encoder_outputs.size(0)

        h = hidden[-1].repeat(max_len, 1, 1)
        # (seq_len, batch_size, hidden_size)

        attn_energies = self.score(h, encoder_outputs)  # compute attention score
        return F.softmax(attn_energies, dim=1)  # normalize with softmax

    def score(self, hidden, encoder_outputs):
        # (seq_len, batch_size, 2*hidden_size)-> (seq_len, batch_size, hidden_size)
        energy = F.tanh(self.attn(torch.cat([hidden, encoder_outputs], 2)))
        energy = energy.permute(1, 2, 0)  # (batch_size, hidden_size, seq_len)
        v = self.v.repeat(encoder_outputs.size(1), 1).unsqueeze(1)  # (batch_size, 1, hidden_size)
        energy = torch.bmm(v, energy)  # (batch_size, 1, seq_len)
        return energy.squeeze(1)  # (batch_size, seq_len)
```

下面是加了attention层的decoder，GRU的输出进过全连接层后，又进行了log_softmax操作计算输出词的概率，主要是为了方便NLLLoss损失函数，如果用CrossEntropyLoss损失函数，可以不加log_softmax操作。

```{.python .input  n=218}
class Decoder(nn.Module):
    def __init__(self, output_dim, embedding_dim, hidden_dim, num_layers=2, dropout=0.2):
        super().__init__()

        self.embedding_dim = embedding_dim
        self.hid_dim = hidden_dim
        self.output_dim = output_dim
        self.num_layers = num_layers
        self.dropout = dropout

        self.embedding = nn.Embedding(output_dim, embedding_dim)

        self.attention = Attention(hidden_dim)

        self.rnn = nn.GRU(embedding_dim + hidden_dim, hidden_dim,
                          num_layers=num_layers, dropout=dropout)

        self.out = nn.Linear(embedding_dim + hidden_dim * 2, output_dim)

        self.dropout = nn.Dropout(dropout)

    def forward(self, input, hidden, encoder_outputs):
        # input = [bsz]
        # hidden = [n layers * n directions, batch size, hid dim]
        # encoder_outputs = [sent len, batch size, hid dim * n directions]

        input = input.unsqueeze(0)
        # input = [1, bsz]
        embedded = self.dropout(self.embedding(input))
        # embedded = [1, bsz, emb dim]

        attn_weight = self.attention(hidden, encoder_outputs)
        # (batch_size, seq_len)
        context = attn_weight.unsqueeze(1).bmm(encoder_outputs.transpose(0, 1)).transpose(0, 1)
        # (batch_size, 1, hidden_dim * n_directions)
        # (1, batch_size, hidden_dim * n_directions)
        emb_con = torch.cat((embedded, context), dim=2)

        # emb_con = [1, bsz, emb dim + hid dim]

        _, hidden = self.rnn(emb_con, hidden)

        # outputs = [sent len, batch size, hid dim * n directions]
        # hidden = [n layers * n directions, batch size, hid dim]

        output = torch.cat((embedded.squeeze(0), hidden[-1], context.squeeze(0)), dim=1)
        output = F.log_softmax(self.out(output), 1)
        # outputs = [sent len, batch size, vocab_size]
        return output, hidden, attn_weight
```

我们再定义一个Seq2seq类，将encoder和decoder结合起来，通过一个循环，模型对每一个batch从前往后依次生成序列，训练的时候可以使用teacher_forcing随机使用真实词或是模型输出的词作为target，测试的时候就不需要了。

```{.python .input  n=227}
class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder, device, teacher_forcing_ratio=0.5):
        super().__init__()

        self.encoder = encoder
        self.decoder = decoder
        self.device = device
        self.teacher_forcing_ratio = teacher_forcing_ratio

    def forward(self, src_seqs, src_lengths, trg_seqs):
        # src_seqs = [sent len, batch size]
        # trg_seqs = [sent len, batch size]
        batch_size = src_seqs.shape[1]
        max_len = trg_seqs.shape[0]
        trg_vocab_size = self.decoder.output_dim

        # tensor to store decoder outputs
        outputs = torch.zeros(max_len, batch_size, trg_vocab_size).to(self.device)

        # hidden used as the initial hidden state of the decoder
        # encoder_outputs used to compute context
        encoder_outputs, hidden = self.encoder(src_seqs, src_lengths)

        # first input to the decoder is the <sos> tokens
        output = trg_seqs[0, :]

        for t in range(1, max_len): # skip sos
            output, hidden, _ = self.decoder(output, hidden, encoder_outputs)
            outputs[t] = output
            teacher_force = random.random() < self.teacher_forcing_ratio
            output = (trg_seqs[t] if teacher_force else output.max(1)[1])

        return outputs
    
    def predict(self, src_seqs, src_lengths, max_trg_len=20, start_ix=1):
        max_src_len = src_seqs.shape[0]
        batch_size = src_seqs.shape[1]
        trg_vocab_size = self.decoder.output_dim
        outputs = torch.zeros(max_trg_len, batch_size, trg_vocab_size).to(self.device)
        encoder_outputs, hidden = self.encoder(src_seqs, src_lengths)
        output = torch.LongTensor([start_ix] * batch_size).to(self.device)
        attn_weights = torch.zeros((max_trg_len, batch_size, max_src_len))
        for t in range(1, max_trg_len):
            output, hidden, attn_weight = self.decoder(output, hidden, encoder_outputs)
            outputs[t] = output
            output = output.max(1)[1]
            attn_weights[t] = attn_weight
        return outputs, attn_weights
```

## 模型训练
---
这里直接取1000个batch进行更新。

```{.python .input  n=249}
embedding_dim = 100
hidden_dim = 100
batch_size = 256
clip = 5

encoder = Encoder(len(src_c2ix) + 1, embedding_dim, hidden_dim)
decoder = Decoder(len(trg_c2ix) + 1, embedding_dim, hidden_dim)
model = Seq2Seq(encoder, decoder, device).to(device)

optimizer = optim.Adam(model.parameters())
criterion = nn.NLLLoss(ignore_index=0).to(device)

model.train()
for batch_id in range(1, 1001):
    src_seqs, src_lengths, trg_seqs, _ = random_batch(batch_size, pairs, src_c2ix, trg_c2ix)
    
    optimizer.zero_grad()
    output = model(src_seqs, src_lengths, trg_seqs)
    loss = criterion(output.view(-1, output.shape[2]), trg_seqs.view(-1))
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
    optimizer.step()
    
    if batch_id % 100 == 0:
        print('current loss: {:.4f}'.format(loss))
```

```{.json .output n=249}
[
 {
  "name": "stdout",
  "output_type": "stream",
  "text": "current loss: 0.8295\ncurrent loss: 0.3438\ncurrent loss: 0.1844\ncurrent loss: 0.0970\ncurrent loss: 0.0738\ncurrent loss: 0.0460\ncurrent loss: 0.0272\ncurrent loss: 0.0170\ncurrent loss: 0.0124\ncurrent loss: 0.0094\n"
 }
]
```

## 模型测试
---
在进行测试时，生成的句子不超过最大目标句子的长度，同时要保存生成的每个词对原端每个词的attention权重，以便可视化。生成时不超过最大长度，如果下一个词是终止符，则退出循环，生成结束。

```{.python .input  n=233}
def show_attention(input_words, output_words, attentions):
    # Set up figure with colorbar
    fig = plt.figure()
    ax = fig.add_subplot(111)
    cax = ax.matshow(attentions, cmap='bone')
    fig.colorbar(cax)

    # Set up axes
    ax.set_xticklabels([''] + input_words)
    ax.set_yticklabels([''] + output_words)

    # Show label at every tick
    ax.xaxis.set_major_locator(ticker.MultipleLocator())
    ax.yaxis.set_major_locator(ticker.MultipleLocator())

    plt.show()
    plt.close()

def evaluate(model, text, src_c2ix, trg_ix2c):
    model.eval()
    with torch.no_grad():
        seq = torch.LongTensor(indexes_from_text(text, src_c2ix)).view(-1, 1).to(device)
        outputs, attn_weights = model.predict(seq, [seq.size(0)], max_trg_len)
        outputs = outputs.squeeze(1).cpu().numpy()
        attn_weights = attn_weights.squeeze(1).cpu().numpy()
        output_words = [trg_ix2c[np.argmax(word_prob)] for word_prob in outputs]
        show_attention(list('^' + text + '$'), output_words, attn_weights)
```

下面是我随便写的一个日期，可以看出attention的效果还是有的。

```{.python .input  n=291}
text = 'thirsty 1 before 3 clock afternon'
evaluate(model, text, src_c2ix, trg_ix2c)
```

```{.json .output n=291}
[
 {
  "data": {
   "image/png": "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\nGrW6AAAAAElFTkSuQmCC\n",
   "text/plain": "<matplotlib.figure.Figure at 0x7f5be10f7240>"
  },
  "metadata": {},
  "output_type": "display_data"
 }
]
```
