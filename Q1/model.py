import torch
import math
import torch.nn as nn


class FNNModel(nn.Module):
    """Container module with an encoder, a dense layer, and a decoder."""
    # ntoken - number of tokens
    # ninp - embedding size default: 200
    # nhid - hidden units per layer
    # nlayers (self explanatory) = 1
    def __init__(self, seq_len, ntoken, ninp, nhid, dropout=0.5, tie_weights=False):
        super(FNNModel, self).__init__()
        self.seq_len = seq_len
        self.ntoken = ntoken
        self.drop = nn.Dropout(dropout)
        self.encoder = nn.Embedding(ntoken, ninp)
        self.hidden_tanh = nn.Linear(ninp*seq_len, nhid)
        self.decoder = nn.Linear(nhid, ntoken)

        # Optionally tie weights as in:
        # "Using the Output Embedding to Improve Language Models" (Press & Wolf 2016)
        # https://arxiv.org/abs/1608.05859
        # and
        # "Tying Word Vectors and Word Classifiers: A Loss Framework for Language Modeling" (Inan et al. 2016)
        # https://arxiv.org/abs/1611.01462
        if tie_weights:
            if nhid != ninp:
                raise ValueError('When using the tied flag, nhid must be equal to emsize')
            self.decoder.weight = self.encoder.weight

        self.init_weights()

        self.nhid = nhid


    def forward(self, input):
        #transpose input
        input = torch.transpose(input,0,1)
        emb = self.drop(self.encoder(input))
        emb = torch.flatten(emb,start_dim=1,end_dim=2)
        output = nn.Tanh()(self.hidden_tanh(emb))
        output = self.drop(output)
        decoded = self.decoder(output)
        decoded = decoded.view(-1, self.ntoken)
        return decoded
        #return F.log_softmax(decoded, dim=1)

    def init_weights(self):
        initrange = 0.1
        nn.init.uniform_(self.encoder.weight, -initrange, initrange)
        nn.init.uniform_(self.hidden_tanh.weight, -initrange, initrange)
        nn.init.zeros_(self.decoder.weight)
        nn.init.uniform_(self.decoder.weight, -initrange, initrange)



class FNNAttenModel(nn.Module):
    """Container module with an encoder, a dense layer, and a decoder."""
    # ntoken - number of tokens
    # ninp - embedding size default: 200
    # nhid - hidden units per layer
    # nlayers (self explanatory) = 1
    def __init__(self, seq_len, ntoken, ninp, nhid, dropout=0.5, tie_weights=False, nheads=8):
        assert ninp % nheads == 0

        super(FNNAttenModel, self).__init__()
        self.seq_len = seq_len
        self.ntoken = ntoken
        self.drop = nn.Dropout(dropout)
        self.encoder = nn.Embedding(ntoken, ninp)

        self.head_dim = ninp // nheads
        self.hid_dim = ninp
        self.nheads = nheads
        self.fc_q = nn.Linear(ninp, ninp)
        self.fc_k = nn.Linear(ninp, ninp)
        self.fc_v = nn.Linear(ninp, ninp)
        self.atten_out = nn.Linear(ninp, ninp)

        self.scale = torch.sqrt(torch.FloatTensor([self.nheads])).cuda()

        self.hidden_tanh = nn.Linear(ninp*seq_len, nhid)
        self.decoder = nn.Linear(nhid, ntoken)

        # Optionally tie weights as in:
        # "Using the Output Embedding to Improve Language Models" (Press & Wolf 2016)
        # https://arxiv.org/abs/1608.05859
        # and
        # "Tying Word Vectors and Word Classifiers: A Loss Framework for Language Modeling" (Inan et al. 2016)
        # https://arxiv.org/abs/1611.01462
        if tie_weights:
            if nhid != ninp:
                raise ValueError('When using the tied flag, nhid must be equal to emsize')
            self.decoder.weight = self.encoder.weight

        self.init_weights()

        self.nhid = nhid


    def forward(self, input):
        #transpose input
        input = torch.transpose(input,0,1)
        emb = self.drop(self.encoder(input))

        batch_size = emb.shape[0]


        Q = self.fc_q(emb)
        K = self.fc_k(emb)
        V = self.fc_v(emb)

        Q = Q.view(batch_size, -1, self.nheads, self.head_dim).permute(0, 2, 1, 3)
        K = K.view(batch_size, -1, self.nheads, self.head_dim).permute(0, 2, 1, 3)
        V = V.view(batch_size, -1, self.nheads, self.head_dim).permute(0, 2, 1, 3)

        energy = torch.matmul(Q, K.permute(0, 1, 3, 2)) / self.scale

        attention = torch.softmax(energy, dim=-1)

        x = torch.matmul(self.drop(attention), V)
        x = x.permute(0, 2, 1, 3).contiguous()
        x = x.view(batch_size, -1, self.hid_dim)
        x = self.atten_out(x)




        emb = torch.flatten(x,start_dim=1,end_dim=2)
        output = nn.Tanh()(self.hidden_tanh(emb))
        output = self.drop(output)
        decoded = self.decoder(output)
        decoded = decoded.view(-1, self.ntoken)
        return decoded
        #return F.log_softmax(decoded, dim=1)

    def init_weights(self):
        initrange = 0.1
        nn.init.uniform_(self.encoder.weight, -initrange, initrange)
        nn.init.uniform_(self.hidden_tanh.weight, -initrange, initrange)

        nn.init.uniform_(self.fc_q.weight, -initrange, initrange)
        nn.init.uniform_(self.fc_k.weight, -initrange, initrange)
        nn.init.uniform_(self.fc_v.weight, -initrange, initrange)
        nn.init.uniform_(self.atten_out.weight, -initrange, initrange)

        nn.init.zeros_(self.decoder.weight)
        nn.init.uniform_(self.decoder.weight, -initrange, initrange)
