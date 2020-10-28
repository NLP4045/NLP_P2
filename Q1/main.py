# coding: utf-8
import argparse
import time
import math
import os
import torch
import torch.onnx
import torch.optim as optim
import torch.nn.functional as F
from torch import nn

import matplotlib.pyplot as plt


import model, data

datapath = (os.path.join(os.getcwd(), "data", "wikitext-2"))

parser = argparse.ArgumentParser(description='PyTorch Wikitext-2 RNN/LSTM/GRU/Transformer Language Model')
parser.add_argument('--data', type=str, default=datapath,
                    help='location of the data corpus')
parser.add_argument('--model', type=str, default='ffn',
                    help='type of recurrent net (RNN_TANH, RNN_RELU, LSTM, GRU, Transformer)')
parser.add_argument('--emsize', type=int, default=200,
                    help='size of word embeddings')
parser.add_argument('--nhid', type=int, default=200,
                    help='number of hidden units per layer')
parser.add_argument('--nlayers', type=int, default=2,
                    help='number of layers')
parser.add_argument('--lr', type=float, default=0.0001,
                    help='initial learning rate')
parser.add_argument('--clip', type=float, default=0.25,
                    help='gradient clipping')
parser.add_argument('--epochs', type=int, default=50,
                    help='upper epoch limit')
parser.add_argument('--batch_size', type=int, default=256, metavar='N',
                    help='batch size')
parser.add_argument('--bptt', type=int, default=7,
                    help='sequence length')
parser.add_argument('--dropout', type=float, default=0.2,
                    help='dropout applied to layers (0 = no dropout)')
parser.add_argument('--l2factor', type=float, default=0,
                    help='L2 regularization factor (0 = no regularization applied)')
parser.add_argument('--tied', action='store_true',
                    help='tie the word embedding and softmax weights')
parser.add_argument('--seed', type=int, default=1111,
                    help='random seed')
parser.add_argument('--cuda', action='store_true', default=True,
                    help='use CUDA')
parser.add_argument('--log-interval', type=int, default=200, metavar='N',
                    help='report interval')
parser.add_argument('--save', type=str, default='model.pt',
                    help='path to save the final model')
parser.add_argument('--onnx-export', type=str, default='',
                    help='path to export the final model in onnx format')

parser.add_argument('--figname', type=str, default='',
                    help='Figure name')

parser.add_argument('--nhead', type=int, default=2,
                    help='the number of heads in the encoder/decoder of the transformer model')
parser.add_argument('--dry-run', action='store_true',
                    help='verify the code and the model')
parser.add_argument('--optim', type=str, default="adam",
                    help='optimizer to use (sgd, adam, rmsprop)')
parser.add_argument('--patience', type=int, default=None,
                    help='Patience (Epochs to wait) for early stopping')

args = parser.parse_args()

if args.optim == "sgd":
    selOptim = optim.SGD
elif args.optim == "adam":
    selOptim = optim.Adam
elif args.optim == "rmsprop":
    selOptim = optim.RMSprop
elif args.optim == "adagrad":
    selOptim = optim.Adagrad
else:
    print("optimizer must be either sgd, adam, adagrad or rmsprop")
    exit()

# Set the random seed manually for reproducibility.
torch.manual_seed(args.seed)
if torch.cuda.is_available():
    if not args.cuda:
        print("WARNING: You have a CUDA device, so you should probably run with --cuda")

device = torch.device("cuda" if args.cuda else "cpu")

###############################################################################
# Load data
###############################################################################

corpus = data.Corpus(args.data)

# Starting from sequential data, batchify arranges the dataset into columns.
# For instance, with the alphabet as the sequence and batch size 4, we'd get
# ┌ a g m s ┐
# │ b h n t │
# │ c i o u │
# │ d j p v │
# │ e k q w │
# └ f l r x ┘.
# These columns are treated as independent by the model, which means that the
# dependence of e. g. 'g' on 'f' can not be learned, but allows more efficient
# batch processing.

def batchify(data, bsz):
    # Work out how cleanly we can divide the dataset into bsz parts.
    nbatch = data.size(0) // (bsz * (args.bptt+1))
    # Trim off any extra elements that wouldn't cleanly fit (remainders).
    data = data.narrow(0, 0, nbatch * (bsz * (args.bptt+1)))

    # Evenly divide the data across the bsz batches.
    data = data.view(bsz, -1).t().contiguous()
    return data.to(device)

eval_batch_size = 10
train_data = batchify(corpus.train, args.batch_size)
val_data = batchify(corpus.valid, eval_batch_size)
test_data = batchify(corpus.test, eval_batch_size)

###############################################################################
# Build the model
###############################################################################

ntokens = len(corpus.dictionary)

# Test: set hidden to 3* ntokens
model = model.FNNModel(args.bptt, ntokens, args.emsize, args.nhid, args.dropout).to(device)

#criterion = nn.NLLLoss()
criterion = F.cross_entropy
###############################################################################
# Training code
###############################################################################



# get_batch subdivides the source data into chunks of length args.bptt.
# If source is equal to the example output of the batchify function, with
# a bptt-limit of 2, we'd get the following two Variables for i = 0:
# ┌ a g m s ┐ ┌ b h n t ┐
# └ b h n t ┘ └ c i o u ┘
# Note that despite the name of the function, the subdivison of data is not
# done along the batch dimension (i.e. dimension 1), since that was handled
# by the batchify function. The chunks are along dimension 0, corresponding
# to the seq_len dimension in the LSTM.

# def get_batch(source, i):
#     seq_len = min(args.bptt, len(source) - 1 - i)
#     data = source[i:i+seq_len]
#     target = source[i+1:i+1+seq_len].view(-1)
#     return data, target

def get_batch(source, i):
    seq_len = min(args.bptt, len(source) - 1 - i)
    data = source[i:i+seq_len]
    target = source[i+seq_len].view(-1)
    return data, target

def evaluate(data_source):
    # Turn on evaluation mode which disables dropout.
    model.eval()
    total_loss = 0.
    length = 0
    total = 0
    correct = 0
    with torch.no_grad():
        for i in range(0, data_source.size(0) - args.bptt):
            data, targets = get_batch(data_source, i)
            if data.size(0) != args.bptt:
                break
            output = model(data)
            total_loss += len(data) * criterion(output, targets).item()

            _, predicted = torch.max(output.data, 1)
            total += targets.size(0)
            length += data.size(0)
            correct += (predicted == targets).sum().item()
    return (total_loss / length), (correct/total)


def train():
    # Turn on training mode which enables dropout.
    optimizer = selOptim(model.parameters(), lr=args.lr, weight_decay=args.l2factor)
    model.train()
    total_loss = 0.
    start_time = time.time()

    for batch, i in enumerate(range(0, train_data.size(0) - args.bptt)):
        data, targets = get_batch(train_data, i)
        # Starting each batch, we detach the hidden state from how it was previously produced.
        # If we didn't, the model would try backpropagating all the way to start of the dataset.
        if data.size(0) != args.bptt:
            break
        model.zero_grad()
        output = model(data)
        loss = criterion(output, targets)

        loss.backward()
        optimizer.step()

        total_loss += loss.item()

        if batch % args.log_interval == 0 and batch > 0:
            cur_loss = total_loss / args.log_interval
            elapsed = time.time() - start_time
            print('| epoch {:3d} | {:5d}/{:5d} batches | lr {:02.5f} | ms/batch {:5.2f} | '
                    'loss {:5.2f} | ppl {:8.2f}'.format(
                epoch, batch, len(train_data) - args.bptt, lr,
                elapsed * 1000 / args.log_interval, cur_loss, math.exp(cur_loss)))
            total_loss = 0
            start_time = time.time()
        if args.dry_run:
            break


def export_onnx(path, batch_size, seq_len):
    print('The model is also exported in ONNX format at {}'.
          format(os.path.realpath(args.onnx_export)))
    model.eval()
    dummy_input = torch.LongTensor(seq_len * batch_size).zero_().view(-1, batch_size).to(device)
    hidden = model.init_hidden(batch_size)
    torch.onnx.export(model, (dummy_input, hidden), path)


# Loop over epochs.
lr = args.lr
best_val_loss = None

# At any point you can hit Ctrl + C to break out of training early.
no_improvement_counts = 0
try:
    train_losses = []
    valid_losses = []
    train_accs = []
    valid_accs = []
    for epoch in range(1, args.epochs+1):
        epoch_start_time = time.time()
        train()
        train_loss, train_acc = evaluate(train_data)
        val_loss, val_acc = evaluate(val_data)
        train_losses.append(train_loss)
        valid_losses.append(val_loss)
        train_accs.append(train_acc)
        valid_accs.append(val_acc)
        print('-' * 89)
        print('| end of epoch {:3d} | time: {:5.2f}s | train loss {:5.2f} | valid loss {:5.2f} | train acc {:5.4f} | valid acc {:5.4f} | '
                'train ppl {:8.2f} | valid ppl {:8.2f}'.format(epoch, (time.time() - epoch_start_time), train_loss,
                                           val_loss, train_acc, val_acc, math.exp(train_loss), math.exp(val_loss)))
        print('-' * 89)
        # Save the model if the validation loss is the best we've seen so far.
        if not best_val_loss or val_loss < best_val_loss:
            with open(args.save, 'wb') as f:
                torch.save(model, f)
            best_val_loss = val_loss
            no_improvement_counts=0
        else:
            # Anneal the learning rate if no improvement has been seen in the validation dataset.
            lr /= 4.0
            no_improvement_counts += 1
            if args.patience is not None and no_improvement_counts == args.patience:
                print("No improvements to model for {} Epoch, hence early stopping training at Epoch: {}".format(args.patience, epoch))
                break

    best_idx = valid_losses.index(best_val_loss)
    best_train_loss = train_losses[best_idx]
    best_train_acc = train_accs[best_idx]
    best_val_acc = valid_accs[best_idx]
    print('Best run @ epoch {}| train loss {:5.2f} | valid loss {:5.2f} | train acc {:5.2f} | valid acc {:5.2f} | '
          'train ppl {:8.2f} | valid ppl {:8.2f} '.format(best_idx+1, best_train_loss, best_val_loss,
                                                          best_train_acc, best_val_acc,
                                                          math.exp(best_train_loss), math.exp(best_val_loss)))

    if args.figname != '':
        fig = plt.figure()
        plt.plot(train_losses, label='loss on training set')
        plt.plot(valid_losses, label='loss on validation set')
        plt.ylabel('Loss (Cross-entropy)')
        plt.xlabel('No. epoch')
        plt.legend()
        plt.savefig(args.figname + "_loss.png")

        fig = plt.figure()
        plt.plot(train_accs, label='Accuracy on training set')
        plt.plot(valid_accs, label='Accuracy validation set')
        plt.ylabel('Prediction Accuracy')
        plt.xlabel('No. epoch')
        plt.legend()
        plt.savefig(args.figname + "acc.png")


except KeyboardInterrupt:
    print('-' * 89)
    print('Exiting from training early')

# Load the best saved model.
with open(args.save, 'rb') as f:
    model = torch.load(f)

# Run on test data.
test_loss, test_acc = evaluate(test_data)
print('=' * 89)
print('| End of training | test loss {:5.2f} | test ppl {:8.2f}'.format(
    test_loss, math.exp(test_loss)))
print('=' * 89)

if len(args.onnx_export) > 0:
    # Export the model in ONNX format.
    export_onnx(args.onnx_export, batch_size=1, seq_len=args.bptt)
