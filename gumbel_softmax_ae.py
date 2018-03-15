import numpy as np
from random import randint
#from modules import BatchTopK_attention
#from modules import SparseConnect
#from modules import KLDivLossGumbel
import modules as M
import argparse
import torch
import torchwordemb
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data
import shutil
from tqdm import tqdm

# Training settings
parser = argparse.ArgumentParser(description='PyTorch VAE gumbel Softmax word embeddings')
parser.add_argument('--batch_size', type=int, default=128, metavar='N',
                    help='batch size for training (default: 300)')
parser.add_argument('--test_batch_size', type=int, default=1000, metavar='N',
                    help='batch size for testing (default: 1000)')
parser.add_argument('--epochs', type=int, default=150, metavar='N',
                    help='number of epochs to train (default: 10)')
parser.add_argument('--lr', type=float, default=0.0001, metavar='LR',
                    help='learning rate (default: 0.0001)')
parser.add_argument('--no_cuda', action='store_true', default=False,
                    help='disables CUDA training')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--log_interval', type=int, default=2000, metavar='N',
                    help='how many batches to wait before logging training status')
parser.add_argument('--input_size', type=int, default=300, metavar='N',
                    help = 'number of dimensions of input size')
parser.add_argument('--latent_dim', type=int, default=2, metavar='N',
                    help = 'number of latent dimensions ')
parser.add_argument('--categorical_dim', type=int, default=1024, metavar='N',
                    help= 'number of clusters')
parser.add_argument('--intermediate_dim', type=int, default=1024, metavar='N',
                    help= 'number of dimensions in intermediate layer')
parser.add_argument('--matrix_sparsity', type=float, default=0.0, metavar='MS',
                    help='learning rate (default: 0.0)')
parser.add_argument('--path_word_vectors', type=str, default='/media/storage/glove_vectors/glove.6B.300d.txt', metavar='E',
                    help='path word embeddings')
parser.add_argument('--path_output_codes', type=str, default='/media/storage/word_vectors/codes_bin_40K_6B_', help='path output codes')
parser.add_argument('--path_output_reconstruction', type=str, default='/media/storage/word_vectors/dense_40K_6B_', help='path ouput vector reconstruction')
parser.add_argument('--version',type=str, default='version_std.txt',help='version')

args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()

def save_checkpoint(state, is_best, filename='/media/storage/word_vectors/checkpoint'+args.version+'.pth.tar'):
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, '/media/storage/word_vectors/model_best'+args.version+'.pth.tar')

torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)

path=args.path_word_vectors

vocab, vec = torchwordemb.load_glove_text(path)
word_vec_dataset = torch.utils.data.TensorDataset(vec, vec)
train_loader = torch.utils.data.DataLoader(word_vec_dataset,shuffle=True,batch_size=args.batch_size, num_workers=10)
test_loader = torch.utils.data.DataLoader(word_vec_dataset,shuffle=False,batch_size=args.test_batch_size, num_workers=10)
inverse_vocab = dict()
for k in vocab.keys():
    inverse_vocab[vocab[k]]=k

class Encoder(nn.Module):
    def __init__(self, input_size, intermediate_dim, categorical_dim, latent_dim, matrix_sparsity):
        super(Encoder, self).__init__()
        self.input_size = input_size
        self.intermediate_dim = intermediate_dim
        self.categorical_dim = categorical_dim
        self.latent_dim = latent_dim
        self.matrix_sparsity = matrix_sparsity
        self.encoder_proj = nn.Linear(self.input_size, self.intermediate_dim)
        self.alpha_proj = nn.Linear(self.intermediate_dim, self.categorical_dim*self.latent_dim)

    def forward(self, x, eps=1e-10):
        x = self.encoder_proj(x)
        #x = F.relu(x)
        x = F.tanh(x)
        x = self.alpha_proj(x)
        #alpha = self.alpha_proj(x)
        x = F.softplus(x)
        alpha = torch.log(x+eps)
        alpha = alpha.view(-1, self.categorical_dim, self.latent_dim)
        softmax_alpha = F.softmax(alpha, dim=-1)
        return alpha, softmax_alpha, F.log_softmax(alpha, dim=-1)


class Decoder(nn.Module):
    def __init__(self, categorical_dim, latent_dim, output_dim):
        super(Decoder, self).__init__()
        self.categorical_dim = categorical_dim
        self.latent_dim = latent_dim
        self.output_dim = output_dim
        self.decoder_proj = nn.Linear(self.categorical_dim * self.latent_dim, self.output_dim)

    def forward(self, x):
        x = self.decoder_proj(x)
        return x

class AE_gumbel(nn.Module):
    def __init__(self, input_size, intermediate_dim, categorical_dim, latent_dim, matrix_sparsity):
        super(AE_gumbel, self).__init__()
        self.input_size = input_size
        self.intermediate_dim = intermediate_dim
        self.categorical_dim = categorical_dim
        self.latent_dim = latent_dim
        self.matrix_sparsity = matrix_sparsity
        self.encoder = Encoder(self.input_size, self.intermediate_dim, self.categorical_dim, self.latent_dim, self.matrix_sparsity)
        self.decoder = Decoder(self.categorical_dim, self.latent_dim, self.input_size)

    def forward(self, x):
        logits, softmax_logits, log_softmax_logits = self.encoder(x)
        encoder_output= M.gumbel_softmax(logits.view(-1, self.categorical_dim,self.latent_dim), hard=False)
        #print(encoder_output)
        x_decoded = self.decoder(encoder_output)
        return x_decoded, softmax_logits, log_softmax_logits, encoder_output


model = AE_gumbel(args.input_size, args.intermediate_dim, args.categorical_dim, args.latent_dim, args.matrix_sparsity)

optimizer = optim.Adam(model.parameters(), lr=args.lr)

lossMSE = nn.MSELoss()
#lossKL = KLDivLossGumbel(args.categorical_dim, args.latent_dim, size_average=False)

if args.cuda:
    model.cuda()

minLossValue = float('inf')

def train(epoch):
    global minLossValue
    model.train()
    lossSum = 0.0
    is_best = False
    for batch_idx, (data, target) in enumerate(train_loader):
        if args.cuda:
            data, target = data.cuda(), target.cuda()
        data, target = Variable(data), Variable(target)
        x = data
        optimizer.zero_grad()
        x_decoded, output_encoder, log_output_encoder, _ = model(data)
        #loss_KL = lossKL(output_encoder, log_output_encoder)
        loss_MSE = args.input_size*lossMSE(x_decoded, x)
        loss_KL = loss_MSE
        # elbo loss
        loss = loss_MSE #- loss_KL
        loss.backward()
        optimizer.step()
        lossSum += loss.data[0]
        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}\tLoss_KL: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss_MSE.data[0], loss_KL.data[0]))
    # Saving best model
    if lossSum < minLossValue:
        minLossValue=lossSum
        is_best = True
        save_checkpoint({'epoch':epoch,'state_dict':model.state_dict(),'lossSum':lossSum, 'optimizer':optimizer.state_dict()},is_best)


for epoch in tqdm(range(1, args.epochs + 1)):
    train(epoch)

# Load best model
checkpoint = torch.load('/media/storage/word_vectors/model_best'+args.version+'.pth.tar')
model.load_state_dict(checkpoint['state_dict'])
model.eval()
word_codes=dict()
word_dense=dict()
count=0
for batch_idx, (data, target) in enumerate(test_loader):
    if args.cuda:
            data, target = data.cuda(), target.cuda()
    data, target = Variable(data), Variable(target)
    x = data
    _, _, _, output_encoder= model(data)
    hard_codes = M.hard_binarization(output_encoder, args.categorical_dim, args.latent_dim)
    dense_vec = model.decoder(hard_codes.view(-1, args.categorical_dim*args.latent_dim))
    batch_dense = dense_vec.data.cpu().numpy()
    batch_codes = hard_codes.view(-1, args.categorical_dim*args.latent_dim).data.cpu().numpy()
    for i in range(batch_codes.shape[0]):
        word_codes[count]=batch_codes[i]
        word_dense[count]=batch_dense[i]
        count+=1


def vector_for(w):
    w_i=vocab[w]
    return word_codes[w_i], word_dense[w_i]

codes_file_name = args.path_output_codes+args.version
dense_file_name = args.path_output_reconstruction+args.version
# Save codes in a file
i=0
f=open(codes_file_name, 'w')
g=open(dense_file_name, 'w')
for w in vocab:
    code, dense = vector_for(w)
    vec_str= " ".join(map(str, code))
    dense_str= " ".join(map(str, dense))
    f.write(w+' '+vec_str+'\n')
    g.write(w+' '+dense_str+'\n')
    if i%10000==0:
        print('i =',i)
    i=i+1
f.close()
g.close()
