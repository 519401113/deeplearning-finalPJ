import torch
from torch import nn
from torch import functional as F
from torch import optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from tqdm import tqdm
import math
import matplotlib.pyplot as plt
import numpy as np
import os
from tensorboardX import SummaryWriter
from utils import cutout,mixup, mixup_criterion, cutmix, cutmix_criterion

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

class Attention(nn.Module):
    def __init__(self, embed_dim, heads=8, activation=None, dropout=0.1):
        super(Attention, self).__init__()
        self.heads = heads
        self.embed_dim = embed_dim
        self.query = nn.Linear(embed_dim, embed_dim)
        self.key = nn.Linear(embed_dim, embed_dim)
        self.value = nn.Linear(embed_dim, embed_dim)
        self.softmax = nn.Softmax(dim=-1)
        if activation == 'relu':
            self.activation = nn.ReLU()
        else:
            self.activation = nn.Identity()
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, inp):
        # inp: (batch_size, seq_len, embed_dim)
        batch_size, seq_len, embed_dim = inp.size()
        assert embed_dim == self.embed_dim

        query = self.activation(self.query(inp))
        key   = self.activation(self.key(inp))
        value = self.activation(self.value(inp))

        # output of _reshape_heads(): (batch_size * heads, seq_len, reduced_dim) | reduced_dim = embed_dim // heads
        query = self._reshape_heads(query)
        key   = self._reshape_heads(key)
        value = self._reshape_heads(value)

        # attention_scores: (batch_size * heads, seq_len, seq_len) | Softmaxed along the last dimension
        attention_scores = self.softmax(torch.matmul(query, key.transpose(1, 2)))

        # out: (batch_size * heads, seq_len, reduced_dim)
        out = torch.matmul(self.dropout(attention_scores), value)
        
        # output of _reshape_heads_back(): (batch_size, seq_len, embed_size)
        out = self._reshape_heads_back(out)

        return out, attention_scores

    def _reshape_heads(self, inp):
        # inp: (batch_size, seq_len, embed_dim)
        batch_size, seq_len, embed_dim = inp.size()

        reduced_dim = self.embed_dim // self.heads
        assert reduced_dim * self.heads == self.embed_dim
        out = inp.reshape(batch_size, seq_len, self.heads, reduced_dim)
        out = out.permute(0, 2, 1, 3)
        out = out.reshape(-1, seq_len, reduced_dim)

        # out: (batch_size * heads, seq_len, reduced_dim)
        return out

    def _reshape_heads_back(self, inp):
        # inp: (batch_size * heads, seq_len, reduced_dim) | reduced_dim = embed_dim // heads
        batch_size_mul_heads, seq_len, reduced_dim = inp.size()
        batch_size = batch_size_mul_heads // self.heads

        out = inp.reshape(batch_size, self.heads, seq_len, reduced_dim)
        out = out.permute(0, 2, 1, 3)
        out = out.reshape(batch_size, seq_len, self.embed_dim)

        # out: (batch_size, seq_len, embed_dim)
        return out
    
# Check if Dropout should be used after second Linear Layer
class FeedForward(nn.Module):
    def __init__(self, embed_dim, forward_expansion=1, dropout=0.1):
        super(FeedForward, self).__init__()
        self.embed_dim = embed_dim
        self.fc1 = nn.Linear(embed_dim, embed_dim * forward_expansion)
        self.activation = nn.GELU()
        self.fc2 = nn.Linear(embed_dim * forward_expansion, embed_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, inp):
        # inp: (batch_size, seq_len, embed_dim)
        batch_size, seq_len, embed_dim = inp.size()
        assert embed_dim == self.embed_dim

        out = self.dropout(self.activation(self.fc1(inp)))
        # out = self.dropout(self.fc2(out))
        out = self.fc2(out)

        # out: (batch_size, seq_len, embed_dim)
        return out 

class TransformerBlock(nn.Module):
    def __init__(self, embed_dim, heads=8, activation=None, forward_expansion=1, dropout=0.1):
        super(TransformerBlock, self).__init__()
        self.embed_dim = embed_dim
        self.norm1 = nn.LayerNorm(embed_dim)
        self.attention = Attention(embed_dim, heads, activation, dropout)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.feed_forward = FeedForward(embed_dim, forward_expansion, dropout)

    def forward(self, inp):
        # inp: (batch_size, seq_len, embed_dim)
        batch_size, seq_len, embed_dim = inp.size()
        assert embed_dim == self.embed_dim

        res = inp
        out = self.norm1(inp)
        out, _ = self.attention(out)
        out = out + res
        
        res = out
        out = self.norm2(out)
        out = self.feed_forward(out)
        out = out + res

        # out: (batch_size, seq_len, embed_dim)
        return out
    
class Transformer(nn.Module):
    def __init__(self, embed_dim, layers, heads=8, activation=None, forward_expansion=1, dropout=0.1):
        super(Transformer, self).__init__()
        self.embed_dim = embed_dim
        self.trans_blocks = nn.ModuleList(
            [TransformerBlock(embed_dim, heads, activation, forward_expansion, dropout) for i in range(layers)]
        )

    def forward(self, inp):
        # inp: (batch_size, seq_len, embed_dim)

        out = inp
        for block in self.trans_blocks:
            out = block(out)

        # out: (batch_size, seq_len, embed_dim)
        return out


# Not Exactly Same as Paper
class ClassificationHead(nn.Module):
    '''
    Classification Head attached to the first sequence token which is used as the arbitrary 
    classification token and used to optimize the transformer model by applying Cross-Entropy 
    loss. The sequence of operations is as follows :-

    Input -> FC1 -> GELU -> Dropout -> FC2 -> Output

    Args:
        embed_dim: Dimension size of the hidden embedding
        classes: Number of classification classes in the dataset
        dropout: Dropout value for the layer on attention_scores (Default=0.1)

    Methods:
        forward(inp) :-
        Applies the sequence of operations mentioned above.
        (batch_size, embed_dim) -> (batch_size, classes)

    Examples:
        >>> CH = ClassificationHead(embed_dim, classes, dropout)
        >>> out = CH(inp)
    '''
    def __init__(self, embed_dim, classes, dropout=0.1):
        super(ClassificationHead, self).__init__()
        self.embed_dim = embed_dim
        self.classes = classes
        self.fc1 = nn.Linear(embed_dim, embed_dim // 2)
        self.activation = nn.GELU()
        self.fc2 = nn.Linear(embed_dim // 2, classes)
        self.softmax = nn.Softmax(dim=-1)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, inp):
        # inp: (batch_size, embed_dim)
        batch_size, embed_dim = inp.size()
        assert embed_dim == self.embed_dim

        out = self.dropout(self.activation(self.fc1(inp)))
        # out = self.softmax(self.fc2(out))
        out = self.fc2(out)

        # out: (batch_size, classes) 
        return out

class VisionTransformer(nn.Module):
    def __init__(self, patch_size, max_len, embed_dim, classes, layers, channels=3, heads=8, activation=None, forward_expansion=1, dropout=0.1):
        super(VisionTransformer, self).__init__()
        self.name = 'VisionTransformer'
        self.patch_size = patch_size
        self.embed_dim = embed_dim
        self.channels = channels
        self.patch_to_embed = nn.Linear(patch_size * patch_size * channels, embed_dim)
        self.position_embed = nn.Parameter(torch.randn((max_len, embed_dim)))
        self.transformer = Transformer(embed_dim, layers, heads, activation, forward_expansion, dropout)
        self.classification_head = ClassificationHead(embed_dim, classes)
        self.class_token = nn.Parameter(torch.zeros(1, 1, embed_dim))

    def forward(self, inp):
        # inp: (batch_size, channels, width, height)
        batch_size, channels, width, height = inp.size()
        assert channels == self.channels

        out = inp.unfold(2, self.patch_size, self.patch_size).unfold(3, self.patch_size, self.patch_size).contiguous()
        out = out.view(batch_size, channels, -1, self.patch_size, self.patch_size)
        out = out.permute(0, 2, 3, 4, 1)
        # out: (batch_size, seq_len, patch_size, patch_size, channels) | seq_len would be (width*height)/(patch_size**2)
        batch_size, seq_len, patch_size, _, channels = out.size()
        
        out = out.reshape(batch_size, seq_len, -1)
        out = self.patch_to_embed(out)
        # out: (batch_size, seq_len, embed_dim)

        class_token = self.class_token.expand(batch_size, -1, -1)
        out = torch.cat([class_token, out], dim=1)
        # out: (batch_size, seq_len+1, embed_dim)

        position_embed = self.position_embed[:seq_len+1]
        position_embed = position_embed.unsqueeze(0).expand(batch_size, seq_len+1, self.embed_dim)
        out = out + position_embed
        # out: (batch_size, seq_len+1, embed_dim) | Added Positional Embeddings

        out = self.transformer(out)
        # out: (batch_size, seq_len+1, embed_dim) 
        class_token = out[:, 0]
        # class_token: (batch_size, embed_dim)

        class_out = self.classification_head(class_token)
        # class_out: (batch_size, classes)
        
        return class_out#, out

# Initializations of all the constants used in the training and testing process


# 10756452
def train(args,model, dataloader, criterion, optimizer):
    running_loss = 0.0
    running_accuracy = 0.0
    total_loss = 0.0
    num = 0
    acc_num = 0

    for inputs, targets in tqdm(dataloader):

        if args.mixup:
            r = np.random.rand()
            if r < args.mixup_prob:
                inputs, targets, targets1, lam = mixup(inputs, targets,beta=args.beta)
                inputs = inputs.cuda()
                targets = targets.cuda()
                targets1 = targets.cuda()
                # forward
                output = model(inputs)
                loss = mixup_criterion(criterion,output, targets, targets1, lam)
            else:
                inputs = inputs.cuda()
                targets = targets.cuda()

                # forward
                output = model(inputs)
                loss = criterion(output, targets)
        elif args.cutmix:
            r = np.random.rand()
            if r < args.cutmix_prob:
                inputs, targets, targets1, lam = cutmix(inputs, targets,beta=args.beta)
                inputs = inputs.cuda()
                targets = targets.cuda()
                targets1 = targets.cuda()
                # forward
                output = model(inputs)
                loss = cutmix_criterion(criterion,output, targets, targets1, lam)
            else:
                inputs = inputs.cuda()
                targets = targets.cuda()

                # forward
                output = model(inputs)
                loss = criterion(output, targets)
        elif args.cutout:
            r = np.random.rand()
            if r < args.cutout_prob:
                inputs, targets = cutout(inputs, targets)

            inputs = inputs.cuda()
            targets = targets.cuda()

            # forward
            output = model(inputs)
            loss = criterion(output, targets)
        else:
            inputs = inputs.cuda()
            targets = targets.cuda()

            # forward
            output = model(inputs)
            loss = criterion(output, targets)


        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        # scheduler.step()

        # acc = (output.argmax(dim=1) == target).float().mean()
        # running_accuracy += acc / len(dataloader)
        running_loss += loss.item() / len(dataloader)
        acc_num += (output.argmax(dim=1) == targets).float().sum()
        num += inputs.size(0)
        #total_loss += loss.item()
    
    running_accuracy = acc_num/num
    #running_loss = total_loss/num
    return running_loss, running_accuracy

def test(model, dataloader, criterion):
    with torch.no_grad():
        test_accuracy = 0.0
        test_loss = 0.0
        total_loss = 0.0
        acc_num = 0
        num = 0
        for inputs, targets in tqdm(dataloader):
            inputs = inputs.to(device)
            targets = targets.to(device)

            output = model(inputs)

            loss = criterion(output, targets)

            # acc = (output.argmax(dim=1) == target).float().mean()
            # test_accuracy += acc / len(dataloader)
            test_loss += loss.item()/len(dataloader)
            
            acc_num += (output.argmax(dim=1) == targets).float().sum()
            num += inputs.size(0)
            
        test_accuracy = acc_num/num
            
    return test_loss, test_accuracy

def CIFAR100DataLoader(split, batch_size=8, num_workers=2, shuffle=True, normalize='standard'):
    cifar_data_path = '/remote-home/mfdu/nlpbeginner/hw2'
    if normalize == 'imagenet':
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]
    elif normalize == 'standard':
        mean = [0.5, 0.5, 0.5]
        std =  [0.5, 0.5, 0.5]

    if split == 'train':
        train_transform = transforms.Compose([
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.RandomRotation(15),
                transforms.ToTensor(),
                transforms.Normalize(mean, std)
            ])
        
        cifar100 = torchvision.datasets.CIFAR100(root=cifar_data_path, train=True, download=True, transform=train_transform)
        dataloader = DataLoader(cifar100, batch_size=batch_size, num_workers=num_workers, shuffle=shuffle)
    
    elif split == 'test':
        test_transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(mean, std)
            ])

        cifar100 = torchvision.datasets.CIFAR100(root=cifar_data_path, train=False, download=True, transform=test_transform)
        dataloader = DataLoader(cifar100, batch_size=batch_size, num_workers=num_workers, shuffle=shuffle)

    return dataloader

def write_to_record_file(data, file_path, verbose=True):
    if verbose:
        print(data)
    record_file = open(file_path, 'a')
    record_file.write(data+'\n')
    record_file.close()

def adjust_learning_rate(optimizer, epoch, initial_lr, num_epochs):
    # lr = initial_lr * (0.1 ** (epoch // (num_epochs * 0.5))) * (0.1 ** (epoch // (num_epochs * 0.75)))
    #lr = initial_lr * lr_decay ** int(epoch/decay_step)
    # lr = initial_lr * 1.0/ (1.0 + lr_decay*epoch)
    """decrease the learning rate at 50 and 100 epoch"""
    lr = initial_lr
    if epoch >= 50:
        lr /= 10
    if epoch >= 100:
        lr /= 10
    if epoch >= 150:
        lr /= 10
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--name', type=str, default='vit_base_1', help='net type')
    parser.add_argument('--batch_size', type=int, default=32, help='batch size for dataloader')
    parser.add_argument('--lr', type=float, default=0.003, help='initial learning rate')
    parser.add_argument('--resume',default=False, help='resume training')
    parser.add_argument('--num_epochs', default=200, help='num epoch')
    parser.add_argument('--ckpt_dir', type=str,default=None, help='ckpts')
    parser.add_argument('--logs_dir', type=str,default=None, help='logs')
    parser.add_argument('--record_file', type=str,default=None, help='record')
    parser.add_argument('--mixup', default=False,action='store_true')
    parser.add_argument('--mixup_prob', type=float,default=0.5)
    parser.add_argument('--cutmix', default=False,action='store_true')
    parser.add_argument('--cutmix_prob', type=float,default=0.1)
    parser.add_argument('--cutout', default=False,action='store_true')
    parser.add_argument('--cutout_prob', type=float,default=0.5)
    parser.add_argument('--beta', type=float,default=1.0)
    parser.add_argument('--num_workers', type=int,default=2)
    
    # vision transformer
    parser.add_argument('--patch_size', type=int,default=2)
    parser.add_argument('--max_len', type=int,default=257)
    parser.add_argument('--embed_dim', type=int,default=512)
    parser.add_argument('--classes', type=int,default=100)        
    parser.add_argument('--layers', type=int,default=8)
    parser.add_argument('--channels', type=int,default=3)
    parser.add_argument('--heads', type=int,default=16)

    
    parser.add_argument('--model_name', type=str,default='vit',choices=['vit','resnet'])

    args = parser.parse_args()
    
    # para
    args.ckpt_dir = os.path.join(args.name, 'ckpts')
    args.logs_dir = os.path.join(args.name, 'logs')
    os.makedirs(args.name, exist_ok=True)
    os.makedirs(args.ckpt_dir, exist_ok=True)
    os.makedirs(args.logs_dir, exist_ok=True)
    args.record_file = os.path.join(args.logs_dir, 'train.txt')
    

    # Vision Transformer Architecture
    writer = SummaryWriter(log_dir=args.logs_dir)

    model = VisionTransformer(
        patch_size=args.patch_size,
        max_len=args.max_len,
        embed_dim=args.embed_dim,
        classes=args.classes,
        layers=args.layers,
        channels=args.channels,
        heads=args.heads).to(device)


    train_dataloader = CIFAR100DataLoader(split='train', batch_size=args.batch_size, num_workers=args.num_workers, shuffle=True, normalize='standard')
    test_dataloader = CIFAR100DataLoader(split='test', batch_size=args.batch_size, num_workers=args.num_workers, shuffle=False, normalize='standard')

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)


    best_acc = 0
    best_epoch = 0
    best_state = 0
    for epoch in range(args.num_epochs):
        adjust_learning_rate(optimizer, epoch, args.lr, args.num_epochs)
        running_loss, running_accuracy = train(args, model, train_dataloader, criterion, optimizer)
        log = f"Epoch : {epoch+1} - train acc: {running_accuracy:.4f} - loss : {running_loss:.4f}\n"

        test_loss, test_accuracy = test(model, test_dataloader, criterion)
        log += f"test acc: {test_accuracy:.4f} - test loss : {test_loss:.4f}\n"

        
        writer.add_scalar("train_loss", running_loss, epoch)    
        writer.add_scalar("test_loss", test_loss, epoch)    
        writer.add_scalar("test_acc", test_accuracy, epoch)    

        is_best = test_accuracy > best_acc
        best_acc = max(test_accuracy, best_acc)
        if is_best:
            best_epoch = epoch
            best_state = model.state_dict()
        write_to_record_file(log,args.record_file)
    log = f"best epoch: {best_epoch:d} - best acc : {best_acc:.4f}\n"
    write_to_record_file(log,args.record_file)
    torch.save({'epoch':best_epoch,'state_dict':best_state,'acc': best_acc,}, os.path.join(args.ckpt_dir, 'best_test')) 

if __name__ == '__main__':
    main()


# sum([param.nelement() for param in model.parameters()])
# 9039652
# 10756452
# model = ResNet18()
# sum([param.nelement() for param in model.parameters()])
# 11220132
