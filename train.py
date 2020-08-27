# DXNET Training Script


from dx7pytorch import dxdataset as dxds
import argparse
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import random
import numpy as np

# Training settings
parser = argparse.ArgumentParser(description='DXnet Training Script.')
parser.add_argument('--dataset', type=str, metavar='PATH', help='Path to patch collection.')
parser.add_argument('--fs', type=int, default=8000, metavar='N', help='Sampling frequency of audio instances (default: 8kHz)')
#parser.add_argument('--note_on', type=int, default=16000, metavar='N', help='Note on instance length (default: 16000 samples)')
#parser.add_argument('--note_off', type=int, default=16000, metavar='N', help='Note off instance length (default: 16000 samples)')

parser.add_argument('--batch_size', type=int, default=8, metavar='N', help='input batch size for training (default: 8)')
#parser.add_argument('--test_batch_size', type=int, default=256, metavar='N', help='input batch size for testing (default: 256)')
parser.add_argument('--epochs', type=int, default=500, metavar='N', help='number of epochs to train (default: 500)')
parser.add_argument('--lr', type=float, default=0.001, metavar='LR', help='learning rate (default: 0.001)')

parser.add_argument('--no-cuda', action='store_true', default=False, help='disables CUDA training')
parser.add_argument('--seed', type=int, default=1234, metavar='S', help='random seed (default: 1234)')
parser.add_argument('--gpus', default=0, help='gpus used for training - e.g 0,1,3')
#parser.add_argument('--log_interval', type=int, default=1000, metavar='N', help='how many batches to wait before logging training status')
parser.add_argument('--resume', default=False, action='store_true', help='Perform only evaluation on val dataset.')
parser.add_argument('--eval', default=False, action='store_true', help='perform evaluation of trained model')
args = parser.parse_args()

args.cuda = not args.no_cuda and torch.cuda.is_available()

save_path='./results/dxnet.pt'
best_loss = float('inf')

'''
CNN Definition
'''

class DXNET_block(nn.Module):
    def __init__(self, in_channels, out_channels,k,s,p):
        super().__init__()
        
        modules = []
        modules.append(nn.Conv1d(in_channels, out_channels, kernel_size=k,stride = s, padding=p))
        modules.append(nn.BatchNorm1d(out_channels))
        modules.append(nn.ReLU(inplace=True))
    
        self.block = nn.Sequential(*modules)
    
    def forward(self, x):
        #print("In: {}".format(x.shape))
        fx = self.block(x)
        #print(fx.shape)
        return fx

class DXNET(nn.Module):
    def __init__(self, output_dim, block, pool):
        super().__init__()
        
        self.features = nn.Sequential(
            block(1,256,k=512,s=4,p=254), #in_channels, out_channels,kernel_size,stride,padding
            block(256,64,k=64,s=2,p=31),
            pool(2),
            block(64,64,k=64,s=2,p=31),
            block(64,128,k=64,s=2,p=31),
            pool(2),
            block(128,128,k=64,s=2,p=32),
            block(128,256,k=64,s=2,p=31)
        )
        
        self.classifier = nn.Linear(31*256, output_dim) # For 8kHz
        #self.classifier = nn.Linear(16128, output_dim) # For 16kHz
        self.output_fn = torch.nn.Sigmoid()

    def forward(self, x):
        #print("Input: {}".format(x.shape))
        x = self.features(x)
        #print("Features: {}".format(x.shape))
        #Flatten the vector
        x = x.view(x.shape[0], -1)
        #print("Flattened: {}".format(x.shape))
        x = self.classifier(x)
        x = self.output_fn(x)
        return x

'''
Patch Loss Criterion Definition
'''


class DXPatch_Loss(torch.nn.Module):
    def __init__(self,device):
        super(DXPatch_Loss,self).__init__()
         
        self.patch_weights = torch.tensor( [
          1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, # osc6
          1, 1, 1, 1, 1, 4, 4, 10, 4, 1,
          1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, # osc5
          1, 1, 1, 1, 1, 4, 4, 10, 4, 1,
          1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, # osc4
          1, 1, 1, 1, 1, 4, 4, 10, 4, 1,
          1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, # osc3
          1, 1, 1, 1, 1, 4, 4, 10, 4, 1,
          1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, # osc2
          1, 1, 1, 1, 1, 4, 4, 10, 4, 1,
          1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, # osc1
          1, 1, 1, 1, 1, 4, 4, 10, 4, 1,
          1, 1, 1, 1, 1, 1, 1, 1, # pitch eg rate & level 
          10, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1  # algorithm feedback osc-sync ... #original value for transpose(last one) = 10
          ] ).to(device)
        self.patch_maxes = torch.tensor( [
          99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, # osc6
          3, 3, 7, 3, 7, 99, 1, 31, 99, 14,
          99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, # osc5
          3, 3, 7, 3, 7, 99, 1, 31, 99, 14,
          99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, # osc4
          3, 3, 7, 3, 7, 99, 1, 31, 99, 14,
          99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, # osc3
          3, 3, 7, 3, 7, 99, 1, 31, 99, 14,
          99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, # osc2
          3, 3, 7, 3, 7, 99, 1, 31, 99, 14,
          99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, # osc1
          3, 3, 7, 3, 7, 99, 1, 31, 99, 14,
          99, 99, 99, 99, 99, 99, 99, 99, # pitch eg rate & level 
          31, 7, 1, 99, 99, 99, 99, 1, 5, 7,48  # algorithm feedback osc-sync ...
          ] ).to(device)

    def forward(self,x,y):
        #Normalize patch label
        yn = torch.div(y,self.patch_maxes)
        #Compute L1 difference
        diff = torch.abs(x-yn)
        #Weight some parameters and sum everything
        totloss = torch.sum(torch.mul(diff,self.patch_weights))
        return totloss

'''
Train and Test functions definition
'''

def train(epoch):
    model.train()
    epoch_loss = 0
    for batch_idx, instance in enumerate(train_loader):
        data = instance['audio']
        target = instance['patch']
        if args.cuda:
            data, target = data.cuda(), target.cuda()
        #data, target = Variable(data), Variable(target)

        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
    epoch_loss /= len(train_loader)
    print('Epoch {} - Train Loss: {:.6f}'.format(
        epoch,epoch_loss))

def validate(save_model=False):
    model.eval()
    valid_loss = 0
    global best_loss
    with torch.no_grad():
        for instance in valid_loader:
            data = instance['audio']
            target = instance['patch']            
            if args.cuda:
                data, target = data.cuda(), target.cuda()
            
            output = model(data)
            loss = criterion(output, target)
            valid_loss += loss.item()
    valid_loss /= len(valid_loader)
    print('\tValid Loss: {:.4f}'.format(valid_loss))    
    if valid_loss < best_loss:
        # save model
        if save_model:
            #torch.save(model.state_dict(), save_path, _use_new_zipfile_serialization=False)
            torch.save(model, save_path,_use_new_zipfile_serialization=False)
            print("[INFO] Model saved at: ", save_path, "\n")
        best_loss = valid_loss

def test():
    model.eval()
    test_loss = 0

    with torch.no_grad():
        for instance in test_loader:
            data = instance['audio']
            target = instance['patch']  
            if args.cuda:
                data, target = data.cuda(), target.cuda()
            output = model(data)
            loss = criterion(output, target)
            test_loss += loss.item()

    test_loss /= len(test_loader)
    print('\nTest Loss: {:.4f}\n'.format(test_loss))

if __name__ == '__main__':

    # Indentify
    SEED = args.seed

    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed(SEED)
    torch.backends.cudnn.deterministic = True

    kwargs = {'num_workers': 1, 'pin_memory': True} if args.cuda else {}

    # Load dataset and define transforms.
    note_on_len = args.fs
    note_off_len = args.fs
    dataset = dxds.dxdataset(args.fs,args.dataset,(22,36,40,54),(127,),
        note_on_len,note_off_len,random_seed=SEED,filter_function='all_ratio')

    n_train_examples = int(len(dataset)*0.7)
    n_valid_examples = int(len(dataset)*0.2)
    n_test_examples =  len(dataset) - n_train_examples - n_valid_examples

    train_data, valid_data, test_data = torch.utils.data.random_split(dataset, 
                                                           [n_train_examples, n_valid_examples, n_test_examples])    
    
    train_loader = torch.utils.data.DataLoader(train_data, 
                                                 shuffle = True, 
                                                 batch_size = args.batch_size)

    valid_loader = torch.utils.data.DataLoader(valid_data, 
                                                 batch_size = args.batch_size)

    test_loader = torch.utils.data.DataLoader(test_data, 
                                                batch_size = args.batch_size)
    

    OUTPUT_DIM = 145
    model = DXNET(OUTPUT_DIM, DXNET_block, nn.MaxPool1d)
    if args.cuda:
        torch.cuda.set_device(0)
        model.cuda()
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
    criterion = DXPatch_Loss(device)
    criterion = criterion.to(device)
    # test model
    if args.eval:
        print("Evaluating stored model . . .")
        #model.load_state_dict(torch.load(save_path))
        model = torch.load(save_path)
        test()
    # train model
    else:
        if args.resume:
            print("Resuming . . .")
            #model.load_state_dict(torch.load(save_path))
            model = torch.load(save_path)
            validate(save_model = False)
        else:
            print("Training . . .")
        optimizer = optim.Adam(model.parameters(), lr=args.lr)  
        for epoch in range(1, args.epochs + 1):
            train(epoch)
            validate(save_model=True)
            #Learning rate scaling.
#            if epoch%40==0 and epoch != 0:
#                new_lr = optimizer.param_groups[0]['lr']*0.1
#                print("[INFO] Scaling lr to {}.".format(new_lr))
#                optimizer.param_groups[0]['lr']=new_lr

        #Finished Training. Load the best model and test it.
        #model.load_state_dict(torch.load(save_path))
        model = torch.load(save_path)
        test()
