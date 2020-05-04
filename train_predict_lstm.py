import numpy as np
import scipy as sc
import sklearn as sk
from sklearn import datasets, linear_model, model_selection, metrics
import pandas as pd

import matplotlib.pyplot as plt
import matplotlib as mpl
np.set_printoptions(precision=2)
plt.rcParams['figure.figsize'] = [12, 8]

import time
import datetime


#from sklearn.preprocessing import MinMaxScaler

import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
import torch.nn.functional as F

from torch.utils.data import DataLoader

import torch

import compare_models_functions as myfxns


def sliding_windows(data, seq_length):
    '''given 1D arr of timepoints, return array of sliding windows (x) 
    and value immediately following (y)'''
    x = []
    y = []

    for i in range(len(data)-seq_length-1):
        _x = data[i:(i+seq_length)]
        _y = data[i+seq_length]
        x.append(_x)
        y.append(_y)

    return np.array(x),np.array(y)


def sliding_windows_counties(Xtrain, input_seq_len = 14):
    ''' generate sliding windows to train on. 
    Xtrain is np arr of (counties, fips and counts for each day).
    return np arr of sliding windows training samples for each county.
    ex/ if using 14 days as input, each is row [fips, day1, ..., day14, day_15],
    and multiple such rows for each county for different windows'''


    # total number of samples will be num counties * num windows for each
    num_counties = Xtrain.shape[0]
    num_dates = Xtrain.shape[1] - 1 # don't include fips as date

    if input_seq_len > num_dates:
        print('ERROR: input seq len %d is longer than number of dates %d available for training'
              % (input_seq_len, num_dates) )
        return None
    
    num_windows = num_dates - input_seq_len - 1 # num windows per county
    print('num counties, dates, windows:', num_counties, num_dates, num_windows)

    # slide_train: fips is col 0, middle is input seq, target is last col
    slide_train = -1*np.ones((num_counties*num_windows, input_seq_len + 2))

    for i in range(Xtrain.shape[0]):
        x, y = sliding_windows(Xtrain[i,1:], input_seq_len)
        fips_i = Xtrain[i,0]
        rows_i = tuple(range((i*num_windows),((i+1)*num_windows)))
        slide_train[rows_i,0] = fips_i
        slide_train[rows_i,1:-1] = x
        slide_train[rows_i,-1] = y
    
    return slide_train



class SeqDataset(torch.utils.data.Dataset):
    '''dataset that optionally maintains list of IDs for each sample (e.g. county name or fips).
    inputs, labels, and list_IDs must be in same order'''
    def __init__(self, inputs, labels, has_header_col=False):
        'Initialization'

        self.inputs = inputs # assume 2D array, each row is sample
        self.labels = labels # assume 2D, each row is sample
        self.list_IDs = ['no_id' for i in range(self.inputs.shape[0])]
            
        if has_header_col: # if inputs has header col
            self.list_IDs = inputs[:,0]
            self.inputs = inputs[:,1:]

    def __len__(self):
        'Denotes the total number of samples'
        return self.inputs.shape[0]

    def __getitem__(self, index):
        'Generates one sample of data'
        
        sample_id = self.list_IDs[index]
    
        X = torch.Tensor(self.inputs[index,:])
        y = torch.Tensor(self.labels[index,:])

        return X, y, sample_id
    
    # could add option to look up ID of a datapoint, and make inputs and labels dictionary
    #ID = self.list_IDs[index]
    

class LSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim, batch_size=1, output_dim=1,
                    num_layers=10, dropout=0.2):
        super(LSTM, self).__init__()
        self.input_dim = input_dim   # how many features per time point (or token)
        self.hidden_dim = hidden_dim
        self.batch_size = batch_size
        self.num_layers = num_layers
        self.dropout = dropout
        self.output_dim = output_dim

        # lstm layers and linear layer to output single time point
        self.lstm = nn.LSTM(self.input_dim, self.hidden_dim, self.num_layers, 
                            dropout=self.dropout, batch_first=True)
        self.linear = nn.Linear(self.hidden_dim, self.output_dim)
        
        # hidden layer
        self.hidden = (torch.zeros(self.batch_size, self.num_layers, self.hidden_dim),
                torch.zeros(self.batch_size, self.num_layers, self.hidden_dim))
 
    def reset_hidden(self): # use to reset hidden state
        self.hidden = (torch.zeros(self.num_layers, self.batch_size, self.hidden_dim),
                torch.zeros(self.num_layers, self.batch_size, self.hidden_dim))
        # should it be this for batch first?? TODO
        #self.hidden = (torch.zeros(self.batch_size, self.num_layers, self.hidden_dim),
        #        torch.zeros(self.batch_size, self.num_layers, self.hidden_dim))
    

    def forward(self, inputs):
        # note: dim depends on whether lstm layer initialized with batch_first. 
        # Code assumes batch_first, unidirectional
        
        # nn.LSTM expects input shape ( timepoints, batch_size, features per timepoint)
        #        or (batch_size, timepoints, features per timepoint) if batch_first
        # lstm_out: (seq_len, batch, num_directions * hidden_size) 
        #        or (batch, seq_len, numdir*hidden_size)
        # self.hidden: tup of two (num_layers, batch_size, hidden_dim)
        #                     or (batch_size, num_layers, hidden_dim)
        lstm_out, self.hidden = self.lstm(inputs.view(self.batch_size, -1, self.input_dim), self.hidden) 
        # linear layer takes output of final timestep
        y_pred = self.linear(lstm_out[:,-1,:])
        
        return y_pred
    

# input_dim = # of features per timepoint/token
# pred_seq_len = # days to output at once
def train_lstm(train_dataset, batch_size=100, num_epochs=10, lr=0.05, loss_fxn='MSE', cpu_or_cuda = 'cpu',
               input_dim=1, hidden_dim=21, pred_seq_len=1, num_layers=2, lstm_dropout=0.5):
    '''simple train function w batches w/o early stopping or checkpoints'''
    
    DEVICE = torch.device(cpu_or_cuda) # 'cuda')
    print('Using device: ' + str(DEVICE))
    train_loader = DataLoader(train_dataset, batch_size = batch_size, shuffle = True, 
                              num_workers = 8, drop_last = True)
    # TODO trying with 8 workers instead of 4
    # TODO handle batches which aren't multiple of whole dataset, currently drop, could pad


    model = LSTM(input_dim, hidden_dim, output_dim=pred_seq_len, batch_size=batch_size, 
                 num_layers=num_layers, dropout=lstm_dropout)
    print('training model with [hidden dim %d, num layers %d, dropout %.3f, loss %s, batch %d, lr %.5f, epoch %d]' 
          % (hidden_dim, num_layers, lstm_dropout, loss_fxn, batch_size, lr, num_epochs) )
    model = model.to(DEVICE)
    model = model.train()
    ####model = model.double() # set model to use double, otherwise float vs double error
    
    criterion = torch.nn.MSELoss() # regression
    if loss_fxn == 'L1':
        criterion = torch.nn.L1Loss()
        
    optimizer = torch.optim.SGD(model.parameters(), lr=lr) # can try other optimizer

    #num_batches = int(len(train_dataset) / float(batch_size)) + 1
    

    for epoch in range(num_epochs):
        running_loss = 0.0
        for i, data in enumerate(train_loader, 0):
            # get batch
            inputs, labels, sample_id = data[0].to(DEVICE), data[1].to(DEVICE), data[2]
                        
            # zero the parameter gradients and reset hidden
            optimizer.zero_grad()
            model.reset_hidden()

            # forward + backward + optimize
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # update loss
            running_loss += loss.item()


        #End of epoch report (could do per item loss)
        if epoch % 1 == 0:
            print('[Epoch %d] loss: %.3f' %
                  (epoch + 1, running_loss )) 

    print('Finished Training')
    return model




def predict_lstm(slide_model, input_arr, pred_seq_len=7, input_seq_len=14):
    '''slide_model is trained model that outputs next day prediction.
    input_arr is 2d arr with counties as rows, fips as 0th col, and input days as remaining cols.
    input_seq_len is how many of previous days to input.
    pred_seq_length is how many days to predict'''
    
    slide_model.eval().cpu() # put on cpu since outputting np arr
    slide_model.reset_hidden()
    slide_model.batch_size = 1
    # TODO this could be easily done in batches for speedup, would have to handle last batch

    # first col is fips, rest is predictions
    preds = -1 * np.ones((input_arr.shape[0], pred_seq_len+1)) 
    preds[:,0] = input_arr[:,0]
    county_dataset = SeqDataset(input_arr, input_arr, has_header_col=True)
    preds_county_loader = DataLoader(county_dataset, batch_size = 1, shuffle = False, 
                                  num_workers = 6, drop_last = True)

    with torch.no_grad():
        for i, data in enumerate(preds_county_loader, 0):
            data_i, dummy_labels, fips = data[0], data[1], data[2]
            # pass in known data for county i to predict first of future days
            input_seq = data_i[:,-input_seq_len:]
            slide_model.reset_hidden()
            next_day_pred = slide_model(input_seq) #.item()
            preds[i,1] = next_day_pred
            for j in range(1, pred_seq_len): # pass in prev day prediction to next
                input_seq = torch.FloatTensor( np.reshape(preds[i,j], (1,1,1)) )
                next_day_pred = slide_model(input_seq) #.item() # pred next day
                preds[i,j+1] = next_day_pred # first col is fips

    return preds

if __name__ == '__main__':

    # repeat train/pred using different LSTM model and training settings
    # [hidden dim 20, num layers 2, dropout 0.500, loss MSE, batch 100, lr 0.01000, epoch 20]
    # [hidden dim 10, num layers 1, dropout 0.000, loss MSE, batch 50, lr 0.02000, epoch 20]
    argsTups1 = [('hidden_dim', 20), ('num_layers',2), ('dropout', 0.5), ('batch_size', 100), ('lr', 0.01), ('num_epochs', 20)]
    argsTups2 = [('hidden_dim', 10), ('num_layers',1), ('dropout', 0.0), ('batch_size', 50), ('lr', 0.02), ('num_epochs', 20)]
    
    ### load data
    cases_df, deaths_df = myfxns.load_county_datasets()

    # dict of fips to county, state (capitalized)
    fips2countystateD = myfxns.make_fips2countystateD()

    # filter data - only use 28 days as features, and use only counties with 28 days
    use_last_n_days=28
    use_log_counts = True # better performance pred log, then exponentiate
    X_df, X, y = myfxns.filter_and_scale_data(cases_df, cases_or_deaths = 'cases', 
        use_last_n_days=use_last_n_days,  max_frac_missing=0.0, 
        use_rel_counts=False, use_log_counts=use_log_counts, use_counts_only = True)
    print('use_last_n_days: %d, use_log_counts: %d' % (use_last_n_days, use_log_counts))

    X = np.array(  X_df.drop(['county', 'state'], axis=1) ) 

    ### make sliding windows to train on
    Xtrain = X[:,:-7] # first col of X is fips, which is converted to sampleID in data loader
    slide_train = sliding_windows_counties(Xtrain, input_seq_len = 14)
    # note: labels should be a 2D array, even if only predicting 1 day at a time
    train_dataset = SeqDataset(slide_train[:,:-1], slide_train[:,-1].reshape((-1,1)), has_header_col=True)


    ### make sliding windows of all 28 days
    slide_all = sliding_windows_counties(X, input_seq_len = 14)
    all_dataset = SeqDataset(slide_all[:,:-1], slide_all[:,-1].reshape((-1,1)), has_header_col=True)

    ### for different param settings, pred test and future counts
    for argsTups in (argsTups1, argsTups2):
        argsD = {}
        for arg, val in argsTups:
            argsD[arg] = val

        ### train, excluding last week as test set
        slide_model = train_lstm(train_dataset, 
            num_epochs=argsD['num_epochs'], batch_size=argsD['batch_size'], lr=argsD['lr'],
            hidden_dim=argsD['hidden_dim'], num_layers=argsD['num_layers'], 
            loss_fxn = 'MSE', lstm_dropout=argsD['dropout'], cpu_or_cuda='cpu')

        ## predict and write to file
        preds = predict_lstm(slide_model, Xtrain, pred_seq_len=7, input_seq_len=14)

        startDate = datetime.date(2020, 4, 24) # date of first prediction
        endDate = startDate + datetime.timedelta(days = 7)

        fname = 'prediction_csv/test_lstm_%d_%d_%d_%.3f_%d.csv' % \
                        (argsD['hidden_dim'], argsD['num_layers'], argsD['batch_size'],
                        argsD['lr'], argsD['num_epochs']) 
        myfxns.output_csv_preds(preds, fips2countystateD, startDate, endDate, fname, 
                         convertLog=True) #, startCol=-7, endCol=None)
        print('wrote %s\n' % fname)


        #### redo training on all 28 days of data, output next month
        slide_model = train_lstm(all_dataset, 
            num_epochs=argsD['num_epochs'], batch_size=argsD['batch_size'], lr=argsD['lr'],
            hidden_dim=argsD['hidden_dim'], num_layers=argsD['num_layers'], 
            loss_fxn = 'MSE', lstm_dropout=argsD['dropout'], cpu_or_cuda='cpu')

        preds = predict_lstm(slide_model, Xtrain, pred_seq_len=30, input_seq_len=14)

        startDate = datetime.date(2020, 5, 3) # date of first prediction
        endDate = startDate + datetime.timedelta(days = 30)

        fname = 'prediction_csv/future_lstm_%d_%d_%d_%.3f_%d.csv' % \
                        (argsD['hidden_dim'], argsD['num_layers'], argsD['batch_size'],
                        argsD['lr'], argsD['num_epochs']) 
        myfxns.output_csv_preds(preds, fips2countystateD, startDate, endDate, fname, 
                         convertLog=True) #, startCol=-30, endCol=None)
        print('wrote %s\n\n\n' % fname)


# TODO could grab start/end dates from X_df.columns or cases_df.columns
# TODO predict cases and deaths simultaneously
# TODO try passing in differences instead of raw counts
# TODO add one-hot encoding of state to linear layer, can modify data loader to handle this using fip2labelID
# TODO could pass in all available data for each county, even if only a couple days, rather than fixed number