import numpy as np 
import theano.tensor as T
import theano 
import h5py
import time 
from preprocessor import TextProcessor


class TextDataset(object):

    def __init__(self, filename,**kwargs):
        """
        args:
            -filename: should be an hdf5 file 
        """
        self.filename = filename  
        self.get_data(**kwargs)  

    def get_data(self,shared=False):
        """
        Grab the data from the dataset file and turn it into one hot vectors.
        kwargs:
            -shared: If true, load data into shared variable.s 
        """
        print("Loading data into memory")
        t0 = time.time() 
        f = h5py.File(self.filename,'r') 
        dat = f['dat'][...]
        dat = dat[int(dat.shape[0]*0.2):int(dat.shape[0]*0.4)]
        print("Data loaded from hdf5 file. Took {:.2f} seconds".format(time.time() - t0))
        chars = f.attrs['chars'] 
        dat_one_hot = np.zeros((dat.shape[0], len(chars)),dtype=theano.config.floatX) #need float because we're doing regression 
        indices = np.arange(dat.shape[0]) 
        dat_one_hot[indices,dat] = 1.0 
        print("One hot vectors generated. Took {:.2f} second".format(time.time() - t0))
        input_dat = dat_one_hot[:-1]
        obs_dat = dat_one_hot[1:]
        if shared:
            input_dat = theano.shared(input_dat)
            obs_dat = theano.shared(obs_dat) 
            print("Theano shared variables created. Took {:.2f} seconds.".format(time.time()  - t0))
        else:
            pass 
        f.close() 
        self.char_len = len(chars) 
        self.dat = dat 
        self.data_one_hot = dat_one_hot 
        self.obs_dat = obs_dat 
        self.input_dat = input_dat

    def cut_by_sequence(self, seq_len, classify=True):
        """
        Cut up the dataset into matrices, each (seq_len x char_len) big.
        This also creates training and testing datasets. 
        args:
            - seq_len: The length of the sequence 
        kwargs:
            - classify: Are we doing classification? If so, we need to store variables differently.(True) 
        """
        self.seq_len = seq_len
        print('Creating shared variables...') 
        t0 = time.time()
        n_seq = (self.dat.shape[0] // seq_len) -1
        if classify:
            obs_dat = self.obs_dat.astype('int64') 
            obs_dat = np.argmax(obs_dat,axis=1) 
            obs_dat = obs_dat[:n_seq*seq_len].reshape((n_seq,seq_len))
            self.obs_train = theano.shared(obs_dat[:int(0.6*n_seq),:])
            self.obs_test = theano.shared(obs_dat[int(0.6*n_seq):,:])

        elif not classify:
            obs_dat = self.obs_dat[:n_seq*seq_len,:].reshape((n_seq,seq_len,self.char_len))
            self.obs_train = theano.shared(obs_dat[:int(0.6*n_seq),:,:])
            self.obs_test = theano.shared(obs_dat[int(0.6*n_seq):,:,:])
        
        in_dat = self.input_dat[:n_seq*seq_len,:].reshape((n_seq,seq_len,self.char_len))
        self.in_train = theano.shared(in_dat[:int(0.6*n_seq),:,:])
        self.in_test = theano.shared(in_dat[int(0.6*n_seq):,:,:])
        print("Created shared variables. Took {:.2f} seconds.".format(time.time() - t0))
        







if __name__ == '__main__':
    filename = 'shakespeare.hdf5'
    txter = TextDataset(filename)
    print(txter.char_len) 
    txter.cut_by_sequence(5,classify=False) 
