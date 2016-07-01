import theano
import theano.tensor as T 
import numpy as np
import time 
from RNN_new import RNN 
from dataset import TextDataset 

class Trainer(object):

    def __init__(self, model, dataset):
        
        self.model = model
        self.dataset = dataset 


    def compile_functions(self, x, y):
        
        mb = T.scalar('mb',dtype='int64')
        lr = T.scalar('lr') 
        index = T.scalar('index',dtype='int64')

        print("Compiling theano functions...\n") 
        t0 = time.time()
        self.feed_forward = theano.function([x],self.model.out) 
        
        self.cost = self.model.cross_entropy_SGD(y)
        self.error = self.model.error_SGD(y)

        grad_params = [T.grad(self.cost, param) for param in self.model.params]
        updates = [(param, param-lr*gparam) for param, gparam in zip(self.model.params, grad_params)]
        
        
        self.train_model = theano.function(
            inputs = [index, lr,mb],
            outputs = self.cost, 
            updates = updates,
            givens = {
                x: self.dataset.in_train[(index*mb):(index+1)*mb],
                y: self.dataset.obs_train[(index*mb):(index+1)*mb],
            }
        )

        self.error = theano.function(
            inputs = [x,y],
            outputs = self.error,
        ) 
        print("Functions compiled. Took {:.2f} seconds".format(time.time() - t0))
        #return self.train_model, self.error 

    def gradient_descent(self, lr, mb, n_epochs):
        in_test = self.dataset.in_test.get_value() 
        obs_test = self.dataset.obs_test.get_value()
        in_train = self.dataset.in_train.get_value() 
        obs_train = self.dataset.obs_train.get_value()

        train_batches = self.dataset.in_train.get_value().shape[0] // mb  
        print("There are {} minibatches per epoch".format(train_batches)) 
        for epoch in xrange(n_epochs):
            print("\n\nStarting epoch {}...\n\n".format(epoch+1))
            for b in xrange(train_batches):
                t0 = time.time()
                cur_cost = self.train_model(b, lr,mb)
                print("Time calculating minibatch cost: {:.4f}. Cost: {}".format(time.time() - t0,cur_cost)) 
                if b % 20 == 0 and b != 0:
                    t0 = time.time() 
                    r = np.random.randint(0, in_test.shape[0] - 1001)
                    err_test = self.error(in_test[r:r+1000], obs_test[r:r+1000]) 
                    err_train = self.error(in_train[r:r+1000], obs_train[r:r+1000]) 
                    print("Current cost: {}".format(cur_cost))
                    print("Current Test Error: {}".format(err_test))
                    print("Current Train Error: {}".format(err_train))
                    print("Time calculating errors: {:.4f}".format(time.time() - t0))

if __name__ == '__main__':
    
    dataset = TextDataset('shakespeare.hdf5') 
    dataset.cut_by_sequence(10,classify=False)
    #x = T.matrix('x') 
    #y = T.matrix('y') 
    x = T.tensor3('x')
    y = T.tensor3('y') 
    foo = np.random.rand(10,50) #random data 
    nhid = 200
    rnn = RNN(x, dataset.seq_len, [dataset.char_len,dataset.char_len,nhid],mode='LSTM',bptt_truncate=-1)

    trainer = Trainer(rnn, dataset)

    trainer.compile_functions(x,y) 
    trainer.gradient_descent(0.01,200,10) 
    #print(trainer.feed_forward(foo).shape) 
