import theano
import theano.tensor as T 
import numpy as np 
import time  

class RNN_HiddenLayer(object):

    def __init__(self,x,chardim,nout,nhid,**kwargs):
        
        self.bptt_truncate = kwargs.get('bptt_truncate',5) 
        
        #Wxh = np.random.uniform(-np.sqrt(1.0/chardim),np.sqrt(1.0/chardim),(nhid,chardim))
        #Whh = np.random.uniform(-np.sqrt(1.0/nhid),np.sqrt(1.0/nhid),(nhid,nhid))
        #Woh = np.random.uniform(-np.sqrt(1.0/nhid),np.sqrt(1.0/nhid),(nout,nhid))
        Wxh = np.random.uniform(-np.sqrt(1.0/chardim),np.sqrt(1.0/chardim),(chardim,nhid))
        Whh = np.random.uniform(-np.sqrt(1.0/nhid),np.sqrt(1.0/nhid),(nhid,nhid))
        Woh = np.random.uniform(-np.sqrt(1.0/nhid),np.sqrt(1.0/nhid),(nhid,nout))
        bh = np.zeros(nhid)
        bo = np.zeros(nout)

        self.Wxh = theano.shared(Wxh.astype(theano.config.floatX),name='Wxh')
        self.Whh = theano.shared(Whh.astype(theano.config.floatX),name='Whh')
        self.Woh = theano.shared(Woh.astype(theano.config.floatX),name='Woh')
        self.bh = theano.shared(bh.astype(theano.config.floatX),name='bh')
        self.bo = theano.shared(bo.astype(theano.config.floatX),name='bo')
        
        self.x = x 
        self.chardim = chardim 
        self.nout = nout
        self.nhid = nhid
        
        self.params = [self.Wxh,self.Whh, self.Woh, self.bh, self.bo] 

        def recurrant_step(xt,h_prev):
            
            #ht = T.nnet.sigmoid(T.tensordot(xt,self.Wxh) + T.tensordot(h_prev, self.Whh) + self.bh)
            #ot = T.nnet.softmax(T.tensordot(ht,self.Woh) + self.bo)

            ht = T.nnet.sigmoid(T.dot(xt,self.Wxh) + T.dot(h_prev, self.Whh) + self.bh)
            ot = T.nnet.softmax(T.dot(ht,self.Woh) + self.bo)
            #ht = T.nnet.sigmoid(T.dot(self.Wxh,xt) + T.dot(self.Whh, h_prev) + self.bh)
            #ot = T.nnet.softmax(T.dot(self.Woh, ht) + self.bo)
            return [ot, ht]
       
        
        x_reshape = x.reshape((x.shape[1],x.shape[0],x.shape[2]))
        [o,h], _ = theano.scan(recurrant_step,
                                sequences = x_reshape,
                                truncate_gradient = self.bptt_truncate, 
                                outputs_info = [None, {'initial':T.zeros((x.shape[0],self.nhid))}])
        self.o = o.reshape((x.shape[0],x.shape[1],x.shape[2]))      
        #self.o = o[:,0,:]
        
class RNN(object):

    def __init__(self, x, seq_len, dim):
        """
        Not implemented for multiple layers yet. Soon tho. 
        """
        layer = RNN_HiddenLayer(x,*dim) 
        self.out = layer.o 
        self.params = layer.params 

    def cross_entropy(self,x,y):
         
        return T.sum(T.nnet.categorical_crossentropy(x, y))

    def cross_entropy_SGD(self,y3):
        
        entr, _ = theano.scan(self.cross_entropy,
                                sequences = [self.out,y3])
        return T.sum(entr) 

    def error(self,x,y):

        return T.mean(T.neq(T.argmax(x, axis=1),T.argmax(y, axis=1)))

    def error_SGD(self,y3):

        err, _ = theano.scan(self.error,
                            sequences = [self.out, y3]) 

        return T.mean(err)  

def numpy_test():
    x = T.matrix('x') 
    y = T.matrix('y')
    xi = T.vector('xi') 
    l = RNN_HiddenLayer(x,50,50,100)
    f = theano.function([x], l.o)
    foo = np.random.rand(10,50)
    theano_version = f(foo) 
    print(theano_version.shape) 
    sig = theano.function([xi],T.nnet.sigmoid(xi))
    soft = theano.function([xi],T.nnet.softmax(xi))
    outs = np.zeros((10,50))
    ht = np.zeros(100) 
    for i in xrange(foo.shape[0]):
        ht = sig(np.dot(l.Wxh.get_value(),foo[i]) + np.dot(l.Whh.get_value(),ht) + l.bh.get_value())
        outs[i,:] = soft(np.dot(l.Woh.get_value(),ht) + l.bo.get_value())

    print(np.allclose(outs,theano_version))

if __name__ == '__main__':
    x = T.matrix('x')
    y = T.matrix('y')
    x3 = T.tensor3('x3')
    y3 = T.tensor3('y3')

    foo = np.random.rand(20,10,50)
    foo1 = np.random.rand(20,10,50)
    #print(foo.reshape((foo.shape[1],foo.shape[0],foo.shape[2])).shape) 
    rnn = RNN(x3, 10, [50,50,100]) 
    cost = rnn.cross_entropy_SGD(y3)
    error = rnn.error_SGD(y3) 
    
    #f_cost = theano.function([x3,y3], cost)
    #print(f_cost(foo, foo1).shape) 
    f_err = theano.function([x3,y3],error)
    print(f_err(foo, foo1).shape) 
    
    #f= theano.function([x3],rnn.out)
    #print(f(foo).shape) 
   # numpy_test()
