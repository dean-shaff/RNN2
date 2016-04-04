import theano 
import theano.tensor as T 
import numpy as np 
from character_mapping import Character_Map
import time
        # self.Whnhn = theano.shared(
        #           value=numpy.asarray(
        #                 rng.uniform(
        #                     low=-numpy.sqrt(6. / (n_in + n_out)),
        #                     high=numpy.sqrt(6. / (n_in + n_out)),
        #                     size=(n_in, n_out)
        #                 ),
        #                 dtype=theano.config.floatX
        #             ),
        #             name='Whnhn',
        #             borrow=True)

class RNNlayer(object):

    def __init__(self,input,n_in,n_out,rng=None,transfer=None,h_prev=None):
        """
        args:
        rng: numpy random state 
        input: the x matrix, or the sequence of input vectors 
        n_in: the size of each of the x vectors in the input sequence 
        n_out: the size of the output vector 
        transfer: the transfer function to use 
        h_prev: for multilayer RNNs (not yet implemented) This would be the h output from
            the previous RNN layer. 

        """
        self.input = input 
        self.seq_length = T.shape(input)[0]
        if rng == None:
            self.Whnhn = theano.shared(
                        value=np.zeros(
                            (n_in, n_in),
                            dtype=theano.config.floatX
                        ),
                        name='Whnhn',
                        borrow=True
                    )

            self.Wihn = theano.shared(
                        value=np.zeros(
                            (n_in, n_in),
                            dtype=theano.config.floatX
                        ),
                        name='Wihn',
                        borrow=True
                    )

            self.bnh = theano.shared(
                        value=np.zeros(
                            (n_in,),
                            dtype=theano.config.floatX
                        ),
                        name='bnh',
                        borrow=True 
                    )   
        else:
            print("Not yet implemented!")
            
        self.params = [self.Whnhn, self.Wihn, self.bnh]
        # the below code is just for the first layer.
        def recurrent_step(xt,htm1):
            ht = T.nnet.sigmoid(T.dot(self.Wihn,xt) + T.dot(self.Whnhn,htm1) + self.bnh)
        #     yt = T.nnet.softmax(T.dot(Wh1y,ht) + by)
            return ht#, yt

        def single_sequence(seq):
            h = T.nnet.sigmoid(T.dot(self.Wihn,seq[0]) + self.bnh)
            h0 = T.reshape(h, (1,h.shape[0]))


            results, updates = theano.scan(fn=recurrent_step,
                                            outputs_info=[h],
                                           sequences=[seq[1:]])

            return T.concatenate([h0, results])

        if (input.ndim == 2): 
            self.output = single_sequence(input)
        elif(input.ndim == 3):
            self.output, updates = theano.scan(fn=single_sequence,
                                        sequences=[input])

class RNN(object):

    def __init__(self,input,layers):
        """
        layers is a list or tuple containing the number of elements in the each of the hidden layers. 
        """
        if len(layers) == 1:
            self.layers = RNNlayer(input,layers[0],layers[0])
            self.params = self.layers.params 
            n_out = layers[0]
            n_in = layers[0]
            results = self.layers.output 
        else:
            print("Not yet implemented!")


        self.Whny = theano.shared(
                    value=np.zeros(
                        (n_in, n_out),
                        dtype=theano.config.floatX
                    ),
                    name='Wihn',
                    borrow=True
                )

        self.by = theano.shared(
                    value=np.zeros(
                        (n_out,),
                        dtype=theano.config.floatX
                    ),
                    name='bnh',
                    borrow=True 
                )   
        self.params += [self.Whny, self.by] 
        # self.p_y_given_x = T.nnet.softmax(T.dot(self.Whny, results) + self.by)
        def single_result_sequence(seq_result):

            p_y_given_x = T.nnet.softmax(T.transpose(T.dot(self.Whny, T.transpose(seq_result))) + self.by)
            return p_y_given_x

        if input.ndim == 2:
            self.p_y_given_x = single_sequence(results)
            self.y_pred = T.argmax(self.p_y_given_x,axis=1)
            
        elif input.ndim == 3: #has to be a smarter way to do this! 
            self.p_y_given_x, updates = theano.scan(fn=single_result_sequence,
                                            sequences=[results])
            self.y_pred = T.argmax(self.p_y_given_x,axis=2)
        # self.p_y_given_x = T.transpose(results)
        # I want the 'argmax' to be taken on the last dimension. 
        # this way i get an 'argmax' for each vector in the sequence. 
        # self.y_pred = T.argmax(self.p_y_given_x, axis=2) 

    def neg_log_likelihood(self,y):
        """
        loss function for the RNN 

        Each y is a matrix, as we're passing in the correct value for a minibatch. 
        Each row in the matrix corresponds to the correct elements in the sequence.
        For example, if the first letter in our set were 'a' and there were an 
        'a' in the second spot in the sequence, the second element of the first row would be a '0'.
        y has to be integer for this to work 
        """
        # return -T.mean(T.log(self.p_y_given_x)[T.arange(y.shape[0]),:,y])    
        log_tot = T.log(self.p_y_given_x)

        log_prob, updates = theano.scan(fn=lambda i,yi: log_tot[i,T.arange(y.shape[1]),yi],
                                        sequences = [T.arange(y.shape[0]), y]) 

        return log_prob
        # return T.log(self.p_y_given_x)[T.arange(y.shape[0]),y]     

    def error(self,y):
        """
        get the error across a minibatch 
        """
        return T.sum(T.neq(self.y_pred,y))



def load_dataset(filename):
    foo = Character_Map(filename,'mapping.dat',overwrite=True)
    # print(len(foo.mapping))
    map_matrix = foo.k_map()
    return foo.gen_train_valid_test(filename=None)

def test_train_RNN(**kwargs):
    """
    kwargs
    """

    filename = kwargs.get('filename','./../texts/melville.txt')
    n_hidden = kwargs.get('n_hidden',77)
    n_epochs = kwargs.get('n_epochs',100)
    minibatch_size = kwargs.get('minibatch_size',100)
    lr = kwargs.get('lr',0.01)

    charmap = Character_Map(filename,'mapping.dat',overwrite=True)
    charmap.k_map()
    train, valid, test = charmap.gen_train_valid_test()

    train_set_x, train_set_y = train
    valid_set_x, valid_set_y = valid 
    test_set_x, test_set_y = test

    index = T.lscalar()
    x = T.tensor3('x')
    y = T.imatrix('y')

    rng = numpy.random.RandomState(1234)

    rnn = RNN(x,[77]) #i need to change this to take into account different in and out sizes. 

    cost = rnn.neg_log_likelihood(y)
    print("Compiling training, testing and validating functions...")
    t0 = time.time()
    test_model = theano.function(
            inputs=[index],
            outputs=rnn.error(y),
            givens={
                x: test_set_x[index * batch_size:(index + 1) * batch_size],
                y: test_set_y[index * batch_size:(index + 1) * batch_size]
            }

        )

    valid_model = theano.function(
            inputs=[index],
            outputs=rnn.error(y),
            givens={
                x: valid_set_x[index * batch_size:(index + 1) * batch_size],
                y: valid_set_y[index * batch_size:(index + 1) * batch_size]
            }
        )

    gparams = [T.grad(cost, param) for param in rnn.params]

    updates = [
        (param, param-learning_rate*gparam) for param, gparam in zip(rnn.params,gparams)
    ]

    train_model = theano.function(
            inputs = [index],
            outputs = cost,
            updates = updates,
            givens = {
                x: test_set_x[index * batch_size:(index + 1) * batch_size],
                y: test_set_y[index * batch_size:(index + 1) * batch_size]
            }
        )
    print("Completed compiling functions. Took {:.2f} seconds".format(time.time() - t0))

    #now with all the functions compiled we can go ahead and actually make shit run. 



if __name__ == '__main__':
    filename = './../texts/melville.txt'
    foo = Character_Map(filename,'mapping.dat',overwrite=True)
    # print(len(foo.mapping))
    map_matrix = foo.k_map()
    train, valid, test = foo.gen_train_valid_test(filename=None)
    # print(train[1].get_value().dtype)
    # print(train[1].get_value()[:10].shape)
    x = T.tensor3('x')
    y = T.imatrix('y')
    # x = T.matrix('x')
    rnnlayer = RNNlayer(x,77,77)

    rnn = RNN(x,[77]) #the number of unique characters in Moby Dick 
    # ftest = theano.function(inputs=[x], outputs=rnn.p_y_given_x)
    # print(ftest(train[0].get_value()[:10]).shape)
    print("Compiling training and testing functions...")
    t0 = time.time()
    ftrain = theano.function(inputs=[x,y],outputs=rnn.neg_log_likelihood(y))
    ftest = theano.function(inputs=[x,y], outputs=rnn.error(y))
    print("Completed compiling functions. Took {:.2f} seconds".format(time.time() - t0))
    for i in xrange(10):
        # print(ftrain(train[0].get_value()[i*10:(i+1)*10], train[1].get_value()[i*10:(i+1)*10]))
        print(ftest(test[0].get_value()[i*10:(i+1)*10], test[1].get_value()[i*10:(i+1)*10]))


    # f = theano.function(inputs=[x], outputs=rnnlayer.output)
    # f1 = theano.function(inputs=[x],outputs=rnn.p_y_given_x)
    # print(f(shared_x.get_value()[:5]).shape)
    # print(f1(shared_x.get_value()[:5]).shape)














