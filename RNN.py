import theano 
import theano.tensor as T 
import numpy as np 
from character_mapping import Character_Map
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
        return -T.mean(T.log(self.p_y_given_x)[T.arange(y.shape[0]),y,:])     

    def error(self,y):
        """
        get the error across a minibatch 
        """
        return T.sum(T.neq(self.y_pred,y))

if __name__ == '__main__':
    filename = './../texts/melville.txt'
    foo = Character_Map(filename,'mapping.dat',overwrite=True)
    # print(len(foo.mapping))
    map_matrix = foo.k_map()
    x_,y_,shared_x, shared_y = foo.gen_x_and_y(filename=None)
    print(shared_x.get_value()[0].shape)
    x = T.tensor3('x')
    # x = T.matrix('x')
    rnnlayer = RNNlayer(x,77,77)

    rnn = RNN(x,[77]) #the number of unique characters in Moby Dick 
    f = theano.function(inputs=[x], outputs=rnnlayer.output)
    f1 = theano.function(inputs=[x],outputs=rnn.p_y_given_x)
    print(f(shared_x.get_value()[:5]).shape)
    print(f1(shared_x.get_value()[:5]).shape)














