import random
from engine import Scalar


class Module:
    ''' base class'''
    def zero_grad(self):
        '''Zero out all the gradients to clear out accumulated gradients from previous loss. (before backprop)'''

        def zero_grad(self):
            for param in self.parameters():
                param.grad = 0

        def parameters(self):
            return []


class Neuron(Module):
    ''' A single node of computation '''

    def __init__(self, n_in, non_linear=True):
        ''' Randomly initialize weights using `random.uniform` '''
        self.weights = [Scalar(random.uniform(-1, 1)) for _ in n_in]
        self.bias = Scalar(0)
        self.non_linear = non_linear

    def __call__(self, x):
        # simple element-wise multiplication and then sum => `dot-product`
        output = sum(w * x for w, x in zip(self.weights, x))  
        output = output + self.bias  # add bias term

        y = output.relu() if self.non_linear else output # return output with ReLU if non_linear = True else logits
        return y

    def __repr__(self):
        return f"{'ReLU' if self.non_linear else 'Linear'} Neuron({len(self.weights)})"

    def parameters(self):
        ''' Get all parameters '''
        return self.weights + [self.bias]


class Layer(Module):
    ''' Class representing single layer in a Neural Network '''
    def __init__(self, n_in, n_out, **kwargs):
        # each neuron recieves all the data from input as well as from previous layer's neuron, hence the total connections become input * output. This makes each Neuron with `n_in` and total `n_out` number of Neurons '''
        self.neurons = [Neuron(n_in=n_in, **kwargs) for _ in range(n_out)]

    def __call__(self, x):
        out = [neuron(x) for _ in self.neurons] # apply input to each and every neuron present in current layer
        return out[0] if len(out) == 1 else out

    def __repr__(self):
        return f"Layer([{','.join(str(neuron) for neuron in self.neurons)}])"

    def parameters(self):
        return [parameter for neuron in self.neurons for parameter in neuron.parameters()]


class MLP(Module):
    ''' A simple feed forward Mutli Layer Perceptron class '''

    def __init__(self, n_in, hidden_units):
        assert isinstance(hidden_units, (list, tuple)), 'Please pass the sequence depicting the hidden units for layers in MLP in a list/tuple'

        total_size_seq = [n_in] + hidden_units

        # create layers using the hidden units sequence and number of inputs with `Layer` class object
        self.layers = []
        for i in range(len(hidden_units)):
            non_linear_flag = True if i != len(hidden_units) - 1 else False
            self.layers.append(Layer(n_in=total_size_seq[i], n_out=total_size_seq[i + 1], non_linear=non_linear_flag))

    def __call__(self, x):
        ''' This is the forward propagation with input x '''
        for layer in self.layers:
            x = layer(x)
        return x

    def __repr__(self):
        return f"MLP ([{','.join(str(layer) for layer in self.layers)}])"

    def parameters(self):
        return [parameters for layer in self.layers for parameters in layer.parameters()]