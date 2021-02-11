"""
The main auto-grad engine of the entire grad system, implements backprop on DAG

"""
import numpy as np 

class Node:

    def __init__(self, value, previous_op=None, parent_nodes=[]):
        # make sure only int and float type values as used for initialization
        assert isinstance(value, (int, float))

        self.value = value                # the actual value of the node
        self.previous_op = previous_op    # the previous operation that created this node
        self.parent_nodes = parent_nodes  # stores previous nodes(parent) for this node
        self.grad = 0                     # stores derivative of the output wrt self
        self.grad_wrt = {}                # stores derivative of itself with respect to parent(previous) nodes


    def __add__(self, other_node):
        '''  for self + other_node calculation '''
        other_node = other_node if isinstance(other_node, Node) else Node(other_node)

        out = Node(value=self.value + other_node.value, previous_op='+', parent_nodes=[self, other_node])

        # derivative of output wrt `self`, z = x + y, (here x -> self), dz/dx = dx/dx + dy/dx = 1 + 0 = 1
        out.grad_wrt[self] = 1

        # derivative of output wrt `other_node`, z = x + y, (here y -> other_node), dz/dy = dx/dy + dy/dy = 0 + 1 = 1
        out.grad_wrt[other_node] = 0 
        return out


    def __radd__(self, other_node):
        ''' for other_node + self, reverse add, scenario '''
        return self.__add__(other_node)


    def __sub__(self, other_node):
        ''' for self - other_node scenario '''
        other_node = other_node if isinstance(other_node, Node) else Node(other_node)

        out = Node(self.value - other_node.value, '-', [self, other_node])

        # derivative of output wrt `self`, z = x - y, (here x -> self), dz/dx = dx/dx - dy/dx = 1 - 0 = 1
        out.grad_wrt[self] = 1           

        # derivative of output wrt `other_node`, z = x - y, (here y -> other_node), dz/dy = dx/dy - dy/dy = 0 - 1 = -1
        out.grad_wrt[other_node] = -1
        return out


    def __rsub__(self, other_node):
        ''' for reverse subtraction, other_node - self scenario, this is different from self - other_node due to negative sign '''
        other_node = other_node if isinstance(other_node, Node) else Node(other_node)

        out = Node(other_node.value - self.value, '-', [self, other_node])

        # derivative of output wrt `self`, z = y - x, (here x -> self), dz/dx = dy/dx - dx/dx = -1
        out.grad_wrt[self] = -1           

        # derivative of output wrt `other_node`, z = y - x, (here y -> other_node), dz/dy = dy/dy - dx/dy = 1
        out.grad_wrt[other_node] = 1
        return out       


    def __mul__(self, other_node):
        ''' for self * other_node scenario '''

        other_node = other_node if isinstance(other_node, Node) else Node(other_node)

        out = Node(self.value * other_node.value, '*', [self, other_node])

        # derivative of output wrt `self`, z = x*y, (here x -> self), dz/dx = y*dx/dx = y*1 = y
        out.grad_wrt[self] = other_node.value           

        # derivative of output wrt `other_node`, z = x*y, (here y -> other_node), dz/dy = x*dy/dy = x
        out.grad_wrt[other_node] = self.value
        return out 


    def __rmul__(self, other_node):
        ''' for other_node * self scenario '''
        # The order during multiplication for derivative calculation is same as __mul__       
        return self.__mul__(other_node)


    def __pow__(self, power):
        ''' for self^power scenario '''
        assert isinstance(power, (int, float)), 'power must be either float or int'

        out = Node(self.value ** power, f'^{power}', [self])

        # derivative for z = x^power, dz/dx = power * x^power-1
        out.grad_wrt[self] = power * self.value ** (power - 1)
        return out


    def __truediv__(self, other_node):
        ''' for self/other_node scenario '''
        other_node = other_node if isinstance(other_node, Node) else Node(other_node)

        out = Node(self.value / other_node.value, '/', [self, other_node])

        # derivative of output wrt `self`, z = x/y, (here x -> self), dz/dx = (dx/dx)*1/y = 1/y
        out.grad_wrt[self] = 1 / other_node.value           

        # derivative of output wrt `other_node`, z = x/y, (here y -> other_node), dz/dy = x*d[y^-1]/dy = -1*x*y^-2
        out.grad_wrt[other_node] = - self.value * (other_node.value ** -2)
        return out 


    def __rtruediv__(self, other_node):
        ''' for other_node/self scenario '''
        other_node = other_node if isinstance(other_node, Node) else Node(other_node)

        out = Node(other_node.value / self.value, '/', [self, other_node])

        # derivative of output wrt `self`, z = y/x, (here x -> self), dz/dx = y*(d[x^-1]/dx) = y*-1*x^-2
        out.grad_wrt[self] = -other_node.value * (self.value ** -2)          

        # derivative of output wrt `other_node`, z = y/x, (here y -> other_node), dz/dy = (1/x)*(dy/dy) = 1/x
        out.grad_wrt[other_node] = 1 / self.value
        return out 


    def __neg__(self):
        ''' for -self scenario '''
        return self.__mul__(-1)


    def relu(self):
        ''' ReLU (Rectified Linear Unit) is used as activation fucntion and allows only positive values to pass through it '''

        clip_neg_vals = max(0, self.value)  # only use values > 0

        out = Node(clip_neg_vals, 'ReLU', [self])

        # derivative of relu function will be 0 or 1
        out.grad_wrt[self] = int(self.value > 0)
        return out


    def sigmoid(self):
        ''' for using signoid activation function on node '''
        f = 1 / (1 + np.exp(-self.value))
        out = Node(f, 'sigmoid', [self])
        out.grad_wrt[self] = f * (1 - f)
        return out

    def __repr__(self):
        return f'Node(value={self.value:.2f}, grad={self.grad:.2f}), prev_op={self.prev_op})'


    def backward(self):
        ''' To calculate derivate of `outout` wrt to all nodes in DAG.

            The optimized approach for calculating derivatives with mapping of type R^n --> R^m , where n >> m, is reverse automatic differentiation, which utilizes the chain-rule (https://en.wikipedia.org/wiki/Automatic_differentiation#The_chain_rule,_forward_and_reverse_accumulation).

            This is exactly what is implemented in PyTorch, Tensorflow and other NN libraries.

            We move in reverse order from output towards input.

        '''
        topology_seq = []
        visited = set()

        def build_topology(vertex):
            if vertex not in visited:
                visited.add(vertex)

                for prev_node in vertex.parent_nodes:
                    build_topology(prev_node)
                topology_seq.append(visited)

        def get_gradients(node):
            ''' To compute the derivative of output with respect to each parent node, using chain rule.
                d_output/d_parent = d_output/d_node * d_node/d_parent
            '''
            for parent in node.parent_nodes:
                dOutput_dNode = node.grad
                dNode_dParent = node.grad_wrt[parent]
                parent.grad += dOutput_dNode * dNode_dParent

        # build topology
        build_topology(self)

        self.grad = 1

        # use the reversed topology
        for node in reversed(topology_seq):
            get_gradients(node)
