from random    import seed 
from random    import random
from functools import reduce

import csv
import math
import operator

def init_network(n_in, n_hidden, n_out):
    network      = []
    hidden_layer = [
        {
            'weights':[random() for i in range(n_in+1)]
        } for i in range(n_hidden)
    ]
    network.append(hidden_layer)
    out_layer = [
        {
            'weights':[random() for i in range(n_hidden+1)]
        } for i in range(n_out)
    ]
    network.append(out_layer)
    return network

def dot(weights,inputs):
    return sum(map(operator.mul,weights,inputs))
def sig(dotted):
    return 1.0/(1.0 + math.exp(-dotted))

def forward_prop(net,row):
    inputs = row
    for layer in net:
        new_ins = []
        for unit in layer:
            dotted = dot(unit['weights'],inputs)
            unit['out'] = sig(dotted)
            new_ins.append(unit['out'])
        inputs = new_ins
    return inputs

# Derivative
def df_dh(h):
    return h*(1-h)

# y - label
def backprop(net,y):
    for i in reversed(range(len(net))):
        h    = net[i]
        errs = []
        if i != len(net)-1:
            for j in range(len(h)):
                err = 0.0
                for unit in net[i+1]:
                    err += unit['weights'][j] * \
                           unit['gradient'] 
                errs.append(err)
        else:
            for j in range(len(h)):
                unit = h[j]
                errs.append(y[j] - unit['out'])
        for j in range(len(h)):
            unit = h[j]
            unit['gradient'] = errs[j] * \
                               df_dh(unit['out'])


def update_weights(net, row, l_rate):
    for i in range(len(net)):
        inputs = row[:-1]
        if i != 0:
            inputs=[unit['out'] for unit in net[i-1]]
        for unit in net[i]:
            for j in range(len(inputs)):
                unit['weights'][j] += l_rate * \
                unit['gradient'] * inputs[j]
            unit['weights'][-1] += l_rate * \
            unit['gradient']

def train(net,data,l_rate,n_epoch,n_out):
    for epoch in range(n_epoch):
        sum_err = 0
        for row in data:
            outs   = forward_prop(net,row)
            expect = [0 for i in range(n_out)]
            expect[row[-1]] = 1
            sum_err += sum([(expect[i]-outs[i])**2 for i in range(len(expect))])
            backprop(net,expect)
            update_weights(net,row,l_rate)
        print('Epoch:{}, Loss: {}'.format(epoch, sum_err))



#-----------------------------------------------------

seed(1)
dataset = [[2.7810836,2.550537003,0],
	[1.465489372,2.362125076,0],
	[3.396561688,4.400293529,0],
	[1.38807019,1.850220317,0],
	[3.06407232,3.005305973,0],
	[7.627531214,2.759262235,1],
	[5.332441248,2.088626775,1],
	[6.922596716,1.77106367,1],
	[8.675418651,-0.242068655,1],
	[7.673756466,3.508563011,1]]
n_inputs = len(dataset[0]) - 1
n_outputs = len(set([row[-1] for row in dataset]))
network = init_network(n_inputs, 2, n_outputs)
train(network, dataset, 0.5, 20, n_outputs)
for layer in network:
	print(layer)
