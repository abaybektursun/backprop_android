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


def update_weight:
    pass
#-----------------------------------------------------
seed(1)

net = [[{'out': 0.7105668883115941, 'weights': [0.13436424411240122, 0.8474337369372327, 0.763774618976614]}],[{'out': 0.6213859615555266, 'weights': [0.2550690257394217, 0.49543508709194095]}, {'out': 0.6573693455986976, 'weights': [0.4494910647887381, 0.651592972722763]}]]

y = [0,1]
backprop(net,y)

for h in net:
    print(h)

