#!/usr/bin/env python
# coding: utf-8
#import matplotlib.pyplot as plt
import pandas as pd


def onehot(x):
    return (2**(x-1)) if x != 0 else 0

def comparator(a):
    df= pd.DataFrame(a)
    amax = df.max(axis=0)
    aidx = df.idxmax(axis=0)
    c = []
    for i in range(len(aidx)):
        c.append(onehot(aidx[i]+1) if amax[i]!=0 else 0)
    return c

def flatten_ts(_2d_list):
    if not isinstance(_2d_list, list):
        return [_2d_list]  
    _1d_list = []
    seen = set()
    for i in _2d_list:
        if isinstance(i, list):
            for j in i:
                if j not in seen:
                    seen.add(j)
                    _1d_list.append(j)
        else:
            if i not in seen:
                seen.add(i)
                _1d_list.append(i)
    
    _1d_list.sort()
    return _1d_list


def label_index(label):
    for i in range(len(label)):
        if label[i] != []:      
            break
    return i


def layer_output_spike(input_spike, output_vector, neuron_num):
    time_stamp = [[] for _ in range(neuron_num)]
    ints = flatten_ts(input_spike)

    for i in ints:
        for j in range(neuron_num):
            if(output_vector[i] == onehot(j+1)):
                time_stamp[j].append(i)
            elif(output_vector[i+1] == onehot(j+1)):
                time_stamp[j].append(i+1)
            elif(output_vector[i+2] == onehot(j+1)):
                time_stamp[j].append(i+2)                
    return time_stamp

def get_gas(label):
    ts_gas=[]
    for i in range(len(label)):
        ts_gas.append(flatten_ts(label[i]))
    return flatten_ts(ts_gas)

def get_las(input_spikes, neuron_indices, neuron_num):
    return flatten_ts(layer_output_spike(input_spikes, neuron_indices,neuron_num))



class Synapse:
    def __init__(self, w, ts):
        self.w = w
        self.ts = ts
        self.membrane_potential = 0
        self.time_surface = 0
    def update(self, t):
        if t in self.ts:
            self.membrane_potential = self.membrane_potential + self.w*(2**8)
            self.time_surface = 0xff #self.time_surface+ 0xff
        else:
            self.membrane_potential -= self.w
            self.time_surface -= self.time_surface
        if self.membrane_potential < 0:
            self.membrane_potential = 0
    
    def run(self, time):
        self.membrane_potential = 0
        output = []
        for t in range(time):
            self.update(t)
            output.append(self.membrane_potential)
        return output
    

class Neuron:
    def __init__(self, n, synapse_weights, synapse_spikes, threshold):
        self.n = n
        self.synapses = []
        self.synapse_weights = synapse_weights
        self.time_surface = []
        for i in range(n):
            self.synapses.append(Synapse(w=self.synapse_weights[i], ts=synapse_spikes[i]))
           # self.time_surface.append(self.synapses[i].time_surface)
        self.threshold = threshold
    
    def update(self):
        sum_result = sum(self.synapses[i].membrane_potential for i in range(self.n))
        self.time_surface.append(self.synapses[i].time_surface for i in range(self.n))
        if sum_result >= self.threshold:
            return sum_result
        else:
            return 0
    
    def run(self, time):
        output = []
        for t in range(time):
            for i in range(self.n):
                self.synapses[i].update(t)
            output.append(self.update())
        return output


class Layer:
    def __init__(self, neuron_num, neuron_input_num, synapse_weights, synapse_spikes, threshold):
        self.neuron_num =  neuron_num
        self.neurons = [Neuron(neuron_input_num, synapse_weights[i], synapse_spikes, threshold[i]) for i in range(neuron_num)]
    
    def runlayer(self, time):
        output = []
        for t in range(time):
            neuron_outputs = [n.run(t) for n in self.neurons]
        max_output = max(neuron_outputs)
        neuron_index = comparator(neuron_outputs)
        layer_output = max_output 
        output.append(layer_output)
        return output[0], neuron_index
    