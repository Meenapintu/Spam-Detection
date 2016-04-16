#pintu lal M
# assignment 2
import math
import random
import string
import csv
import sys
import numpy as np

random.seed(0)

def rand(a, b):
    return (b-a)*random.random() + a

def sigmoid(x):
	return 1 / (1 + np.exp(-x))
 
def dsigmoid(y):
    return y *(1.0 - y)

class NN:
    def __init__(self, input_node_num, hidden_node_num, output_node_num):
        self.LR = 0.5;
        self.MF = 0.1;
    
        self.input_node_num = input_node_num
        self.hidden_node_num = hidden_node_num
        self.output_node_num = output_node_num
       
        self.inv = np.zeros(self.input_node_num) #input node  vector  
        self.hnv = np.zeros(self.hidden_node_num) #hidden layer node vector
        self.onv = np.zeros(self.output_node_num) # output node vector 
       
        self.wi = np.zeros((self.input_node_num, self.hidden_node_num))		#input weight matrix
        self.wo = np.zeros((self.hidden_node_num, self.output_node_num))		# output weight matrix
        self.lrci = np.zeros((self.input_node_num, self.hidden_node_num))   # last recent weight change inpput matrix
        self.lrco = np.zeros((self.hidden_node_num, self.output_node_num))  #last recent weight change output matrix
     
        for i in range(self.input_node_num):
            for j in range(self.hidden_node_num):
                self.wi[i][j] = rand(-1.0, 1.0)
                
        for j in range(self.hidden_node_num):
            for k in range(self.output_node_num):
                self.wo[j][k] = rand(-1.0, 1.0)
                #self.lrco[j][k] = rand(-1.0, 1.0)

    def update(self, inputs):
        if len(inputs) != self.input_node_num-1:
            raise ValueError('error update')

        for i in range(self.input_node_num-1):
            self.inv[i] = sigmoid(inputs[i])

        for j in range(self.hidden_node_num):
            sum = 0.0
            for i in range(self.input_node_num):
                sum = sum + self.inv[i] * self.wi[i][j]
            self.hnv[j] = sigmoid(sum)

        for k in range(self.output_node_num):
            sum = 0.0
            for j in range(self.hidden_node_num):
                sum = sum + self.hnv[j] * self.wo[j][k]
            self.onv[k] = sigmoid(sum)

        return self.onv[:]
    def update_out_weight(self,output_deltas):
    	for j in range(self.hidden_node_num):
            for k in range(self.output_node_num):
                change = output_deltas[k]*self.hnv[j]
                self.wo[j][k] = self.wo[j][k] + self.LR*change + self.MF*self.lrco[j][k]
                self.lrco[j][k] = change

    def update_input_weight(self,hidden_deltas ):
    	for i in range(self.input_node_num):
            for j in range(self.hidden_node_num):
                change = hidden_deltas[j]*self.inv[i]
                self.wi[i][j] = self.wi[i][j] + self.LR*change + self.MF*self.lrci[i][j]
                self.lrci[i][j] = change

    def backPropagate(self, targets):
        if len(targets) != self.output_node_num:
            raise ValueError('error backPropagate')
 
        output_deltas = [0.0] * self.output_node_num
        for k in range(self.output_node_num):
            error = targets[k]-self.onv[k]
            output_deltas[k] = dsigmoid(self.onv[k]) * error

        hidden_deltas = [0.0] * self.hidden_node_num
        for j in range(self.hidden_node_num):
            error = 0.0
            for k in range(self.output_node_num):
                error = error + output_deltas[k]*self.wo[j][k]
            hidden_deltas[j] = dsigmoid(self.hnv[j]) * error

        self.update_out_weight(output_deltas)
        self.update_input_weight(hidden_deltas)


    def test(self, patterns):
	 out_file=open("output.csv",'wb')	
	 writer=csv.writer(out_file, dialect='excel')	 
	 writer.writerow(['Id','Label',])	 
	 count = 0
	 for p in patterns:
		if self.update(p[0])[0]>.5:
			writer.writerow([count,1])
		else :
		 	writer.writerow([count,0])
		count=count+1
	 

    def train(self, patterns, iterations=100):
        for i in range(iterations):
            for p in patterns:
                inputs = p[0]
                targets = p[1]
                self.update(inputs)
                self.backPropagate(targets)

def readfle(file):
	input_file = open(file, 'rt')
	inp = []
	try:
	    reader = csv.reader(input_file)
	    for row in reader:
		inp.append(row)
	finally:
	    input_file.close()
	
	return inp;

def run(): 
	n = NN(58, 2, 1)    
	inp = readfle('Train.csv')
	fininp = []
	finout = []
	for row in inp:
		l = []	
		i=0
		for i in range(57):
			l.append(float(row[i]))
			i=i+1	
		fininp.append(l)
		m = [int(row[57])]
		finout.append(m)
	
	pattern=[]

	for i in range(len(finout)):
		l= []
		l.append(fininp[i])
		l.append(finout[i])
		pattern.append(l)
	#print(pattern)
	n.train(pattern)
	 
	inp = readfle('TestX.csv')

	fininp = []
	for row in inp:
		l = []	
		i=0
		for i in range(57):
			l.append(float(row[i]))
			i=i+1	
		fininp.append(l)	
	pattern=[]
	for i in range(len(fininp)):
		l= []
		l.append(fininp[i])
		pattern.append(l)
	n.test(pattern)

run()
