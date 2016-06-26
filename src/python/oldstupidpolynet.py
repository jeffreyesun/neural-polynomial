import random

import numpy as np

class Network():
    
    def __init__(self, inputs, outputs, degree = 3, sections = 1):
        self.inputs = inputs
        self.outputs = outputs
        self.degree = degree
        self.sections = sections
        self.connections = np.zeros([degree + 1, outputs, inputs])
    
    

    def feedforward(self, input):
        out = np.zeros([self.outputs,1])
        for power in xrange(self.degree + 1):
            powerin = np.ones([len(input), 1])
            for p in xrange(power): powerin = input*powerin
            powerpartout = np.dot(self.connections[power], powerin)
            out += powerpartout
        return out
            
    def train(self, training_data, generations, test_data=None):
        for generation in xrange(1, generations + 1):
            self.distribute(training_data, generation, test_data)
            if test_data:
                print(str(self.evaluate(test_data)) + "/10000. Connection distance from origin: " + str(sum(sum(sum(self.connections)))))

    def distribute(self, training_data, generation, test_data):
        random.shuffle(training_data)
        distsize = min(self.inputs*self.outputs*self.sections*self.degree, len(training_data))
        for sample in xrange(distsize):
            self.flash(training_data, sample, generation, distsize, test_data)

    def flash(self, training_data, sample, generation, distsize, test_data):
        [input, output] = training_data[sample]
        trial = self.genrandom(generation)
        cost = self.cost(input, output)
        self.connections = self.connections + (trial-self.connections)/(cost*distsize*5) #Expensive
    
        if sample%500 == 0:
            print("movement: " + str(sum(sum(sum((trial-self.connections)/(cost*distsize*5))))*1000) + ", " + str(sum(sum(sum(self.connections)))) + ", " + str(cost) + self.evaluate(test_data))

    def cost(self, input, output):
        prediction = self.feedforward(input)
        difference = prediction - output
        cost = np.sum(difference*difference)/len(output)
        return cost

    def evaluate(self, test_data):
        test_results = [(np.argmax(self.feedforward(x)), y) for (x, y) in test_data]
        
        return str((sum(int(x == y) for (x, y) in test_results), sum(int(x == 0) for (x, y) in test_results), sum(int(x == 1) for (x, y) in test_results), sum(int(x == 2) for (x, y) in test_results), sum(int(x == 3) for (x, y) in test_results), sum(int(x == 4) for (x, y) in test_results), sum(int(x == 5) for (x, y) in test_results), sum(int(x == 6) for (x, y) in test_results), sum(int(x == 7) for (x, y) in test_results), sum(int(x == 8) for (x, y) in test_results), sum(int(x == 9) for (x, y) in test_results)))



    def genrandom(self, generation):
        return np.random.randn(self.degree + 1, self.outputs, self.inputs)/(2*generation) + self.connections #TODO: change distribution

