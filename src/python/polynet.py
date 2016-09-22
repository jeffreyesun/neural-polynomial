import random
import numpy as np

class Network():
    
    def __init__(self, inputs, outputs, degree = 3, sections = 1):
        self.inputs = inputs
        self.outputs = outputs
        self.degree = degree + 1 #Indexed at 0 to include constants
        self.sections = sections
        self.connections = np.zeros([self.degree, outputs, inputs])


    def feedforward(self, input, net = None):
        if net == None: net = self.connections

        len_input = len(input)
        if len_input != self.inputs: raise ValueError("Wrong number of inputs")

        outp = np.zeros([self.outputs,1])
        inp = np.ones([self.inputs,1])

        for power in xrange(self.degree - 1):
            outp += np.dot(net[power], inp)
            inp *= input
        outp += np.dot(net[power], inp)

        return outp


    def train(self, training_data, generations,
              points_per_generation, tests_per_point, test_data=None):
        self.connections = np.zeros([self.degree, self.outputs, self.inputs])
        self.training_data = training_data
        for generation in xrange(generations):
            self.distribute(generation + 1, generations, tests_per_point, points_per_generation)
            print "Finished generation {0}".format(generation)
            if test_data:
                print "{0}. Connection distance from origin: {1}".format(
                       self.evaluate(test_data), str(sum(sum(sum(self.connections*self.connections)))))

    def distribute(self, generation, generations, tests_per_point, points):
        total_weight = np.zeros([self.outputs, 1])
        average = np.zeros([self.degree, self.outputs, self.inputs])

        tests_per_runthrough = len(self.training_data)
        points_per_runthrough = tests_per_runthrough/tests_per_point
        runsthrough = points/points_per_runthrough
        if runsthrough == 0: runsthrough = 1

        for runthrough in xrange(runsthrough):
            random.shuffle(self.training_data)
            for point in xrange(points_per_runthrough):
                first_test = point*tests_per_point

                this_point = self.genrandom()
                this_weight = 1.0/self.norm(this_point, first_test, tests_per_point)
                total_weight += this_weight
                average += this_point*this_weight

        average /= total_weight

        self.connections += (average - self.connections)/generation


    def norm(self, point, first_test, tests):
        total_norm = np.zeros([self.outputs, 1])

        for test in xrange(first_test, first_test + tests):
            prediction = self.feedforward(self.training_data[test][0],
                                           net = point)
            total_norm += np.absolute(prediction - self.training_data[test][1])

        total_norm *= total_norm
        total_norm *= total_norm

        return total_norm


    def evaluate(self, test_data):
        test_results = [(np.argmax(self.feedforward(x)), y) for (x, y) in test_data]
        number_correct = sum(int(x == y) for (x, y) in test_results)
	total_tests = len(test_results)
	accuracy = float(number_correct)/total_tests
        analysis = "{0} correct out of {1}. Accuracy {2}%".format(number_correct, total_tests, accuracy*100)
        return(analysis)

    def genrandom(self):
        return np.random.randn(self.degree, self.outputs, self.inputs)/40 #/(100+ 10*generation) + self.connections #TODO: change distribution
