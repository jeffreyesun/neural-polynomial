"""
This is a proof-of-concept machine-learning algorithm.

In general, it is an algorithm for efficiently approximating the global maximum (minimum) of a function f : R^n --> R, which can only be evaluated approximately, and the function is not necessarily differentiable, or at least we have no way of calculating the derivative. But, where f has the property that, give the kth coordinate x_k, the expected value of f is positively (negatively) correlated with the distance between x_k and the kth coordinate of the global maximum. More explicitly:

Let Y1, Y{k-1}, Y{k+1}, Y_n be random variables
Let ExpVal_k(x) := E[f(Y1, Y{k-1}, x, Y{k+1}, Yn)].
Let (x1, ... xn) be the coordinates of the global maximum (minimum) of f*g, where g is some gaussian PDF.
We requite that ExpVal_k(x) be positively (negatively) correlated with |x - xk|. The more ExpVal_k(x) resembles |x - xk| the better.

In particular, this algorithm finds polynomials mapping inputs to outputs. In this case, the MNIST data set contains images of digits, together with the information which digit the image contains. These are structured as tuples (i, d), where i - the image - is a 784-tuple taking values in [0,1], and d - the digit - takes values in the digits 0..9.

The resulting polynomial maps 784 inputs to 10 outputs. The inputs represent the image, and the outputs represent the digits 0..9. The nth output is meant to represent how well the image fits the digit n. Thus, the network is trained to map an image of the digit n to a 10-tuple, such that the largest value in the 10-tuple occurs in the nth entry.

One disadvantage with this approach is that the function from inputs to outputs is complex. The function emerges from the neural network, but in general observing the network yields very little insight into the relationship between inputs and outputs.

The Network defined here helps in this regard by mapping datasets to polynomial models approximating them. The final result of this is a set of 784*10 polynomials, each mapping one input to one output.

Returning to the general concept, the function f that we are trying to find a global minimum of is the cost function associated to the degree d polynomial model whose n = 784*10*d coefficients are encoded in R^n.

The mechanism through which the models are found is simple. The cost function f is sampled at a number of candidate models, which are represented as points in R^n. Then for each variable xk, the value yk is found minimizing its distance from small-cost samples, and maximizing its distance from high-cost samples. Since the formula for yk is linear in the sample xk's (it is their average, weighted by some power of the inverse costs), this operation can be done for all coordinates at once. That is, it can be done coordinate-free by simply taking the average of the samples, weighted by their inverse costs. The result then becomes the center of the distributions from which samples are taken.

Although not a neural network, the methods are named by analogy to neural networks, in particular the project that this was forked from.
"""


import random
import numpy as np

class Network():
    
    def __init__(self, inputs, outputs, degree = 3, sections = 1):
        self.inputs = inputs
        self.outputs = outputs
        self.degree = degree + 1 #self.degree technically represents number of coefficients, which is degree + 1
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
              points_per_generation, tests_per_point, test_data=None): #Mostly a wrapper for distribute(), providing status updates and analysis for the user
        self.connections = np.zeros([self.degree, self.outputs, self.inputs])
        self.training_data = training_data
        for generation in xrange(generations):
            self.distribute(generation + 1, generations, tests_per_point, points_per_generation)
            print "Finished generation {0}".format(generation)
            if test_data:
                print "{0}. Connection distance from origin: {1}".format(
                       self.evaluate(test_data), str(sum(sum(sum(self.connections*self.connections)))))

    def distribute(self, generation, generations, tests_per_point, points): #Train the model by sampling approximations of the cost function, and taking their average, weighted by the inverse costs.
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


    def norm(self, point, first_test, tests): #Calculate the cost function by testing the model on the data set
        total_norm = np.zeros([self.outputs, 1])

        for test in xrange(first_test, first_test + tests):
            prediction = self.feedforward(self.training_data[test][0],
                                           net = point)
            total_norm += np.absolute(prediction - self.training_data[test][1])

        total_norm *= total_norm
        total_norm *= total_norm

        return total_norm


    def evaluate(self, test_data): #Generate analytical data about the model for the user
        test_results = [(np.argmax(self.feedforward(x)), y) for (x, y) in test_data]
        number_correct = sum(int(x == y) for (x, y) in test_results)
	total_tests = len(test_results)
	accuracy = float(number_correct)/total_tests
        analysis = "{0} correct out of {1}. Accuracy {2}%".format(number_correct, total_tests, accuracy*100)
        return(analysis)

    def genrandom(self): #Generate a random model, a point in the cartesian product R^d x R^o x R^i, where d is the degree + 1, o is the number of outputs, and i is the number of inputs
        return np.random.randn(self.degree, self.outputs, self.inputs)/40 #/(100+ 10*generation) + self.connections #TODO: change distribution
