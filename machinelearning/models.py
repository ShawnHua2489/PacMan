import nn

class PerceptronModel(object):
    def __init__(self, dimensions):
        """
        Initialize a new Perceptron instance.

        A perceptron classifies data points as either belonging to a particular
        class (+1) or not (-1). `dimensions` is the dimensionality of the data.
        For example, dimensions=2 would mean that the perceptron must classify
        2D points.
        """
        self.w = nn.Parameter(1, dimensions)

    def get_weights(self):
        """
        Return a Parameter instance with the current weights of the perceptron.
        """
        return self.w

    def run(self, x):
        """
        Calculates the score assigned by the perceptron to a data point x.

        Inputs:
            x: a node with shape (1 x dimensions)
        Returns: a node containing a single number (the score)
        """
        "*** YOUR CODE HERE ***"
        return nn.DotProduct(self.w, x)

    def get_prediction(self, x):
        """
        Calculates the predicted class for a single data point `x`.

        Returns: 1 or -1
        """
        "*** YOUR CODE HERE ***"
        return 1 if nn.as_scalar(nn.DotProduct(self.w, x)) >= 0 else -1

    def train(self, dataset):
        """
        Train the perceptron until convergence.
        """
        "*** YOUR CODE HERE ***"
        all_correct = True
        while all_correct:
            all_correct = False

            for x, y in dataset.iterate_once(batch_size=1):
                if self.get_prediction(x) != nn.as_scalar(y):
                    self.w.update(nn.Constant(nn.as_scalar(y) * x.data), 1)
                    all_correct = True





class RegressionModel(object):
    """
    A neural network model for approximating a function that maps from real
    numbers to real numbers. The network should be sufficiently large to be able
    to approximate sin(x) on the interval [-2pi, 2pi] to reasonable precision.
    """
    def __init__(self):
        # Initialize your model parameters here
        "*** YOUR CODE HERE ***"
        #self.m = nn.Parameter(1, 1)
        #self.c = nn.Parameter(1, 1)

        self.m1 = nn.Parameter(1, 128)
        self.c1 = nn.Parameter(1, 128)
        self.m2 = nn.Parameter(128, 64)
        self.c2 = nn.Parameter(1, 64)
        self.m3 = nn.Parameter(64, 1)
        self.c3 = nn.Parameter(1, 1)
        self.params = [self.m1, self.c1, self.m2, self.c2, self.m3, self.c3]

    def run(self, x):
        """
        Runs the model for a batch of examples.

        Inputs:
            x: a node with shape (batch_size x 1)
        Returns:
            A node with shape (batch_size x 1) containing predicted y-values
        """
        "*** YOUR CODE HERE ***"
        first_layer = nn.ReLU(nn.AddBias(nn.Linear(x, self.m1), self.c1))
        second_layer = nn.ReLU(nn.AddBias(nn.Linear(first_layer, self.m2), self.c2))
        third_layer = nn.AddBias(nn.Linear(second_layer, self.m3), self.c3)

        return third_layer

    def get_loss(self, x, y):
        """
        Computes the loss for a batch of examples.

        Inputs:
            x: a node with shape (batch_size x 1)
            y: a node with shape (batch_size x 1), containing the true y-values
                to be used for training
        Returns: a loss node
        """
        "*** YOUR CODE HERE ***"
        return nn.SquareLoss(self.run(x), y)

    def train(self, dataset):
        """
        Trains the model.
        """
        "*** YOUR CODE HERE ***"
        loss_int = 100
        while loss_int > 0.01:
            for x, y in dataset.iterate_once(batch_size=50):
                loss = self.get_loss(x, y)
                grad_wrt = nn.gradients(loss, [self.m1, self.c1, self.m2, self.c2,
                                               self.m3, self.c3])
                #print(grad_wrt_m, grad_wrt_c)
                self.m1.update(grad_wrt[0], -0.01)
                self.c1.update(grad_wrt[1], -0.01)
                self.m2.update(grad_wrt[2], -0.01)
                self.c2.update(grad_wrt[3], -0.01)
                self.m3.update(grad_wrt[4], -0.01)
                self.c3.update(grad_wrt[5], -0.01)
                loss_int = nn.as_scalar(loss)





class DigitClassificationModel(object):
    """
    A model for handwritten digit classification using the MNIST dataset.

    Each handwritten digit is a 28x28 pixel grayscale image, which is flattened
    into a 784-dimensional vector for the purposes of this model. Each entry in
    the vector is a floating point number between 0 and 1.

    The goal is to sort each digit into one of 10 classes (number 0 through 9).

    (See RegressionModel for more information about the APIs of different
    methods here. We recommend that you implement the RegressionModel before
    working on this part of the project.)
    """
    def __init__(self):
        # Initialize your model parameters here
        "*** YOUR CODE HERE ***"

        self.m1 = nn.Parameter(784, 512)
        self.c1 = nn.Parameter(1, 512)
        self.m2 = nn.Parameter(512, 256)
        self.c2 = nn.Parameter(1, 256)
        self.m3 = nn.Parameter(256, 128)
        self.c3 = nn.Parameter(1, 128)
        self.m4 = nn.Parameter(128, 10)
        self.c4 = nn.Parameter(1, 10)
        self.params = [self.m1, self.c1, self.m2, self.c2, self.m3, self.c3
                       ,self.m4, self.c4]


    def run(self, x):
        """
        Runs the model for a batch of examples.

        Your model should predict a node with shape (batch_size x 10),
        containing scores. Higher scores correspond to greater probability of
        the image belonging to a particular class.

        Inputs:
            x: a node with shape (batch_size x 784)
        Output:
            A node with shape (batch_size x 10) containing predicted scores
                (also called logits)
        """
        "*** YOUR CODE HERE ***"
        first_layer = nn.ReLU(nn.AddBias(nn.Linear(x, self.m1), self.c1))
        second_layer = nn.ReLU(nn.AddBias(nn.Linear(first_layer, self.m2), self.c2))
        third_layer = nn.ReLU(nn.AddBias(nn.Linear(second_layer, self.m3), self.c3))
        fourth_layer = nn.AddBias(nn.Linear(third_layer, self.m4), self.c4)

        return fourth_layer

    def get_loss(self, x, y):
        """
        Computes the loss for a batch of examples.

        The correct labels `y` are represented as a node with shape
        (batch_size x 10). Each row is a one-hot vector encoding the correct
        digit class (0-9).

        Inputs:
            x: a node with shape (batch_size x 784)
            y: a node with shape (batch_size x 10)
        Returns: a loss node
        """
        "*** YOUR CODE HERE ***"
        return nn.SoftmaxLoss(self.run(x), y)


    def train(self, dataset):
        """
        Trains the model.
        """
        "*** YOUR CODE HERE ***"

        val_acc = 0
        while val_acc < 0.982:
            for x, y in dataset.iterate_once(batch_size=100):
                loss = self.get_loss(x, y)
                grad_wrt = nn.gradients(loss, [self.m1, self.c1, self.m2, self.c2, self.m3, self.c3
                    , self.m4, self.c4])
                #print(grad_wrt_m, grad_wrt_c)
                self.m1.update(grad_wrt[0], -0.1)
                self.c1.update(grad_wrt[1], -0.1)
                self.m2.update(grad_wrt[2], -0.1)
                self.c2.update(grad_wrt[3], -0.1)
                self.m3.update(grad_wrt[4], -0.1)
                self.c3.update(grad_wrt[5], -0.1)
                self.m4.update(grad_wrt[6], -0.1)
                self.c4.update(grad_wrt[7], -0.1)
                val_acc = dataset.get_validation_accuracy()




class LanguageIDModel(object):
    """
    A model for language identification at a single-word granularity.

    (See RegressionModel for more information about the APIs of different
    methods here. We recommend that you implement the RegressionModel before
    working on this part of the project.)
    """
    def __init__(self):
        # Our dataset contains words from five different languages, and the
        # combined alphabets of the five languages contain a total of 47 unique
        # characters.
        # You can refer to self.num_chars or len(self.languages) in your code
        self.num_chars = 47
        self.languages = ["English", "Spanish", "Finnish", "Dutch", "Polish"]

        # Initialize your model parameters here
        "*** YOUR CODE HERE ***"

        '''
        self.m1 = nn.Parameter(1, 128)
        self.c1 = nn.Parameter(1, 128)
        self.m2 = nn.Parameter(128, 64)
        self.c2 = nn.Parameter(1, 64)
        self.m3 = nn.Parameter(64, 1)
        self.c3 = nn.Parameter(1, 1)
        self.params = [self.m1, self.c1, self.m2, self.c2, self.m3, self.c3]
        '''
        self.m1 = nn.Parameter(self.num_chars, 256)
        self.c1 = nn.Parameter(1, 256)
        self.x_w = nn.Parameter(self.num_chars, 256)
        self.h_w = nn.Parameter(256, 256)
        self.c2 = nn.Parameter(1, 256)
        self.m3 = nn.Parameter(256, len(self.languages))
        self.c3 = nn.Parameter(1, len(self.languages))
        self.params = [self.m1, self.c1, self.x_w, self.h_w,
                       self.c2, self.m3, self.c3]

    def run(self, xs):
        """
        Runs the model for a batch of examples.

        Although words have different lengths, our data processing guarantees
        that within a single batch, all words will be of the same length (L).

        Here `xs` will be a list of length L. Each element of `xs` will be a
        node with shape (batch_size x self.num_chars), where every row in the
        array is a one-hot vector encoding of a character. For example, if we
        have a batch of 8 three-letter words where the last word is "cat", then
        xs[1] will be a node that contains a 1 at position (7, 0). Here the
        index 7 reflects the fact that "cat" is the last word in the batch, and
        the index 0 reflects the fact that the letter "a" is the inital (0th)
        letter of our combined alphabet for this task.

        Your model should use a Recurrent Neural Network to summarize the list
        `xs` into a single node of shape (batch_size x hidden_size), for your
        choice of hidden_size. It should then calculate a node of shape
        (batch_size x 5) containing scores, where higher scores correspond to
        greater probability of the word originating from a particular language.

        Inputs:
            xs: a list with L elements (one per character), where each element
                is a node with shape (batch_size x self.num_chars)
        Returns:
            A node with shape (batch_size x 5) containing predicted scores
                (also called logits)
        """
        "*** YOUR CODE HERE ***"
        '''
        first_layer = nn.ReLU(nn.AddBias(nn.Linear(x, self.m1), self.c1))
        second_layer = nn.ReLU(nn.AddBias(nn.Linear(first_layer, self.m2), self.c2))
        third_layer = nn.AddBias(nn.Linear(second_layer, self.m3), self.c3)
    
        return third_layer
        '''
        h_i = nn.ReLU(nn.AddBias(nn.Linear(xs[0], self.m1), self.c1))
        for charactor in xs[1:]:
            h_i = nn.ReLU(nn.AddBias(nn.Add(nn.Linear(charactor, self.x_w), nn.Linear(h_i, self.h_w)), self.c2))
        output = nn.AddBias(nn.Linear(h_i, self.m3), self.c3)
        return output

    def get_loss(self, xs, y):
        """
        Computes the loss for a batch of examples.

        The correct labels `y` are represented as a node with shape
        (batch_size x 5). Each row is a one-hot vector encoding the correct
        language.

        Inputs:
            xs: a list with L elements (one per character), where each element
                is a node with shape (batch_size x self.num_chars)
            y: a node with shape (batch_size x 5)
        Returns: a loss node
        """
        "*** YOUR CODE HERE ***"
        return nn.SoftmaxLoss(self.run(xs), y)

    def train(self, dataset):
        """
        Trains the model.
        """
        "*** YOUR CODE HERE ***"
        val_acc = 0
        while val_acc < 0.85:
            for x, y in dataset.iterate_once(batch_size=64):
                loss = self.get_loss(x, y)
                grad_wrt = nn.gradients(loss, [self.m1, self.c1, self.x_w, self.h_w,
                                               self.c2, self.m3, self.c3])

                self.m1.update(grad_wrt[0], -0.01)
                self.c1.update(grad_wrt[1], -0.01)
                self.x_w.update(grad_wrt[2], -0.01)
                self.h_w.update(grad_wrt[3], -0.01)
                self.c2.update(grad_wrt[4], -0.01)
                self.m3.update(grad_wrt[5], -0.01)
                self.c3.update(grad_wrt[6], -0.01)
                val_acc = dataset.get_validation_accuracy()


