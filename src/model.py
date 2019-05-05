import numpy as np
from tqdm import tqdm

INPUT_LEN = 3
NUM_EPOCHS = 200
lr = 0.01


class SimpleModel():
    """
    A simple neuron.
    """

    def __init__(self, input_len):
        # Set seed.
        np.random.seed(33)

        # Create and initialize weight matrix.
        self.w = np.random.rand(1, input_len)

    def train(self, train_x, labels, epochs):
        """
        Train the model, For simplicity assuming batch size is equal to 1.
        :param train_x: The training examples.
        :param labels: The labels of the training examples.
        :param epochs: The number of epochs
        """
        print("Started training")
        for e in tqdm(range(epochs)):
            loss = 0
            for i, x in enumerate(train_x):
                h = self.forward(x)

                loss += (h - labels[i])[0] ** 2

                self.update_parameters(h[0], labels[i], x)

            print("Epoch", e, "loss: %.7f" % loss)

    def forward(self, x):
        """
        Run inference on the model.
        :param x: The input example.
        :return: The model output.
        """
        return np.matmul(self.w, x)

    def evaluate(self, test_x, labels):
        """
        Run evaluation on the model.
        :param test_x: test examples.
        :param labels: test labels.
        :return: loss
        """
        loss = 0
        for i, e in enumerate(test_x):
            h = self.forward(e)
            loss += (labels[i] - h) ** 2

        print("Evaluation, Loss = %.5f" % loss[0])
        return loss

    def update_parameters(self, h, label, x):
        """
        Update the model .
        :param h: a predicted value
        :param label: the ground truth label.
        """
        # Get the dL/dW (gradients).
        d_w = np.multiply(2 * (h - label), x)

        # Update the weights according to loss.
        self.w = self.w - lr * d_w


if __name__ == '__main__':
    # Create simple training data.
    x = [[0, 0, 1], [1, 1, 1], [1, 0, 1], [0, 1, 1]]
    y = [0, 1, 1, 0]

    model = SimpleModel(INPUT_LEN)
    model.train(x, y, NUM_EPOCHS)

    # Create simple test data.
    t_x = [[1, 0, 0], [0, 0, 0]]
    t_y = [1, 0]

    # Run evaluation.
    model.evaluate(t_x, t_y)
