import numpy as np

class NN:
    def __init__(self, input_size, hidden_size=8, output_size=1, learning_rate=0.01):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.learning_rate = learning_rate

        # Xavier-style initialization
        # Helps prevent sigmoid neurons from immediately saturating
        self.W1 = np.random.randn(input_size, hidden_size) * np.sqrt(1 / input_size)
        self.b1 = np.zeros((1, hidden_size))

        self.W2 = np.random.randn(hidden_size, output_size) * np.sqrt(1 / hidden_size)
        self.b2 = np.zeros((1, output_size))

    def sigmoid(self, z):
        # Prevent numerical overflow
        z = np.clip(z, -500, 500)
        return 1 / (1 + np.exp(-z))

    def sigmoid_derivative(self, a):
        return a * (1 - a)

    def forward(self, X):
        # Hidden layer
        self.z1 = X @ self.W1 + self.b1
        self.a1 = self.sigmoid(self.z1)

        # Output layer
        self.z2 = self.a1 @ self.W2 + self.b2
        self.a2 = self.sigmoid(self.z2)

        return self.a2

    def binary_cross_entropy(self, y, output):
        epsilon = 1e-10
        output = np.clip(output, epsilon, 1 - epsilon)

        return -np.mean(
            y * np.log(output)
            + (1 - y) * np.log(1 - output)
        )

    def backward(self, X, y):
        n = X.shape[0]

        # With sigmoid + binary cross entropy:
        # derivative simplifies to output - y
        output_delta = self.a2 - y

        # Hidden layer error
        hidden_error = output_delta @ self.W2.T
        hidden_delta = hidden_error * self.sigmoid_derivative(self.a1)

        # Gradients
        dW2 = self.a1.T @ output_delta / n
        db2 = np.mean(output_delta, axis=0, keepdims=True)

        dW1 = X.T @ hidden_delta / n
        db1 = np.mean(hidden_delta, axis=0, keepdims=True)

        # Gradient descent
        self.W2 -= self.learning_rate * dW2
        self.b2 -= self.learning_rate * db2

        self.W1 -= self.learning_rate * dW1
        self.b1 -= self.learning_rate * db1

    def train(self, X, y, epochs=500, verbose=False):
        losses = []

        for epoch in range(epochs):
            output = self.forward(X)

            loss = self.binary_cross_entropy(y, output)
            losses.append(loss)

            self.backward(X, y)

            if verbose and epoch % 50 == 0:
                print(f"Epoch {epoch}: loss = {loss:.4f}")

        return losses

    def predict_proba(self, X):
        return self.forward(X)

    def predict(self, X, threshold=0.5):
        probabilities = self.predict_proba(X)
        return (probabilities >= threshold).astype(int)