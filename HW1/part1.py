import numpy as np
import matplotlib.pyplot as plt

from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error


class MyLinearRegression:
    def __init__(self, weights_init='random', add_bias = True, learning_rate=1e-5, 
        num_iterations=1_000, verbose=None, max_error=1e-5):
        ''' Linear regression model using gradient descent 

        # Arguments
            weights_init: str
                weights initialization option ['random', 'zeros']
            add_bias: bool
                whether to add bias term 
            learning_rate: float
                learning rate value for gradient descent
            num_iterations: int 
                maximum number of iterations in gradient descent
            verbose: Optional[int]
                enabling verbose output on each of the `verbose` epochs
            max_error: float
                error tolerance term, after reaching which we stop gradient descent iterations
        '''

        self.num_iterations = num_iterations
        self.learning_rate = learning_rate
        self.weights_init = weights_init
        self.add_bias = add_bias
        self.verbose = verbose
        self.max_error = max_error
        self.bias = 0
    
    def initialize_weights(self, n_features: int, seed=42) -> np.ndarray:
        ''' weights initialization function '''
        if self.weights_init == 'random':
            ################

            # YOUR CODE HERE
            rng = np.random.default_rng(seed=seed)
            weights = rng.random(size=(n_features, 1))

            ################
        elif self.weights_init == 'zeros':
            ################

            # YOUR CODE HERE
            weights = np.zeros(shape=(n_features, 1), dtype="float32")

            ################
        else:
            raise NotImplementedError
        return weights

    def cost(self, target: np.ndarray, pred: np.ndarray) -> float:
        ''' calculate cost function 
        
            # Arguments:
                target: np.array
                    array of target floating point numbers 
                pred: np.array
                    array of predicted floating points numbers
        '''
        ################

        # YOUR CODE HERE
        M = target.shape[0]
        loss = (np.sum(pred - target)**2) / (2*M)
        ################
        return loss

    def fit(self, x: np.ndarray, y: np.ndarray) -> None:
        self.weights = self.initialize_weights(x.shape[1])
        self.bias = 0

        self.history = {
            "loss": []
        }

        prev_loss = 1e9
        for epoch in range(self.num_iterations):
            ################

            # YOUR CODE HERE
            # step 1: calculate current_loss value
            preds = self.predict(x)
            loss = self.cost(y, preds)
            self.history["loss"].append(loss)
            # step 2: calculate gradient value
            grad = np.mean((preds - y) * x, axis=0)
            grad = np.expand_dims(grad, 1)

            assert grad.shape == self.weights.shape
            # step 3: update weights using learning rate and gradient value
            self.weights -= self.learning_rate * grad

            if self.add_bias:
                self.bias -= self.learning_rate * np.mean(preds - y, axis=0)

            # step 4: calculate new_loss value

            # step 5: if new_loss and current_loss difference is greater than max_error -> break;
            #         if iteration is greater than max_iterations -> break
            if (prev_loss - loss) < self.max_error:
                break

            prev_loss = loss

            if self.verbose and epoch % self.verbose == 0:
                print(f"Epoch {epoch}: loss = {loss}")
            ################
    
    def predict(self, x: np.ndarray) -> np.ndarray:
        ''' prediction function '''
        ################

        # YOUR CODE HERE
        y_hat = x.dot(self.weights) + self.bias

        ################
        return y_hat



def normal_equation(X: np.ndarray, y: np.ndarray) -> np.ndarray:
    ''' TODO: implement normal equation '''
    return np.linalg.pinv(X.T.dot(X)).dot(X.T.dot(y))



if __name__ == "__main__":
    SAVE_PATH = "HW1/plots/part1"
    # generating data samples
    x = np.linspace(-5.0, 5.0, 100)[:, np.newaxis]
    rng = np.random.default_rng(seed=42)
    y = 29 * x + 40 * rng.random(size=(100,1))

    # normalization of input data
    x /= np.max(x)

    plt.title('Data samples')
    plt.scatter(x, y)
    plt.savefig(f"{SAVE_PATH}/data_samples.png")
    plt.close()


    # Sklearn linear regression model
    sklearn_model = LinearRegression()
    sklearn_model.fit(x, y)
    y_hat_sklearn = sklearn_model.predict(x)

    plt.title('Data samples with sklearn model')
    plt.scatter(x, y)
    plt.plot(x, y_hat_sklearn, color='r')
    plt.savefig(f"{SAVE_PATH}/sklearn_model.png")
    plt.close()
    print('Sklearn MSE: ', mean_squared_error(y, y_hat_sklearn))

    # Your linear regression model
    my_model = MyLinearRegression(num_iterations=200_000, verbose=20_000, learning_rate=1e-4, max_error=1e-9)
    my_model.fit(x, y)

    plt.title("Training loss curve")
    plt.plot(my_model.history["loss"])
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.savefig(f"{SAVE_PATH}/loss_curve.png")
    plt.close()    

    y_hat = my_model.predict(x)

    plt.title('Data samples with my model')
    plt.scatter(x, y)
    plt.plot(x, y_hat, color='r')
    plt.savefig(F"{SAVE_PATH}/my_model.png")
    plt.close()
    print('My MSE: ', mean_squared_error(y, y_hat))

    # Normal equation
    weights = normal_equation(x, y)
    y_hat_normal = x @ weights

    plt.title('Data samples with normal equation')
    plt.scatter(x, y)
    plt.plot(x, y_hat_normal, color='r')
    plt.savefig(f"{SAVE_PATH}/normal_equation.png")
    plt.close()
    print('Normal equation MSE: ', mean_squared_error(y, y_hat_normal))