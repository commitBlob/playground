
#%% packages
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

#%% data prep
# source: https://www.kaggle.com/datasets/rashikrahmanpritom/heart-attack-analysis-prediction-dataset
df = pd.read_csv('heart.csv')
df.head()

#%% separate independent / dependent features
X = np.array(df.loc[ :, df.columns != 'output'])
y = np.array(df['output'])

print(f"X: {X.shape}, y: {y.shape}")

#%% Train / Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=12)

#%% scale the data
scaler = StandardScaler()
X_train_scale = scaler.fit_transform(X_train)
X_test_scale = scaler.transform(X_test)

#%% network class
class CustomNeuralNetwork:
    def __init__(self, learning_rate, X_train, y_train, X_test, y_test):
        self.weights = np.random.randn(X_train.shape[1])
        self.biases = np.random.randn()
        self.learning_rate = learning_rate
        self.X_train = X_train
        self.y_train = y_train
        self.X_test = X_test
        self.y_test = y_test
        self.learning_losses_train = []
        self.learning_losses_test = []

    def activationFunction(self, x):
        # sigmoid
        return 1 / (1 + np.exp(-x))
    
    def activationDerivative(self, x):
        # sigmoid derivative
        return self.activationFunction(x) * (1 - self.activationFunction(x))

    def forwardPass(self, derivative_x):
        hidden_layer_1 = np.dot(derivative_x, self.weights) + self.biases
        activate_layer = self.activationFunction(hidden_layer_1)

        return activate_layer

    def backwardPass(self, independent_features, y_true_values):
        # calculcate gradients1
        hidden_layer_1 = np.dot(independent_features, self.weights) + self.biases
        y_predictions = self.forwardPass(independent_features)

        derivative_of_losses_to_predictions = 2 * (y_predictions - y_true_values)
        derivative_of_predictions_to_hidden_layer = self.activationDerivative(hidden_layer_1)
        derivative_of_hidden_layer_biases = 1
        derivative_of_hidden_layer_weights = independent_features

        derivatives_biases = derivative_of_losses_to_predictions * derivative_of_predictions_to_hidden_layer * derivative_of_hidden_layer_biases
        derivatives_weights = derivative_of_losses_to_predictions * derivative_of_predictions_to_hidden_layer * derivative_of_hidden_layer_weights

        return derivatives_biases, derivatives_weights
    
    def optimiser(self, derivatives_biases, derivatives_weights):
        # update weights
        self.biases = self.biases - derivatives_biases * self.learning_rate
        self.weights = self.weights - derivatives_weights * self.learning_rate

    def train(self, iterations):
        for iteration in range(iterations):
            # get random position
            random_position = np.random.randint(len(self.X_train))

            # forward pass
            y_train_true = self.y_train[random_position]
            y_train_predictictions = self.forwardPass(self.X_train[random_position])

            # calculate training losses
            losses = np.sum(np.square(y_train_predictictions - y_train_true))
            self.learning_losses_train.append(losses)

            # calculate gradients
            derivatives_biases, derivatives_weights = self.backwardPass(self.X_train[random_position], self.y_train[random_position])

            # update rates
            self.optimiser(derivatives_biases, derivatives_weights)

            # calculate errors for test data
            losses_sum = 0
            for i in range(len(self.X_test)):
                y_true_value = self.y_test[i]
                y_predictions = self.forwardPass(self.X_test[i])

                losses_sum += np.square(y_predictions - y_true_value)
            self.learning_losses_test.append(losses_sum)
        return 'training done'

#%% Hyper parameters
learning_rate = 0.2
iterations = 1000

#%% model instance and training
neural_network = CustomNeuralNetwork(learning_rate, X_train_scale, y_train, X_test_scale, y_test)

neural_network.train(iterations)

# %% check losses
sns.lineplot(x=list(range(len(neural_network.learning_losses_test))), y=neural_network.learning_losses_test)

# %% iterate over test data
total_observations = X_test_scale.shape[0]
correct_predictions = 0
y_preditctions = []
for i in range(total_observations):
    y_true = y_test[i]
    y_preditction = np.round(neural_network.forwardPass(X_test_scale[i]))
    y_preditctions.append(y_preditction)
    correct_predictions += 1 if y_true == y_preditction else 0

# %% Calculate Accuracy
correct_predictions / total_observations

# %% Baseline Classifier
from collections import Counter
Counter(y_test)

# %% Confusion Matrix
cm = confusion_matrix(y_true=y_test, y_pred=y_preditctions)


# %%
import matplotlib.pyplot as plt
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['true', 'false'])

disp.plot()
plt.show()

# %%
