import numpy as np

def sigmoid(x):
    return 1/(1 + np.exp(-x))

def sigmoid_derivative(x):
    return x * (1-x)

inputs = np.array([[0,0],
                   [0,1],
                   [1,0],
                   [0,0]])
outputs = np.array([[0],
                    [1],
                    [1],
                    [0]])

input_layer_neurons = 2
hidden_layer_neurons = 2
output_neurons = 1
learning_rate = 0.5
epochs = 10000

np.random.seed(42)
hidden_weights = np.random.uniform(size=(input_layer_neurons,hidden_layer_neurons))
hidden_bias = np.random.uniform(size=(1,hidden_layer_neurons))
output_weights = np.random.uniform(size=(hidden_layer_neurons,output_neurons))
output_bias = np.random.uniform(size=(1,output_neurons))

for _ in range(epochs):
    hidden_input = np.dot(inputs,hidden_weights) + hidden_bias
    hidden_output = sigmoid(hidden_input)
    final_input = np.dot(hidden_output,output_weights) + output_bias
    predicted_output = sigmoid(final_input)

error = outputs - predicted_output
d_output = error * sigmoid_derivative(predicted_output)

error_hidden = d_output.dot(output_weights.T)
d_hidden = error_hidden * sigmoid_derivative(hidden_output)

output_weights += hidden_output.T.dot(d_output) * learning_rate
output_bias += np.sum(d_output,axis=0,keepdims=True) * learning_rate
hidden_weights += inputs.T.dot(d_hidden) * learning_rate
hidden_bias += np.sum(d_hidden,axis=0,keepdims=True) * learning_rate

print("Hasil prediksi setelah training")
print(np.round(predicted_output,3))