import numpy as np
import matplotlib.pyplot as plt
import pickle

with open('results_including_1-37214_v2.pkl', 'rb') as fp:
    results = pickle.load(fp)

X = list(results['train'][0].keys())
Y = [max(v) for _, v in results['train'][0].items()]

plt.plot(X, Y)
plt.title("Converges in overparameterization net with zero fakes samples")
plt.xlabel("Number of the parameters")
plt.ylabel("Iterations")
plt.show()


plt.title("Train")
plt.xlabel("Number of the parameters")
plt.ylabel("Iterations")

X = list(results['train'][0].keys())

for k, v in results['train'].items():
    Y = []
    for num_param, epoch_loss in v.items():
        Y.append(max(epoch_loss))
    plt.plot(X, Y, label=f'Fake probability: {k*10}')    

plt.legend()
plt.show()


plt.title("Test Loss")
plt.xlabel("Number of parameters")
plt.ylabel("Loss")

fake_probs = list(results['test'].keys())
X = list(results['train'][0].keys())

for fake_prob in fake_probs:
    Y = []
    for k, v in results['test'][fake_prob].items():
        Y.append(v['loss'])
    plt.plot(X, Y, label=f'Fake probability: {fake_prob * 10}')

plt.legend()
plt.show()


plt.title("Test Accuracy")
plt.xlabel("Number of parameters")
plt.ylabel("Accuracy, %")

fake_probs = list(results['test'].keys())
X = list(results['train'][0].keys())

for fake_prob in fake_probs:
    Y = []
    for k, v in results['test'][fake_prob].items():
        Y.append(v['accuracy'] * 100)
    plt.plot(X, Y, label=f'Fake probability: {fake_prob * 10}')

plt.legend()
plt.show()
