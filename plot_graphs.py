import numpy as np
import matplotlib.pyplot as plt
import pickle

with open('results.pkl', 'rb') as fp:
    results = pickle.load(fp)

X = list(results['train'][0].keys())
Y = [v['epoch'] for _, v in results['train'][0].items()]

plt.plot(X, Y)
plt.title("Converges in overparameterization net with zero fakes samples")
plt.xlabel("Number of the parameters")
plt.ylabel("Iterations")
plt.show()


plt.title("Train")
plt.xlabel("Fake probability")
plt.ylabel("Loss")

params_num = list(results['train'][0].keys())
X = [x*10 for x in results['train'].keys()]

for param_num in params_num:
    Y = []
    for k, v in results['train'].items():
        Y.append(v[param_num]['loss'])
    plt.plot(X, Y, label=f'Number of parameters: {param_num}')    

plt.legend()
plt.show()


plt.title("Test")
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
