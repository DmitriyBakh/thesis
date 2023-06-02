import numpy as np
import matplotlib.pyplot as plt
import pickle

with open('results.pkl', 'rb') as fp:
    results = pickle.load(fp)

X = list(results['train'][0].keys())
Y = [max(v) for _, v in results['train'][0].items()]

plt.plot(X, Y)
plt.title("Converges in overparameterization net with zero fakes samples")
plt.xlabel("Number of the parameters")
plt.ylabel("Iterations")
plt.savefig('converges_num_params-iteraions.png', dpi=200)
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
plt.savefig('train_num_params-iterations.png', dpi=200)
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
plt.savefig('test_loss.png', dpi=200)
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
plt.savefig('test_accuracy.png', dpi=200)
plt.show()


fake_probs = list(results['train'].keys())
num_params = list(results['train'][0].keys())
num_params = [num_params[0], num_params[7], num_params[13]]

for prob in fake_probs:
    for num_param in num_params:
        plt.title(f"Mu, of fake probability:{prob * 10}, parameters number: {num_param}")
        plt.xlabel("Mu")
        plt.ylabel("Loss")

        X, Y = [], []
        for epoch in range(len(results['train'][prob][num_param])):
            X.append(results['train'][prob][num_param][epoch]['mu_boundary'])
            Y.append(results['train'][prob][num_param][epoch]['loss'])
        
        plt.plot(X, Y)
        plt.savefig(f'mu_fake_prob-{prob*10}_num_params-{num_param}.png', dpi=200)
        plt.show()


with open('experiment_2_b.pkl', 'rb') as fp:
    results = pickle.load(fp)

plt.title("Test Trainig iterations over test accuracy")
plt.xlabel("Test accuracy, %")
plt.ylabel("Training iterations")
plt.plot(results['accuracies'], results['epoch_counts'], 'o')
plt.savefig('test_train_iterations.png', dpi=200)
plt.show()
