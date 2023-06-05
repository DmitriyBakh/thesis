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
num_params = [num_params[0], num_params[4], num_params[8], num_params[13]]
learning_rate = 1e-3

for prob in fake_probs:
    for num_param in num_params:
        plt.figure()
        plt.title(f"$\mu$, of fake probability:{prob * 10}, parameters number: {num_param}")
        plt.xlabel("Epochs")
        plt.ylabel("$Loss, log_{10}$")

        X = []
        X = list(results['train'][prob][num_param].keys())
        Y1 = [np.log10(v['loss']) for _, v in results['train'][prob][num_param].items()]
        # Y1 = [v['loss'] for _, v in results['train'][prob][num_param].items()]
        # Y1 = np.array(Y1)
        # Y1 /= np.linalg.norm(Y1)
        # Y1 = np.log10(Y1).tolist()
        
        Y2 = []
        for epoch in range(len(results['train'][prob][num_param])):
            max_t_boundary = results['train'][prob][num_param][0]['loss']
            for t in range(epoch + 1):
                max_t_boundary *= (1 - learning_rate * results['train'][prob][num_param][t]['mu_boundary'])
            Y2.append(max_t_boundary)

        # Y2 = np.array(Y2)
        # Y2 /= np.linalg.norm(Y2)
        # Y2 = np.log10(Y2).tolist()
        
        plt.plot(X, Y1, label=f'True loss')
        plt.plot(X, Y2, label=f'Maximum boundary loss')
        plt.legend()
        plt.savefig(f'mu_fake_prob-{prob*10}_num_params-{num_param}.png', dpi=200)
        # plt.show()
        plt.close()


with open('experiment_2_b.pkl', 'rb') as fp:
    results = pickle.load(fp)

plt.title("Test Trainig iterations over test accuracy")
plt.xlabel("Test accuracy, %")
plt.ylabel("Training iterations")
plt.plot(results['accuracies'], results['epoch_counts'], 'o')
plt.savefig('test_train_iterations.png', dpi=200)
plt.show()
