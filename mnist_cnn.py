import torch as th
from torchvision.datasets import MNIST
from torchvision.transforms import ToTensor
from torch.utils.data import DataLoader
from tqdm import trange
import argparse

# arguments
parser = argparse.ArgumentParser()
parser.add_argument('--skip-training', action='store_true')
parser.add_argument('--model-path', default='mnist_cnn')
args = parser.parse_args()

# data
mnist_train = MNIST(root='data', download=True, train=True, transform=ToTensor())
mnist_test = MNIST(root='data', download=True, train=False, transform=ToTensor())
train_loader = DataLoader(mnist_train, batch_size=32, shuffle=True)
test_loader = DataLoader(mnist_test, batch_size=32, shuffle=False)

inp_size = 28 * 28

device= 'cuda' if th.cuda.is_available() else 'cpu'

model = th.nn.Sequential(
    th.nn.Conv2d(1, 16, 3, padding=1),
    th.nn.ReLU(),
    th.nn.Conv2d(16, 16, 3),
    th.nn.ReLU(),
    th.nn.Conv2d(16, 32, 3, padding=1),
    th.nn.ReLU(),
    th.nn.Flatten(),
    th.nn.Linear(21632, 10),
    th.nn.ReLU(),
).to(device)

if not args.skip_training:
    # training
    n_epochs = 100
    learning_rate = 0.001
    optimizer = th.optim.SGD(model.parameters(), lr=learning_rate)
    loss_fn = th.nn.CrossEntropyLoss()

    try:
        for i in range(n_epochs):
            for x, y in train_loader:
                y_ = model(x.to(device))
                loss = loss_fn(y_, y.to(device))
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            print(f"Epoch {i}: {loss.item()}")

            # testing
            n_correct = 0
            n_total = 0
            for x, y in test_loader:
                y_ = model(x.to(device)).to('cpu')
                n_correct += th.sum(th.argmax(y_, dim=1) == y).item()
                n_total += len(y)
            p_correct = n_correct / n_total
            print(f"Testing: correct: {p_correct * 100}%")
    except KeyboardInterrupt:
        pass
    th.save(model.state_dict(), args.model_path)

print("Evaluating model")
model.load_state_dict(th.load(args.model_path, weights_only=True))
model.eval()
n_correct = 0
n_total = 0
for x, y in test_loader:
    y_ = model(x.to(device)).to('cpu')
    n_correct += th.sum(th.argmax(y_, dim=1) == y).item()
    n_total += len(y)
p_correct = n_correct / n_total
print(f"Testing: correct: {p_correct * 100}%")