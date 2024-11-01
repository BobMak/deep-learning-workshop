import torch as th
from torchvision.datasets import MNIST
from torchvision.transforms import ToTensor
from torch.utils.data import DataLoader

# data
mnist_train = MNIST(root='data', download=True, train=True, transform=ToTensor())
mnist_test = MNIST(root='data', download=True, train=False, transform=ToTensor())
train_loader = DataLoader(mnist_train, batch_size=32, shuffle=True)
test_loader = DataLoader(mnist_test, batch_size=32, shuffle=False)

inp_size = 28 * 28

device= 'cuda'

model = th.nn.Sequential(
    th.nn.Conv2d(1, 16, 3, padding=1),
    th.nn.ReLU(),
    th.nn.Conv2d(16, 16, 3),
    th.nn.ReLU(),
    th.nn.Conv2d(16, 32, 3, padding=1),
    th.nn.ReLU(),
    th.nn.Flatten(),
    th.nn.Linear(5408, 10),
    th.nn.ReLU(),
).to(device)

# training
n_epochs = 100
learning_rate = 0.001
optimizer = th.optim.SGD(model.parameters(), lr=learning_rate)
loss_fn = th.nn.CrossEntropyLoss()

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