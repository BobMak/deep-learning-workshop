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

model = th.nn.Sequential(
    th.nn.Linear(inp_size, 256),
    th.nn.ReLU(),
    th.nn.Linear(256, 10),
    th.nn.ReLU(),
)

# training
n_epochs = 100
learning_rate = 0.001
optimizer = th.optim.SGD(model.parameters(), lr=learning_rate)
loss_fn = th.nn.CrossEntropyLoss()

for i in range(n_epochs):
    for x, y in train_loader:
        x = x.view(-1, inp_size)
        y_ = model(x)
        loss = loss_fn(y_, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    print(f"Epoch {i}: {loss.item()}")

    # testing
    n_correct = 0
    n_total = 0
    for x, y in test_loader:
        x = x.view(-1, inp_size)
        y_ = model(x)
        n_correct += th.sum(th.argmax(y_, dim=1) == y).item()
        n_total += len(y)
    p_correct = n_correct / n_total
    print(f"Testing: correct: {p_correct * 100}%")