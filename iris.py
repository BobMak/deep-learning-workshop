import torch as th
from sklearn import datasets

# data
iris = datasets.load_iris()
train_test_split = 128
rnd_idxs = th.randperm(150)
iris_training_x = th.tensor(iris.data[rnd_idxs[:train_test_split]], dtype=th.float)
# one-hot encoding for categorical target variable, since it is a classification problem
iris_training_y = th.nn.functional.one_hot(th.tensor(iris.target[rnd_idxs[:train_test_split]]), num_classes=3).to(th.float)
iris_testing_x = th.tensor(iris.data[rnd_idxs[train_test_split:]], dtype=th.float)
iris_testing_y = th.nn.functional.one_hot(th.tensor(iris.target[rnd_idxs[train_test_split:]]), num_classes=3).to(th.float)

# our model
input_features = 4  # how many features to we need?
output_features= 3  # how many classes do we have?
model = th.nn.Sequential(
    th.nn.Linear(input_features, 4),
    th.nn.ReLU(),
    th.nn.Linear(4, output_features),
    th.nn.Softmax()
)

# training
n_epochs = 1000
learning_rate = 0.001
batch_size = 32
optimizer = th.optim.SGD(model.parameters(), lr=learning_rate)
loss_fn = th.nn.CrossEntropyLoss()

for i in range(n_epochs):
    for j in range(0, len(iris_training_x), batch_size):
        x = iris_training_x[j:j+batch_size]
        y = iris_training_y[j:j+batch_size]
        y_ = model(x)
        loss = loss_fn(y_, th.argmax(y, dim=1))
        optimizer.zero_grad()
        y_ = model(iris_training_x)
        loss = loss_fn(y_, iris_training_y)
        loss.backward()
        optimizer.step()
    print(f"Epoch {i}: {loss.item()}")

    # testing
    y_ = model(iris_testing_x)
    loss = loss_fn(y_, iris_testing_y)

    n_correct = th.sum(th.argmax(y_, dim=1) == th.argmax(iris_testing_y, dim=1)).item()
    p_correct = n_correct / len(iris_testing_y)
    print(f"Testing: {loss.item()}, correct: {p_correct * 100}%")





