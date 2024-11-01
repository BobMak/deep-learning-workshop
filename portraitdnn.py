import cv2
import numpy as np
import torch as th

# do the image capturing with your webcam thing
# cam = cv2.VideoCapture(0)
# cv2.namedWindow("test")
# ret, frame = cam.read()
# cv2.imshow("test", frame)
# cv2.imwrite("test.png", frame)
# cam.release()
# cv2.destroyAllWindows()

device = 'cuda'
image_name = "test.png"
img = cv2.imread(image_name)
size_x, size_y = img.shape[1], img.shape[0]

def train():
    model = th.nn.Sequential(
        th.nn.Linear(2, 2048), th.nn.ReLU(),
        th.nn.Linear(2048, 2048), th.nn.ReLU(),
        th.nn.Linear(2048, 2048), th.nn.ReLU(),
        th.nn.Linear(2048, 2048), th.nn.ReLU(),
        th.nn.Linear(2048, 3),
    ).to(device)

    n_epochs = 100000
    batch_size = 64
    learning_rate = 0.001
    optimizer = th.optim.Adam(model.parameters(), lr=learning_rate)
    loss_fn = th.nn.MSELoss()
    for epoch in range(n_epochs):
        # sample a batch of "data"
        x = th.rand((batch_size, 2), device=device)
        pixel_idxs = (x * th.tensor([size_y-1, size_x-1], device=device).reshape(1,2)).to(th.int)
        y = th.tensor([img[idx[0], idx[1]] for idx in pixel_idxs], dtype=th.float, device=device)
        # process y into a float tensor in the range [0, 1]
        y = y / 255.0
        y_ = model(x)
        loss = loss_fn(y_, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        print(f"Epoch {epoch}: {loss.item()}")
    th.save(model, f"model.pth")

model = th.load("model.pth")

size_x //= 2
size_y //= 2
idxs_x = np.linspace(0, 1, size_x).reshape(size_x, 1)
idxs_y = np.linspace(0, 1, size_y).reshape(1, size_y)
cont_idxs = idxs_x.repeat(size_y, axis=1).reshape(-1, 1)
cont_idxs = np.hstack((cont_idxs, idxs_y.repeat(size_x, axis=0).reshape(-1, 1)))
cont_idxs = th.tensor(cont_idxs, device=device, dtype=th.float).reshape(-1, 2)
y_ = model(cont_idxs)
y_ = y_.reshape(size_x, size_y, 3)
y_ = (y_ * 255).to(th.uint8).to('cpu').numpy()
cv2.imshow("test", y_)
cv2.waitKey(0)
