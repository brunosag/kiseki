import os
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as f
import torchvision
import torchvision.transforms.v2 as transforms
from torch.utils.data import DataLoader, random_split

# Settings
SEED = 42
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
DATA_ROOT = './data'
BATCH_SIZE = 32
VAL_SIZE = 10_000
MAX_EPOCHS = 1000

# Constants
MNIST_MEAN = 0.1307
MNIST_STD = 0.3081


class CNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 4, kernel_size=3)
        self.conv2 = nn.Conv2d(4, 8, kernel_size=3)
        self.linear1 = nn.Linear(5 * 5 * 8, 32)
        self.linear2 = nn.Linear(32, 10)
        self._init_weights()

    def forward(self, x):
        x = f.max_pool2d(f.relu(self.conv1(x)), kernel_size=2)
        x = f.max_pool2d(f.relu(self.conv2(x)), kernel_size=2)
        x = torch.flatten(x, start_dim=1)
        x = f.relu(self.linear1(x))
        x = self.linear2(x)
        return x

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.Linear)):
                nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.normal_(m.bias, mean=0.0, std=0.01)


def set_seed(seed: int) -> None:
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True


def load_mnist(
    root: str | Path, batch_size: int, val_size: int
) -> tuple[DataLoader, DataLoader, DataLoader]:
    os.makedirs(root, exist_ok=True)

    transform = transforms.Compose(
        [
            transforms.ToImage(),
            transforms.ToDtype(torch.float32, scale=True),
            transforms.Normalize(mean=[MNIST_MEAN], std=[MNIST_STD]),
        ]
    )

    base_train_set = torchvision.datasets.MNIST(
        root=root, train=True, download=True, transform=transform
    )
    test_set = torchvision.datasets.MNIST(
        root=root, train=False, download=True, transform=transform
    )

    train_set, val_set = random_split(
        base_train_set, [len(base_train_set) - val_size, val_size]
    )

    train_loader = DataLoader(
        train_set, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True
    )
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, test_loader


def preload_to_vram(data_loader: DataLoader) -> tuple[torch.Tensor, torch.Tensor]:
    x_batches, y_batches = [], []

    for x_batch, y_batch in data_loader:
        x_batches.append(x_batch)
        y_batches.append(y_batch)

    x = torch.cat(x_batches).to(DEVICE)
    y = torch.cat(y_batches).to(DEVICE)

    return x, y


def train_epoch(
    x: torch.Tensor,
    y: torch.Tensor,
    batch_size: int,
    model: nn.Module,
    loss_fn: nn.Module,
    optimizer: torch.optim.Optimizer,
) -> float:
    model.train()

    num_batches = len(x) // batch_size
    num_samples = num_batches * batch_size

    indices = torch.randperm(num_samples, device=DEVICE)

    total_loss = 0
    for start_idx in range(0, num_samples, batch_size):
        end_idx = start_idx + batch_size
        batch_indices = indices[start_idx:end_idx]

        x_batch = x[batch_indices]
        y_batch = y[batch_indices]

        pred = model(x_batch)

        loss = loss_fn(pred, y_batch)
        loss.backward()
        total_loss += loss.item()

        optimizer.step()
        optimizer.zero_grad(set_to_none=True)

    avg_loss = total_loss / num_batches
    return avg_loss


def validate(
    x: torch.Tensor,
    y: torch.Tensor,
    model: nn.Module,
    loss_fn: nn.Module,
) -> tuple[float, float]:
    model.eval()

    with torch.no_grad():
        pred = model(x)
        loss = loss_fn(pred, y).item()
        correct_count = (pred.argmax(1) == y).type(torch.float).sum().item()

    accuracy = correct_count / len(x)
    return loss, accuracy


def evaluate(data_loader: DataLoader, model: nn.Module):
    model.eval()
    num_samples = len(data_loader.dataset)  # type: ignore

    correct_count = 0
    for x, y in data_loader:
        x, y = x.to(DEVICE), y.to(DEVICE)
        pred = model(x)
        correct_count += (pred.argmax(1) == y).type(torch.float).sum().item()

    accuracy = correct_count / num_samples
    return accuracy


torch.backends.cudnn.benchmark = True

set_seed(SEED)

train_loader, val_loader, test_loader = load_mnist(
    root=DATA_ROOT, batch_size=BATCH_SIZE, val_size=VAL_SIZE
)

x_train, y_train = preload_to_vram(train_loader)
x_val, y_val = preload_to_vram(val_loader)

model = CNN().to(DEVICE)
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), fused=True)

print(f"Using device: '{DEVICE}'")

for epoch in range(MAX_EPOCHS):
    avg_train_loss = train_epoch(
        x_train, y_train, BATCH_SIZE, model, loss_fn, optimizer
    )
    val_loss, val_acc = validate(x_val, y_val, model, loss_fn)
    print(
        f'Epoch {epoch + 1} |    L(train) = {avg_train_loss:.4f}    L(val) = {val_loss:.4f}    Acc = {val_acc * 100:.2f}%'
    )

test_acc = evaluate(test_loader, model)
print(f'\nFinal Test Accuracy: {100 * test_acc:.2f}%')
