# MNIST CNN: train + best model + checkpoint + curves (короткая версия)

import time
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from tqdm import tqdm
import matplotlib.pyplot as plt

# -------------------- setup --------------------
torch.manual_seed(42)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", device, "| PyTorch:", torch.__version__)

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

train_ds = datasets.MNIST("./data", train=True, download=True, transform=transform)
test_ds  = datasets.MNIST("./data", train=False, download=True, transform=transform)

train_loader = DataLoader(train_ds, batch_size=64, shuffle=True)
test_loader  = DataLoader(test_ds, batch_size=1000, shuffle=False)

# -------------------- model --------------------
class MNIST_CNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1), nn.ReLU(), nn.BatchNorm2d(32),
            nn.Conv2d(32, 32, 3, padding=1), nn.ReLU(),
            nn.MaxPool2d(2), nn.Dropout2d(0.25),

            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(), nn.BatchNorm2d(64),
            nn.Conv2d(64, 64, 3, padding=1), nn.ReLU(),
            nn.MaxPool2d(2), nn.Dropout2d(0.25),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 7 * 7, 256), nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, 10)
        )

    def forward(self, x):
        return self.classifier(self.features(x))

model = MNIST_CNN().to(device)

# -------------------- train utils --------------------
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-3)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=2)

@torch.no_grad()
def evaluate(model, loader):
    model.eval()
    loss_sum, correct, total = 0.0, 0, 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        out = model(x)
        loss = criterion(out, y)
        loss_sum += loss.item() * x.size(0)
        pred = out.argmax(1)
        correct += (pred == y).sum().item()
        total += x.size(0)
    return loss_sum / total, 100.0 * correct / total

# -------------------- training loop --------------------
epochs = 15
best_acc, best_epoch = 0.0, 0

train_losses, train_accs = [], []
test_losses, test_accs = [], []
lrs = []

start = time.time()
for epoch in range(1, epochs + 1):
    model.train()
    running_loss, correct, total = 0.0, 0, 0

    pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{epochs}", leave=False)
    for x, y in pbar:
        x, y = x.to(device), y.to(device)

        optimizer.zero_grad()
        out = model(x)
        loss = criterion(out, y)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * x.size(0)
        pred = out.argmax(1)
        correct += (pred == y).sum().item()
        total += x.size(0)

        pbar.set_postfix(loss=f"{loss.item():.4f}", acc=f"{100.0*(pred==y).float().mean().item():.1f}%")

    train_loss = running_loss / total
    train_acc = 100.0 * correct / total
    test_loss, test_acc = evaluate(model, test_loader)

    prev_lr = optimizer.param_groups[0]["lr"]
    scheduler.step(test_loss)
    lr = optimizer.param_groups[0]["lr"]

    train_losses.append(train_loss); train_accs.append(train_acc)
    test_losses.append(test_loss);   test_accs.append(test_acc)
    lrs.append(lr)

    print(f"Epoch {epoch:2d} | train: {train_loss:.4f}/{train_acc:.2f}% | "
          f"test: {test_loss:.4f}/{test_acc:.2f}% | lr: {prev_lr:.1e}->{lr:.1e}")

    if test_acc > best_acc:
        best_acc, best_epoch = test_acc, epoch
        torch.save(model.state_dict(), "best_cnn_model.pth")

total_time = time.time() - start
print(f"\nDone. Best acc: {best_acc:.2f}% (epoch {best_epoch}) | time: {total_time:.1f}s")

# -------------------- save checkpoint --------------------
torch.save({
    "epoch": epochs,
    "best_epoch": best_epoch,
    "best_acc": best_acc,
    "model_state_dict": model.state_dict(),
    "optimizer_state_dict": optimizer.state_dict(),
}, "mnist_cnn_checkpoint.pth")

# -------------------- curves --------------------
plt.figure(figsize=(12, 4))

plt.subplot(1, 3, 1)
plt.plot(train_losses, label="train")
plt.plot(test_losses, label="test")
plt.title("Loss"); plt.legend(); plt.grid(True, alpha=0.3)

plt.subplot(1, 3, 2)
plt.plot(train_accs, label="train")
plt.plot(test_accs, label="test")
plt.title("Accuracy"); plt.legend(); plt.grid(True, alpha=0.3)

plt.subplot(1, 3, 3)
plt.plot(lrs)
plt.yscale("log")
plt.title("LR"); plt.grid(True, alpha=0.3, which="both")

plt.tight_layout()
plt.savefig("cnn_training_curves.png", dpi=150, bbox_inches="tight")
plt.show()
