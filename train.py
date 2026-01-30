from dataset import FaceDataset
from torch.utils.data import DataLoader
import torch, torch.nn as nn
from models.cnn import DeepFakeCNN

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Using device:", device)

    train_ds = FaceDataset("faces_split/train")
    val_ds   = FaceDataset("faces_split/val")

    train_loader = DataLoader(train_ds, 32, shuffle=True, num_workers=0)
    val_loader   = DataLoader(val_ds, 32, shuffle=False, num_workers=0)

    model = DeepFakeCNN().to(device)

    weights = torch.tensor([1.0, 1.5]).to(device)
    criterion = nn.CrossEntropyLoss(weight=weights)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    for epoch in range(15):
        model.train()
        correct, total = 0, 0

        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            out = model(x)
            loss = criterion(out, y)
            loss.backward()
            optimizer.step()

            correct += (out.argmax(1) == y).sum().item()
            total += y.size(0)

        train_acc = correct / total

        model.eval()
        val_correct = 0
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(device), y.to(device)
                out = model(x)
                val_correct += (out.argmax(1) == y).sum().item()

        val_acc = val_correct / len(val_ds)

        print(f"Epoch {epoch+1} | Train: {train_acc:.2%} | Val: {val_acc:.2%}")

    torch.save(model.state_dict(), "models/deepfake_cnn.pth")

if __name__ == "__main__":
    main()
