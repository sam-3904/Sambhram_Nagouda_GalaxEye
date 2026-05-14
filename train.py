import torch.nn.functional as F
from torch.utils.data import DataLoader 
import torch
import numpy as np 

# 1. Initialize Model and Data 
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = SiameseResUNet().to(device) # Using the Residual version for better features 
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

# Enable augmentation only for training set
train_loader = DataLoader(GalaxEyeDataset('/content/dataset/train', augment=True), batch_size=8, shuffle=True)
val_loader = DataLoader(GalaxEyeDataset('/content/dataset/val', augment=False), batch_size=8)

# 2. Training Loop
for epoch in range(15): 
    model.train()
    epoch_loss = 0
    for eo, sar, mask in train_loader:
        eo, sar, mask = eo.to(device), sar.to(device), mask.to(device)
        optimizer.zero_grad()
        output = model(eo, sar)

        # SOLVING CLASS IMBALANCE: Weighted BCE + Dice
        # Weighting the positive class (1) by 15.0
        weight = (mask * 14.0) + 1.0
        bce = F.binary_cross_entropy(output, mask, weight=weight)

        inter = (output * mask).sum()
        dice = 1 - (2. * inter + 1e-6) / (output.sum() + mask.sum() + 1e-6)
        loss = bce + dice

        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()

    # 3. Validation and Metrics 
    model.eval()
    val_ious, val_f1s = [], []
    with torch.no_grad():
        for v_eo, v_sar, v_mask in val_loader:
            v_mask = v_mask.to(device)
            v_out = model(v_eo.to(device), v_sar.to(device))

            # Binary prediction for metrics
            pred = (v_out > 0.5).float()

            # IoU Calculation
            intersection = (pred * v_mask).sum()
            union = (pred + v_mask).clamp(0, 1).sum()
            val_ious.append((intersection / (union + 1e-6)).item())

            # F1 Score calculation
            tp = (pred * v_mask).sum()
            fp = (pred * (1 - v_mask)).sum()
            fn = ((1 - pred) * v_mask).sum()
            f1 = (2 * tp) / (2 * tp + fp + fn + 1e-6)
            val_f1s.append(f1.item())

    print(f"Epoch {epoch+1} | Loss: {epoch_loss/len(train_loader):.4f} | Val IoU: {np.mean(val_ious):.4f} | Val F1: {np.mean(val_f1s):.4f}")
