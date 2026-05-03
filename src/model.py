import torch
import torch.nn as nn


class PurchaseNet(nn.Module):
    
    def __init__(self, input_size):
        super().__init__()
        
        self.network = nn.Sequential(
            nn.Linear(input_size, 16),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(16, 8),
            nn.ReLU(),
            nn.Linear(8, 1),
            nn.Sigmoid()
        )
    def forward(self, x):
        return self.network(x)

def train_model(model, X_train_t, y_train_t, epochs=150, lr=0.001):
    criterion = nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    losses    = []

    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        output = model(X_train_t)
        loss   = criterion(output, y_train_t)
        loss.backward()
        optimizer.step()
        losses.append(loss.item())

        if epoch % 15 == 0:
            print(f"Epoch {epoch:3d} | Loss: {loss.item():.4f}")

    return losses   

def predict(model, X_t, threshold=0.5):
    model.eval()
    with torch.no_grad():
        probs       = model(X_t)
        predictions = (probs >= threshold).int().squeeze().tolist()
    return predictions
