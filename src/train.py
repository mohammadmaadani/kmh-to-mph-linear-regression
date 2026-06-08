import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler


def main():
    # ------------------------------------------------------------------------
    # Dataset (km/h -> mph)
    # ------------------------------------------------------------------------
    X = torch.tensor(
        [[0], [10], [20], [50], [100]],
        dtype=torch.float32,
    )

    y = torch.tensor(
        [[0.0], [6.2137], [12.4274], [31.0686], [62.1371]],
        dtype=torch.float32,
    )

    # ------------------------------------------------------------------------
    # Normalization
    # ------------------------------------------------------------------------
    scaler_X = StandardScaler()
    scaler_y = StandardScaler()

    X_normalized = scaler_X.fit_transform(X)
    y_normalized = scaler_y.fit_transform(y)

    X_norm = torch.tensor(X_normalized, dtype=torch.float32)
    y_norm = torch.tensor(y_normalized, dtype=torch.float32)

    # ------------------------------------------------------------------------
    # Model
    # ------------------------------------------------------------------------
    model = nn.Linear(1, 1)

    loss_fn = nn.MSELoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)

    # ------------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------------
    epochs = 1000

    for epoch in range(epochs + 1):
        model.train()

        predictions = model(X_norm)
        loss = loss_fn(predictions, y_norm)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if epoch % 100 == 0:
            print(f"Epoch {epoch:4d} | Loss: {loss.item():.6f}")

    # ------------------------------------------------------------------------
    # Learned Parameters
    # ------------------------------------------------------------------------
    print("\nLearned Parameters")
    print("-" * 30)
    print(f"Weight: {model.weight.item():.4f}")
    print(f"Bias:   {model.bias.item():.4f}")

    # ------------------------------------------------------------------------
    # Predictions
    # ------------------------------------------------------------------------
    test_speeds = torch.tensor(
        [[30], [60], [120]],
        dtype=torch.float32,
    )

    model.eval()

    with torch.no_grad():
        test_speeds_norm = torch.tensor(
            scaler_X.transform(test_speeds),
            dtype=torch.float32,
        )

        predicted_norm = model(test_speeds_norm)
        predicted_mph = scaler_y.inverse_transform(predicted_norm)

    print("\nPredictions")
    print("-" * 30)

    for index, kmh in enumerate(test_speeds):
        kmh_value = kmh.item()
        predicted = predicted_mph[index][0]
        expected = kmh_value * 0.621371

        print(
            f"{kmh_value:.0f} km/h -> "
            f"{predicted:.4f} mph "
            f"(Expected: {expected:.4f} mph)"
        )


if __name__ == "__main__":
    main()