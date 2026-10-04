import numpy as np
from sklearn.datasets import load_iris
import torch
import argparse

train_ratio = 0.7

X, y = load_iris(return_X_y=True)
X = X.astype(np.float64)
rng1 = np.random.default_rng(0)

train_indices = []
test_indices = []

for c in np.unique(y):
    idx = rng1.permutation(np.flatnonzero(y == c))
    
    split_point = int(len(idx) * train_ratio)
    
    train_indices.extend(idx[:split_point])
    test_indices.extend(idx[split_point:])


X_train, y_train = X[train_indices], y[train_indices]
X_test, y_test = X[test_indices], y[test_indices]

mean = np.mean(X_train, axis=0)
std = np.std(X_train, axis=0)


X_train = (X_train - mean) / std
X_test = (X_test - mean) / std

rng2 = np.random.default_rng(0)

W1 = rng2.normal(0, np.sqrt(2/4), (4, 8))
b1 = np.zeros(8)

W2 = rng2.normal(0, np.sqrt(2/(8+3)), (8, 3))
b2 = np.zeros(3)


def mlp_forward(X, y, *W_b):
    A = X
    cache = {"X": X, "y": y, "Z": [], "A": [], "W": [pair[0] for pair in W_b]}

    for W, b in W_b:
        Z = A @ W + b
        cache["Z"].append(Z)

        A = np.maximum(0, Z)
        cache["A"].append(A)

    cache["A"].pop()
    logits = cache["Z"].pop()

    shifted = logits - np.max(logits, axis=1, keepdims=True)
    log_probs = shifted - np.log(np.sum(np.exp(shifted), axis=1, keepdims=True))

    loss = -np.mean(log_probs[np.arange(len(y)), y])
    cache["probs"] = np.exp(log_probs)

    return loss, cache


def mlp_backward(cache, normed=True):
    N, C = cache["probs"].shape
    

    dZ_unnormed = (cache["probs"] - (cache["y"][:, None] == np.arange(C)))
    dZ = dZ_unnormed/N if normed else dZ_unnormed

    inputs = [cache["X"]] + cache["A"]
    Z = cache["Z"].copy()
    
    dW, db = [], []

    for x_in, W in zip(reversed(inputs), reversed(cache["W"])):
        dW.append(x_in.T @ dZ)
        db.append(np.sum(dZ, axis=0))

        if Z:
            dZ = (dZ @ W.T) * (Z.pop() > 0)

    return {"dW": dW[::-1], "db": db[::-1]}


def mlp_full(X, y, *W_b, normed=True):
    loss, cache = mlp_forward(X, y, *W_b)
    grads = mlp_backward(cache, normed=normed)
    return loss, grads


def verify_with_pytorch(normed=True):
    loss_np, grads_np = mlp_full(X_train, y_train, (W1, b1), (W2, b2), normed=normed)

    [dW1_np, dW2_np], [db1_np, db2_np] = grads_np["dW"], grads_np["db"]

    fc1 = torch.nn.Linear(4, 8, dtype=torch.float64)
    fc2 = torch.nn.Linear(8, 3, dtype=torch.float64)
    relu = torch.nn.ReLU()
    criterion = torch.nn.CrossEntropyLoss()

    with torch.no_grad():
        fc1.weight.copy_(torch.from_numpy(W1.T))
        fc1.bias.copy_(torch.from_numpy(b1))
        fc2.weight.copy_(torch.from_numpy(W2.T))
        fc2.bias.copy_(torch.from_numpy(b2))

    X_t = torch.from_numpy(X_train)
    y_t = torch.from_numpy(y_train).long()

    logits_t = fc2(relu(fc1(X_t)))
    loss_t = criterion(logits_t, y_t)

    loss_t.backward()

    dW1_t = fc1.weight.grad.numpy().T
    db1_t = fc1.bias.grad.numpy()
    dW2_t = fc2.weight.grad.numpy().T
    db2_t = fc2.bias.grad.numpy()

    error_lim = 1e-12

    diffs = {
        "Втрата": np.abs(loss_np - loss_t.item()),
        "Градієнт W1": np.max(np.abs(dW1_np - dW1_t)),
        "Градієнт b1": np.max(np.abs(db1_np - db1_t)),
        "Градієнт W2": np.max(np.abs(dW2_np - dW2_t)),
        "Градієнт b2": np.max(np.abs(db2_np - db2_t)),
    }

    print(f"NumPy loss:   {loss_np:.15f}")
    print(f"PyTorch loss: {loss_t.item():.15f}")

    results = {}
    headers = ["Величина", "Максимальна абсолютна різниця NumPy / PyTorch", "Перевірку пройдено"]
    table_rows = []

    for name, diff in diffs.items():
        is_passed = diff <= error_lim and np.isfinite(diff)
        status = "Так" if is_passed else "Ні"

        results[name] = {
            "diff": diff,
            "is_passed": is_passed,
        }
        table_rows.append([name, f"{diff:.2e}", status])

    col_widths = [
        max(len(str(row[i])) for row in [headers] + table_rows)
        for i in range(len(headers))
    ]

    print()
    print("| " + " | ".join(f"{h:<{w}}" for h, w in zip(headers, col_widths)) + " |")
    print("| " + " | ".join("-" * w for w in col_widths) + " |")
    for row in table_rows:
        print("| " + " | ".join(f"{cell:<{w}}" for cell, w in zip(row, col_widths)) + " |")

    return results


def verify_numerical_gradients(e=1e-6, normed=True):
    _, grads_np = mlp_full(X_train, y_train, (W1, b1), (W2, b2), normed=normed)

    [dW1_np, dW2_np], [db1_np, db2_np] = grads_np["dW"], grads_np["db"]

    params_to_check = {
        "W1[0, 0]": (W1, (0, 0), dW1_np[0, 0]),
        "b1[0]":    (b1, 0,      db1_np[0]),
        "W2[0, 0]": (W2, (0, 0), dW2_np[0, 0]),
        "b2[0]":    (b2, 0,      db2_np[0]),
    }

    error_lim = 1e-7
    results = {}
    headers = ["Параметр", "Градієнт backward()", "Чисельна похідна", "Абсолютна різниця", "Перевірку пройдено"]
    table_rows = []

    for name, (param, idx, g_manual) in params_to_check.items():
        start_val = param[idx]

        param[idx] = start_val + e
        l_plus, _ = mlp_forward(X_train, y_train, (W1, b1), (W2, b2))

        param[idx] = start_val - e
        l_minus, _ = mlp_forward(X_train, y_train, (W1, b1), (W2, b2))

        param[idx] = start_val

        g_num = (l_plus - l_minus) / (2 * e)
        diff = np.abs(g_num - g_manual)
        is_passed = diff <= error_lim and np.isfinite(diff)
        status = "Так" if is_passed else "Ні"

        results[name] = {
            "g_manual": g_manual,
            "g_num": g_num,
            "diff": diff,
            "is_passed": is_passed,
        }
        table_rows.append([name, f"{g_manual:+.2e}", f"{g_num:+.2e}", f"{diff:.2e}", status])

    col_widths = [
        max(len(str(row[i])) for row in [headers] + table_rows)
        for i in range(len(headers))
    ]

    print()
    print("| " + " | ".join(f"{h:<{w}}" for h, w in zip(headers, col_widths)) + " |")
    print("| " + " | ".join("-" * w for w in col_widths) + " |")
    for row in table_rows:
        print("| " + " | ".join(f"{cell:<{w}}" for cell, w in zip(row, col_widths)) + " |")

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Реалізація та перевірка зворотного поширення похибки")
    parser.add_argument(
        "--no-normed",
        dest="normed",
        action="store_false",
        default=True,
        help="Вимкнути нормування на N у зворотному проході (дослід із навмисною помилкою)",
    )
    args = parser.parse_args()

    verify_with_pytorch(normed=args.normed)
    verify_numerical_gradients(normed=args.normed)