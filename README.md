# ternary-logistic

**Ternary Logistic Regression**

A Rust library for logistic regression classification with ternary `{-1, 0, +1}` features. Supports both binary (2-class) and multinomial (3-class) outcomes with L2 regularization and gradient-based optimization.

## Features

- **Sigmoid & Softmax**: Numerically stable implementations of the core activation functions
- **Binary Logistic Regression**: `BinaryLogisticRegression` for 2-class problems with ternary features
- **Multinomial Logistic Regression**: `TernaryLogisticRegression` for 3-class classification
- **L2 Regularization**: Configurable penalty to prevent overfitting
- **Gradient Descent**: Full gradient computation for fitting
- **Probability Prediction**: Get class probabilities, not just labels
- **Metrics**: Accuracy, log-loss, and cross-entropy computation built-in

## Quick Start

### Binary Classification

```rust
use ternary_logistic::{BinaryLogisticRegression, LogisticConfig};

let x = vec![
    vec![-1, -1],
    vec![-1,  0],
    vec![ 1,  0],
    vec![ 1,  1],
];
let y = vec![0, 0, 1, 1];

let mut model = BinaryLogisticRegression::with_config(2, LogisticConfig {
    learning_rate: 0.5,
    max_iter: 2000,
    l2_penalty: 0.01,
    tol: 1e-10,
});
model.fit(&x, &y);

let prob = model.predict_proba(&vec![1, 1]);  // P(Y=1 | x)
let label = model.predict(&vec![1, 1]);        // 0 or 1
let acc = model.accuracy(&x, &y);
```

### Multinomial (3-class) Classification

```rust
use ternary_logistic::{TernaryLogisticRegression, LogisticConfig};

let x = vec![
    vec![-1, -1],  // class 0
    vec![ 0,  0],  // class 1
    vec![ 1,  1],  // class 2
];
let y = vec![0usize, 1, 2];

let mut model = TernaryLogisticRegression::new(2);
model.fit(&x, &y);

let probs = model.predict_proba(&vec![1, 1]);  // [P(0), P(1), P(2)]
let class = model.predict(&vec![1, 1]);         // 0, 1, or 2
```

## API Overview

### Core Functions

| Function | Description |
|----------|-------------|
| `sigmoid(z)` | σ(z) = 1/(1+e⁻ᶻ), numerically stable |
| `softmax(logits)` | Probabilities summing to 1.0 |

### `BinaryLogisticRegression`

| Method | Description |
|--------|-------------|
| `new(d)` | Zero-initialized d-dimensional model |
| `with_config(d, config)` | Custom training config |
| `fit(x, y)` | Train via gradient descent |
| `predict(x)` | Class label (0 or 1) |
| `predict_proba(x)` | P(Y=1 | x) |
| `accuracy(x, y)` | Classification accuracy |
| `log_loss(x, y)` | Negative log-likelihood + L2 |

### `TernaryLogisticRegression`

| Method | Description |
|--------|-------------|
| `new(d)` | 3-class, d-dimensional model |
| `with_config(d, config)` | Custom training config |
| `fit(x, y)` | Train via gradient descent |
| `predict(x)` | Most likely class (0, 1, or 2) |
| `predict_proba(x)` | Class probabilities via softmax |
| `accuracy(x, y)` | Classification accuracy |
| `cross_entropy_loss(x, y)` | Cross-entropy loss |

### `LogisticConfig`

| Field | Default | Description |
|-------|---------|-------------|
| `learning_rate` | 0.1 | Gradient step size |
| `max_iter` | 1000 | Maximum training iterations |
| `l2_penalty` | 0.0 | L2 regularization strength |
| `tol` | 1e-8 | Convergence tolerance |

## Mathematical Details

### Binary: P(Y=1 | X) = σ(w·x + b)

Loss: ℒ = -1/N Σ [yᵢ log(ŷᵢ) + (1-yᵢ) log(1-ŷᵢ)] + λ/2 ||w||²

### Multinomial: P(Y=k | X) = exp(wₖ·x + bₖ) / Σⱼ exp(wⱼ·x + bⱼ)

Loss: ℒ = -1/N Σᵢ log P(Y=yᵢ | X=xᵢ) + λ/2 Σₖ ||wₖ||²

## Numerical Stability

- Sigmoid uses the split formulation to avoid overflow for large |z|
- Softmax subtracts max(logit) before exponentiation
- Log-loss clamps probabilities away from 0 and 1

## Testing

```bash
cargo test
```

14 tests covering:
- Sigmoid bounds, midpoint, monotonicity
- Softmax normalization, equal logits, numerical stability
- Binary classification on separable data
- Ternary classification
- Regularization reducing weight magnitudes
- Log-loss and cross-entropy positivity
- Linear prediction correctness

## License

MIT
