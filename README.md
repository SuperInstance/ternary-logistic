# ternary-logistic

Logistic regression for ternary feature spaces.

Most logistic regression libraries assume continuous features. When your features are {−1, 0, +1} — quantized neural network weights, ternary hash codes, balanced ternary encodings — you don't need feature scaling, polynomial expansion, or regularization tricks to handle the feature distribution. You need a classifier that exploits the structure.

This crate provides binary and multinomial logistic regression built specifically for `i8` ternary features: numerically stable sigmoid/softmax, gradient descent with L2 regularization, and probability outputs you can threshold yourself.

## Why This Exists

Ternary features have three beautiful properties that general-purpose logistic regression misses:

1. **No missing values**: every feature is exactly one of {−1, 0, +1}. Zero means "absent" naturally.
2. **Linear separability is common**: ternary encoding tends to spread classes along simple decision boundaries.
3. **Feature interaction is bounded**: the dot product `w·x` lives in a predictable range, making learning rate tuning easier.

The key insight: with ternary features, the gradient `∂L/∂wⱼ = (ŷ − y) · xⱼ` is one of three values scaled by the error. This means gradient updates have a natural discretization that prevents oscillation — the optimizer can't overshoot by arbitrary amounts.

## Quick Start

### Binary Classification

```rust
use ternary_logistic::{BinaryLogisticRegression, LogisticConfig};

// Ternary features: each row is a sample, each element in {-1, 0, +1}
let x = vec![
    vec![-1, -1],  // class 0
    vec![-1,  0],  // class 0
    vec![ 0, -1],  // class 0
    vec![ 1,  0],  // class 1
    vec![ 1,  1],  // class 1
    vec![ 0,  1],  // class 1
];
let y = vec![0u8, 0, 0, 1, 1, 1];

let mut model = BinaryLogisticRegression::with_config(2, LogisticConfig {
    learning_rate: 0.5,
    max_iter: 2000,
    l2_penalty: 0.01,
    tol: 1e-10,
});
model.fit(&x, &y);

// Predict
let prob = model.predict_proba(&vec![1, 1]);  // P(Y=1|x), close to 1.0
let label = model.predict(&vec![-1, -1]);      // 0
println!("Accuracy: {:.1}%", model.accuracy(&x, &y) * 100.0);
```

### Multinomial (3-class) Classification

```rust
use ternary_logistic::{TernaryLogisticRegression, LogisticConfig};

let x = vec![
    vec![-1, -1],  // class 0
    vec![ 0,  0],  // class 1
    vec![ 1,  1],  // class 2
    vec![-1,  0],  // class 0
    vec![ 0,  1],  // class 1
    vec![ 1,  0],  // class 2
];
let y = vec![0usize, 1, 2, 0, 1, 2];

let mut model = TernaryLogisticRegression::with_config(2, LogisticConfig {
    learning_rate: 0.5,
    max_iter: 3000,
    l2_penalty: 0.0,
    tol: 1e-10,
});
model.fit(&x, &y);

let probs = model.predict_proba(&vec![1, 1]);  // [P(0), P(1), P(2)]
let class = model.predict(&vec![-1, -1]);        // 0
```

## Architecture

```
BinaryLogisticRegression          TernaryLogisticRegression
┌───────────────────────┐        ┌───────────────────────────┐
│ weights: Vec<f64>     │        │ weights: Vec<Vec<f64>>    │
│ bias: f64             │        │ biases: Vec<f64>          │
│ config: LogisticConfig│        │ n_classes: 3              │
├───────────────────────┤        │ config: LogisticConfig    │
│ linear_predict(x)     │        ├───────────────────────────┤
│   = w·x + b           │        │ logits(x)                 │
│ predict_proba(x)      │        │   = [wₖ·x + bₖ for k=0,1,2]
│   = σ(w·x + b)        │        │ predict_proba(x)          │
│ predict(x)            │        │   = softmax(logits)       │
│   = 1 if p ≥ 0.5      │        │ predict(x)                │
│ fit(x, y)             │        │   = argmax(logits)        │
│   gradient descent     │        │ fit(x, y)                 │
│ log_loss(x, y)        │        │   gradient descent         │
│ accuracy(x, y)        │        │ cross_entropy_loss(x, y)  │
└───────────────────────┘        │ accuracy(x, y)            │
                                 └───────────────────────────┘
```

### Numerical Stability

Both activation functions are implemented with care:

**Sigmoid** — the naive `1 / (1 + exp(-z))` overflows for large negative z. The split formulation handles this:

```rust
pub fn sigmoid(z: f64) -> f64 {
    if z >= 0.0 {
        1.0 / (1.0 + (-z).exp())     // safe: exp is negative
    } else {
        let ez = z.exp();
        ez / (1.0 + ez)               // safe: ez is small
    }
}
```

**Softmax** — subtracts the maximum logit before exponentiating:

```rust
pub fn softmax(logits: &[f64]) -> Vec<f64> {
    let max = logits.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    let exps: Vec<f64> = logits.iter().map(|&z| (z - max).exp()).collect();
    // ... normalize
}
```

This handles logits like `[1000.0, 1001.0, 1002.0]` without overflow.

## API Reference

### Core Functions

| Function | Description |
|----------|-------------|
| `sigmoid(z)` | σ(z) = 1/(1+e⁻ᶻ), numerically stable |
| `softmax(logits)` | Probability vector summing to 1.0 |

### `LogisticConfig`

| Field | Default | Purpose |
|-------|---------|---------|
| `learning_rate` | 0.1 | Gradient step size |
| `max_iter` | 1000 | Maximum training epochs |
| `l2_penalty` | 0.0 | L2 regularization (ridge) |
| `tol` | 1e-8 | Convergence tolerance |

### `BinaryLogisticRegression`

| Method | Description |
|--------|-------------|
| `new(d)` | Zero-initialized d-dimensional model |
| `with_config(d, config)` | Custom config |
| `fit(x, y)` | Train via gradient descent |
| `predict(x)` | Class label (0 or 1) |
| `predict_proba(x)` | P(Y=1 \| x) ∈ (0, 1) |
| `linear_predict(x)` | Raw w·x + b |
| `log_loss(x, y)` | Negative log-likelihood + L2 |
| `accuracy(x, y)` | Classification accuracy |

### `TernaryLogisticRegression`

| Method | Description |
|--------|-------------|
| `new(d)` | 3-class model with d features |
| `with_config(d, config)` | Custom config |
| `fit(x, y)` | Train via gradient descent |
| `predict(x)` | Most likely class (0, 1, or 2) |
| `predict_proba(x)` | [P(0), P(1), P(2)] via softmax |
| `logits(x)` | Raw class scores |
| `cross_entropy_loss(x, y)` | Cross-entropy loss |
| `accuracy(x, y)` | Classification accuracy |

## Real-World Example: Ternary Weight Classifier

You have a dataset of quantized neural network weights and want to predict which layer they came from:

```rust
use ternary_logistic::{TernaryLogisticRegression, LogisticConfig};

// Each sample: 8 ternary features representing a weight patch
// Classes: 0=conv layer, 1=dense layer, 2=attention layer
let training_x = load_ternary_features("train_patches.txt");
let training_y = load_labels("train_labels.txt"); // Vec<usize>

let mut model = TernaryLogisticRegression::with_config(8, LogisticConfig {
    learning_rate: 0.3,
    max_iter: 5000,
    l2_penalty: 0.001,  // mild regularization
    tol: 1e-10,
});
model.fit(&training_x, &training_y);

// Evaluate
let test_x = load_ternary_features("test_patches.txt");
let test_y = load_labels("test_labels.txt");
println!("Test accuracy: {:.1}%", model.accuracy(&test_x, &test_y) * 100.0);

// Predict on a new patch
let probs = model.predict_proba(&vec![1, -1, 0, 1, 0, -1, 1, 0]);
println!("P(conv)={:.3}, P(dense)={:.3}, P(attn)={:.3}",
    probs[0], probs[1], probs[2]);
```

## Ecosystem Connections

- **`ternary-em`** — Use EM to discover cluster structure, then logistic regression for classification
- **`ternary-regression`** — Same feature space, continuous targets instead of class labels
- **`ternary-warp-block`** — Warp-level voting (majority) is a hardware analog of logistic decision boundaries

## Performance Notes

- **Per-iteration cost**: O(N × D) for binary, O(N × D × K) for multinomial. With ternary features, the inner loop multiplies by −1, 0, or +1 — potential for SIMD bit tricks.
- **Convergence**: For linearly separable ternary data, typically converges in 500-2000 iterations with learning rate 0.1-0.5.
- **L2 regularization**: Start with 0.0 and increase only if you see overfitting (training accuracy >> test accuracy). Ternary features are naturally regularized by their bounded range.
- **Learning rate**: With ternary features, the gradient magnitude is bounded by `max(|ŷ − y|) = 1`. Learning rates up to 1.0 can work; start at 0.1.

## Open Questions

- **Mini-batch gradient descent**: Currently uses full-batch. For large datasets, stochastic/mini-batch would converge faster.
- **Newton's method**: Second-order optimization (IRLS) would converge in fewer iterations but requires computing the Hessian. Worth investigating for the ternary case.
- **Multinomial >3 classes**: The API is hard-coded for 3 classes. A generic K-class version would be more flexible.
- **Feature selection**: L1 regularization would produce sparse weight vectors, effectively selecting the most informative ternary features.

## License

MIT
