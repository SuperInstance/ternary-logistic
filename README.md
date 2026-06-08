# ternary-logistic

Logistic regression where every feature is {−1, 0, +1}.

## The Problem

You need to classify data with ternary features — quantized weights, ternary hash codes, balanced ternary encodings — and you need calibrated probabilities, not just labels. Standard logistic regression works, but it carries baggage: feature scaling, one-hot encoding for categoricals, learning rate schedules designed for continuous inputs. With ternary features, none of that applies.

The deeper problem: most logistic regression implementations treat the feature space as unbounded. When features are bounded to three values, the dot product `w·x` has a predictable range, the gradient has a predictable structure, and the optimizer can use larger learning rates without instability. General-purpose libraries don't exploit this.

## The Insight

With ternary features, the gradient of the log-loss with respect to weight `j` is:

```
∂L/∂wⱼ = (ŷ − y) · xⱼ / n
```

Since `xⱼ ∈ {-1, 0, +1}`, this gradient is one of three discrete values scaled by the prediction error. The optimizer moves in steps of size `lr · (ŷ − y)` or zero — there's no gradual drift from a continuous feature. This natural discretization prevents oscillation and makes convergence predictable.

For the multinomial case (3-class softmax), the same property holds: the error `pₖ − 𝟙(y=k)` is multiplied by ternary features, so gradients are bounded and learning rate tuning is forgiving.

**Three more structural properties of ternary features:**
1. No missing values — zero is a valid, meaningful value ("neutral"), not absence.
2. No scaling needed — all features have the same range. No `StandardScaler`.
3. Linear separability is common — ternary encodings tend to spread classes along clean decision boundaries.

## How It Works

Two models, one optimization strategy (gradient descent), one key difference (sigmoid vs. softmax).

### Binary: σ(w·x + b)

```
BinaryLogisticRegression
  weights: Vec<f64>      // one per feature
  bias: f64
  config: LogisticConfig

fit → for max_iter:
        compute gradients: (ŷ − y) · xⱼ for each j
        update: wⱼ -= lr · (grad + λ·wⱼ)   // L2 regularization
                b  -= lr · grad_b
```

The sigmoid is numerically stabilized with a split formulation: for `z ≥ 0`, use `1/(1+e⁻ᶻ)`; for `z < 0`, use `eᶻ/(1+eᶻ)`. This avoids overflow in `exp()` for large-magnitude logits.

### Multinomial: softmax(W·x + b)

```
TernaryLogisticRegression
  weights: Vec<Vec<f64>>  // weights[k][j]: class k, feature j
  biases: Vec<f64>        // one per class
  n_classes: 3            // fixed

fit → for max_iter:
        for each sample:
          compute probs via softmax(logits)
          errorₖ = pₖ − 𝟙(y=k)
          accumulate gradients
        update with L2 regularization
```

Softmax uses the max-subtraction trick: subtract the largest logit before exponentiating. This handles logits like `[1000, 1001, 1002]` without overflow.

## Code Example

### Binary Classification

```rust
use ternary_logistic::{BinaryLogisticRegression, LogisticConfig};

let x: Vec<Vec<i8>> = vec![
    vec![-1, -1], vec![-1, 0], vec![0, -1],  // class 0
    vec![ 1,  0], vec![ 1, 1], vec![1, -1],  // class 1
];
let y: Vec<u8> = vec![0, 0, 0, 1, 1, 1];

let mut model = BinaryLogisticRegression::with_config(2, LogisticConfig {
    learning_rate: 0.5,
    max_iter: 2000,
    l2_penalty: 0.01,
    tol: 1e-10,
});
model.fit(&x, &y);

let prob = model.predict_proba(&vec![1, 1]);   // P(Y=1|x) → near 1.0
let label = model.predict(&vec![-1, -1]);       // → 0
let loss = model.log_loss(&x, &y);             // negative log-likelihood + L2
let acc = model.accuracy(&x, &y);
```

### Multinomial (3-class) Classification

```rust
use ternary_logistic::{TernaryLogisticRegression, LogisticConfig};

let x: Vec<Vec<i8>> = vec![
    vec![-1, -1], vec![-1, 0],  // class 0
    vec![ 0,  0], vec![ 0, 1],  // class 1
    vec![ 1,  1], vec![ 1, 0],  // class 2
];
let y: Vec<usize> = vec![0, 0, 1, 1, 2, 2];

let mut model = TernaryLogisticRegression::with_config(2, LogisticConfig {
    learning_rate: 0.5,
    max_iter: 3000,
    l2_penalty: 0.0,
    tol: 1e-10,
});
model.fit(&x, &y);

let probs = model.predict_proba(&vec![1, 1]);  // [P(0), P(1), P(2)] → sum to 1.0
let class = model.predict(&vec![-1, -1]);       // → 0
let ce = model.cross_entropy_loss(&x, &y);     // cross-entropy
```

### Free Functions: sigmoid and softmax

```rust
use ternary_logistic::{sigmoid, softmax};

assert!((sigmoid(0.0) - 0.5).abs() < 1e-10);
assert!((sigmoid(-100.0) - 0.0).abs() < 1e-6);  // no overflow

let probs = softmax(&[1.0, 2.0, 3.0]);
assert!((probs.iter().sum::<f64>() - 1.0).abs() < 1e-10);
```

## Module Map

```
ternary_logistic
├── Free functions
│   ├── sigmoid(z) → f64            — numerically stable σ(z)
│   └── softmax(logits) → Vec<f64>  — probability vector, sums to 1.0
├── LogisticConfig
│   ├── learning_rate: f64          — default 0.1
│   ├── max_iter: usize             — default 1000
│   ├── l2_penalty: f64             — default 0.0
│   └── tol: f64                    — default 1e-8
├── BinaryLogisticRegression
│   ├── new(d)                      — d features, zero-initialized
│   ├── with_config(d, config)
│   ├── fit(x, y)                   — x: &[Vec<i8>], y: &[u8]
│   ├── predict(x) → u8             — 0 or 1
│   ├── predict_proba(x) → f64      — P(Y=1|x) ∈ (0,1)
│   ├── linear_predict(x) → f64     — raw w·x + b
│   ├── log_loss(x, y) → f64        — NLL + L2
│   └── accuracy(x, y) → f64
└── TernaryLogisticRegression
    ├── new(d)                      — 3 classes, d features
    ├── with_config(d, config)
    ├── fit(x, y)                   — x: &[Vec<i8>], y: &[usize]
    ├── predict(x) → usize          — argmax class
    ├── predict_proba(x) → Vec<f64> — [P(0), P(1), P(2)]
    ├── logits(x) → Vec<f64>        — raw class scores
    ├── cross_entropy_loss(x, y) → f64
    └── accuracy(x, y) → f64
```

## Design Decisions

**Full-batch gradient descent, not stochastic.** Every iteration computes the gradient over the entire dataset. This is simpler, deterministic, and fine for the typical dataset sizes where ternary features appear. For large datasets, mini-batch would converge faster per epoch but requires shuffling and batch-size tuning.

**Fixed iteration count, no early stopping.** The `tol` field exists in `LogisticConfig` but isn't checked during the fit loop. The solver runs for exactly `max_iter` iterations. This is honest — convergence monitoring adds complexity and the user can observe loss externally.

**L2 regularization on weights, not bias.** The bias gradient is averaged but not regularized. This is standard practice — regularizing the bias pushes the decision boundary toward zero regardless of class balance.

**Three classes, hardcoded.** `TernaryLogisticRegression` is fixed at 3 classes. The name is intentional: it models ternary *outcomes* with ternary *features*. A generic K-class version would be more flexible but less focused. If you need K > 3, instantiate multiple binary classifiers (one-vs-rest).

**Weights exposed as public fields.** `model.weights` and `model.bias` are `pub`. You can inspect them, set them manually, or serialize them. No getter/setter ceremony.

## Status

| Aspect | State |
|--------|-------|
| Binary classification | Stable, tested |
| Multinomial (3-class) | Stable, tested |
| L2 regularization | Working |
| L1 regularization | Not supported (use [ternary-regression](https://github.com/SuperInstance/ternary-regression) for sparse feature selection) |
| Early stopping | Config field exists but not checked |
| Mini-batch SGD | Not implemented |
| Newton's method / IRLS | Not implemented |
| K > 3 classes | Not supported |
| MSRV | Edition 2024 |

**Known limitations:** The optimizer doesn't use the convergence tolerance for early stopping — it always runs `max_iter` iterations. For the multinomial model, classes are fixed at 3. Learning rate tuning is manual; no line search or adaptive rates.

## Related Crates

- **[ternary-regression](https://github.com/SuperInstance/ternary-regression)** — Same features, continuous targets (OLS/Ridge/Lasso)
- **[ternary-em](https://github.com/SuperInstance/ternary-em)** — Discover clusters before classification
- **[ternary-pool](https://github.com/SuperInstance/ternary-pool)** — Downsample feature maps before feeding to logistic head

## License

MIT
