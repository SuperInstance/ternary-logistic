//! # ternary-logistic
//!
//! Ternary logistic regression: classification with ternary `{-1, 0, +1}` features
//! and binary or multinomial (3-class) outcomes.
//!
//! Supports L2 regularization, gradient-based optimization, and probability prediction.


/// Sigmoid function: σ(z) = 1 / (1 + exp(-z))
pub fn sigmoid(z: f64) -> f64 {
    if z >= 0.0 {
        1.0 / (1.0 + (-z).exp())
    } else {
        let ez = z.exp();
        ez / (1.0 + ez)
    }
}

/// Softmax function for a vector of logits.
/// Returns probabilities that sum to 1.0.
pub fn softmax(logits: &[f64]) -> Vec<f64> {
    let max = logits.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    let exps: Vec<f64> = logits.iter().map(|&z| (z - max).exp()).collect();
    let sum: f64 = exps.iter().sum();
    exps.iter().map(|&e| e / sum).collect()
}

/// Configuration for logistic regression training.
#[derive(Debug, Clone)]
pub struct LogisticConfig {
    /// Learning rate for gradient descent.
    pub learning_rate: f64,
    /// Number of training iterations.
    pub max_iter: usize,
    /// L2 regularization strength (lambda).
    pub l2_penalty: f64,
    /// Convergence tolerance on parameter change.
    pub tol: f64,
}

impl Default for LogisticConfig {
    fn default() -> Self {
        LogisticConfig {
            learning_rate: 0.1,
            max_iter: 1000,
            l2_penalty: 0.0,
            tol: 1e-8,
        }
    }
}

/// Binary logistic regression with ternary features.
///
/// Fits P(Y=1 | X) = σ(w·x + b) where x ∈ {-1,0,+1}^d.
#[derive(Debug, Clone)]
pub struct BinaryLogisticRegression {
    /// Feature weights.
    pub weights: Vec<f64>,
    /// Bias term.
    pub bias: f64,
    config: LogisticConfig,
}

impl BinaryLogisticRegression {
    /// Create a new model with zero-initialized weights of dimension `d`.
    pub fn new(d: usize) -> Self {
        BinaryLogisticRegression {
            weights: vec![0.0; d],
            bias: 0.0,
            config: LogisticConfig::default(),
        }
    }

    /// Create with custom config.
    pub fn with_config(d: usize, config: LogisticConfig) -> Self {
        BinaryLogisticRegression {
            weights: vec![0.0; d],
            bias: 0.0,
            config,
        }
    }

    /// Compute the linear combination w·x + b.
    pub fn linear_predict(&self, x: &[i8]) -> f64 {
        assert_eq!(x.len(), self.weights.len());
        self.weights
            .iter()
            .zip(x.iter())
            .map(|(w, &xi)| w * xi as f64)
            .sum::<f64>()
            + self.bias
    }

    /// Predict P(Y=1 | x).
    pub fn predict_proba(&self, x: &[i8]) -> f64 {
        sigmoid(self.linear_predict(x))
    }

    /// Predict class label (0 or 1).
    pub fn predict(&self, x: &[i8]) -> u8 {
        if self.predict_proba(x) >= 0.5 {
            1
        } else {
            0
        }
    }

    /// Compute gradients of the log-loss.
    fn compute_gradients(&self, x: &[Vec<i8>], y: &[u8]) -> (Vec<f64>, f64) {
        let n = x.len() as f64;
        let d = self.weights.len();
        let mut grad_w = vec![0.0; d];
        let mut grad_b = 0.0;

        for (xi, &yi) in x.iter().zip(y.iter()) {
            let pred = self.predict_proba(xi);
            let error = pred - yi as f64;
            for (j, &xij) in xi.iter().enumerate() {
                grad_w[j] += error * xij as f64;
            }
            grad_b += error;
        }

        // Average and add L2 regularization
        for j in 0..d {
            grad_w[j] = grad_w[j] / n + self.config.l2_penalty * self.weights[j];
        }
        grad_b /= n;

        (grad_w, grad_b)
    }

    /// Compute the log-loss (negative log-likelihood + L2).
    pub fn log_loss(&self, x: &[Vec<i8>], y: &[u8]) -> f64 {
        let n = x.len() as f64;
        let mut loss = 0.0;
        for (xi, &yi) in x.iter().zip(y.iter()) {
            let p = self.predict_proba(xi).max(1e-15).min(1.0 - 1e-15);
            let yi_f = yi as f64;
            loss += -(yi_f * p.ln() + (1.0 - yi_f) * (1.0 - p).ln());
        }
        loss /= n;
        // L2 regularization term
        let l2: f64 = self.weights.iter().map(|w| w * w).sum::<f64>();
        loss += 0.5 * self.config.l2_penalty * l2;
        loss
    }

    /// Fit the model using gradient descent.
    pub fn fit(&mut self, x: &[Vec<i8>], y: &[u8]) {
        assert!(!x.is_empty());
        for _ in 0..self.config.max_iter {
            let (grad_w, grad_b) = self.compute_gradients(x, y);

            let lr = self.config.learning_rate;
            for j in 0..self.weights.len() {
                self.weights[j] -= lr * grad_w[j];
            }
            self.bias -= lr * grad_b;
        }
    }

    /// Accuracy on given data.
    pub fn accuracy(&self, x: &[Vec<i8>], y: &[u8]) -> f64 {
        let correct = x
            .iter()
            .zip(y.iter())
            .filter(|(xi, yi)| self.predict(xi) == **yi)
            .count();
        correct as f64 / x.len() as f64
    }
}

/// Multinomial (3-class) logistic regression with ternary features.
///
/// Predicts P(Y=k | X) via softmax over linear models, k ∈ {0, 1, 2}.
#[derive(Debug, Clone)]
pub struct TernaryLogisticRegression {
    /// Weight matrix: `weights[k][j]` is the weight for class k, feature j.
    pub weights: Vec<Vec<f64>>,
    /// Bias for each class.
    pub biases: Vec<f64>,
    config: LogisticConfig,
    n_classes: usize,
}

impl TernaryLogisticRegression {
    /// Create a new 3-class model with `d` features.
    pub fn new(d: usize) -> Self {
        let n_classes = 3;
        TernaryLogisticRegression {
            weights: vec![vec![0.0; d]; n_classes],
            biases: vec![0.0; n_classes],
            config: LogisticConfig::default(),
            n_classes,
        }
    }

    /// Create with custom config.
    pub fn with_config(d: usize, config: LogisticConfig) -> Self {
        let n_classes = 3;
        TernaryLogisticRegression {
            weights: vec![vec![0.0; d]; n_classes],
            biases: vec![0.0; n_classes],
            config,
            n_classes,
        }
    }

    /// Compute logits for each class.
    pub fn logits(&self, x: &[i8]) -> Vec<f64> {
        (0..self.n_classes)
            .map(|k| {
                self.weights[k]
                    .iter()
                    .zip(x.iter())
                    .map(|(w, &xi)| w * xi as f64)
                    .sum::<f64>()
                    + self.biases[k]
            })
            .collect()
    }

    /// Predict class probabilities via softmax.
    pub fn predict_proba(&self, x: &[i8]) -> Vec<f64> {
        softmax(&self.logits(x))
    }

    /// Predict the most likely class.
    pub fn predict(&self, x: &[i8]) -> usize {
        let logits = self.logits(x);
        logits
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
            .unwrap()
            .0
    }

    /// Fit the model using gradient descent.
    pub fn fit(&mut self, x: &[Vec<i8>], y: &[usize]) {
        assert!(!x.is_empty());
        let n = x.len();
        let d = self.weights[0].len();

        for _ in 0..self.config.max_iter {
            let mut grad_w = vec![vec![0.0; d]; self.n_classes];
            let mut grad_b = vec![0.0; self.n_classes];

            for (xi, &yi) in x.iter().zip(y.iter()) {
                let probs = self.predict_proba(xi);
                for k in 0..self.n_classes {
                    let indicator = if k == yi { 1.0 } else { 0.0 };
                    let error = probs[k] - indicator;
                    for (j, &xij) in xi.iter().enumerate() {
                        grad_w[k][j] += error * xij as f64;
                    }
                    grad_b[k] += error;
                }
            }

            let lr = self.config.learning_rate;
            let n_f = n as f64;
            for k in 0..self.n_classes {
                for j in 0..d {
                    self.weights[k][j] -= lr * (grad_w[k][j] / n_f + self.config.l2_penalty * self.weights[k][j]);
                }
                self.biases[k] -= lr * grad_b[k] / n_f;
            }
        }
    }

    /// Accuracy on given data.
    pub fn accuracy(&self, x: &[Vec<i8>], y: &[usize]) -> f64 {
        let correct = x
            .iter()
            .zip(y.iter())
            .filter(|(xi, yi)| self.predict(xi) == **yi)
            .count();
        correct as f64 / x.len() as f64
    }

    /// Cross-entropy loss.
    pub fn cross_entropy_loss(&self, x: &[Vec<i8>], y: &[usize]) -> f64 {
        let n = x.len() as f64;
        let mut loss = 0.0;
        for (xi, &yi) in x.iter().zip(y.iter()) {
            let probs = self.predict_proba(xi);
            loss -= probs[yi].max(1e-15).ln();
        }
        loss / n
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_sigmoid_bounds() {
        for z in [-100.0, -10.0, -1.0, 0.0, 1.0, 10.0, 100.0] {
            let s = sigmoid(z);
            assert!(s > 0.0 && s <= 1.0, "sigmoid({}) = {} not in (0,1)", z, s);
        }
    }

    #[test]
    fn test_sigmoid_midpoint() {
        assert!((sigmoid(0.0) - 0.5).abs() < 1e-10);
    }

    #[test]
    fn test_sigmoid_monotonic() {
        let s1 = sigmoid(-1.0);
        let s2 = sigmoid(0.0);
        let s3 = sigmoid(1.0);
        assert!(s1 < s2 && s2 < s3);
    }

    #[test]
    fn test_softmax_sums_to_one() {
        let logits = vec![1.0, 2.0, 3.0];
        let probs = softmax(&logits);
        assert_eq!(probs.len(), 3);
        let sum: f64 = probs.iter().sum();
        assert!((sum - 1.0).abs() < 1e-10, "Softmax sum = {}", sum);
        for &p in &probs {
            assert!(p > 0.0 && p < 1.0);
        }
    }

    #[test]
    fn test_softmax_all_equal() {
        let logits = vec![2.0, 2.0, 2.0];
        let probs = softmax(&logits);
        for &p in &probs {
            assert!((p - 1.0 / 3.0).abs() < 1e-10);
        }
    }

    #[test]
    fn test_softmax_numerical_stability() {
        let logits = vec![1000.0, 1001.0, 1002.0];
        let probs = softmax(&logits);
        let sum: f64 = probs.iter().sum();
        assert!((sum - 1.0).abs() < 1e-10);
        assert!((probs[2] - probs[0]) > 0.0); // 1002 > 1000
    }

    #[test]
    fn test_binary_classification_linearly_separable() {
        // Class 0: x[0] tends to be -1, Class 1: x[0] tends to be +1
        let x: Vec<Vec<i8>> = vec![
            vec![-1, -1],
            vec![-1, 0],
            vec![-1, 1],
            vec![0, -1],
            vec![1, 0],
            vec![1, 1],
            vec![1, -1],
            vec![1, 0],
        ];
        let y: Vec<u8> = vec![0, 0, 0, 0, 1, 1, 1, 1];

        let mut model = BinaryLogisticRegression::with_config(
            2,
            LogisticConfig {
                learning_rate: 0.5,
                max_iter: 2000,
                l2_penalty: 0.0,
                tol: 1e-10,
            },
        );
        model.fit(&x, &y);

        let acc = model.accuracy(&x, &y);
        assert!(acc >= 0.75, "Accuracy should be at least 75%, got {}", acc);
    }

    #[test]
    fn test_binary_predict_proba_range() {
        let model = BinaryLogisticRegression::new(2);
        let x = vec![1, -1];
        let p = model.predict_proba(&x);
        assert!(p >= 0.0 && p <= 1.0);
    }

    #[test]
    fn test_ternary_classification() {
        // 3 classes based on feature patterns
        let x: Vec<Vec<i8>> = vec![
            vec![-1, -1], // class 0
            vec![-1, 0],
            vec![0, -1],
            vec![0, 0],   // class 1
            vec![0, 1],
            vec![1, 0],
            vec![1, 1],   // class 2
            vec![1, 0],
        ];
        let y: Vec<usize> = vec![0, 0, 0, 1, 1, 1, 2, 2];

        let mut model = TernaryLogisticRegression::with_config(
            2,
            LogisticConfig {
                learning_rate: 0.5,
                max_iter: 3000,
                l2_penalty: 0.0,
                tol: 1e-10,
            },
        );
        model.fit(&x, &y);

        // Should get reasonable accuracy
        let acc = model.accuracy(&x, &y);
        assert!(acc >= 0.5, "Accuracy should be reasonable, got {}", acc);
    }

    #[test]
    fn test_ternary_predict_proba_sums_to_one() {
        let model = TernaryLogisticRegression::new(3);
        let x = vec![1, -1, 0];
        let probs = model.predict_proba(&x);
        let sum: f64 = probs.iter().sum();
        assert!((sum - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_regularization_reduces_weight_magnitude() {
        let x: Vec<Vec<i8>> = vec![
            vec![-1], vec![1], vec![-1], vec![1], vec![-1], vec![1],
        ];
        let y: Vec<u8> = vec![0, 1, 0, 1, 0, 1];

        let mut model_no_reg = BinaryLogisticRegression::with_config(
            1,
            LogisticConfig {
                learning_rate: 0.5,
                max_iter: 1000,
                l2_penalty: 0.0,
                tol: 1e-10,
            },
        );
        model_no_reg.fit(&x, &y);

        let mut model_reg = BinaryLogisticRegression::with_config(
            1,
            LogisticConfig {
                learning_rate: 0.01,
                max_iter: 2000,
                l2_penalty: 1.0,
                tol: 1e-10,
            },
        );
        model_reg.fit(&x, &y);

        let w_no_reg = model_no_reg.weights[0].abs();
        let w_reg = model_reg.weights[0].abs();
        assert!(
            w_reg < w_no_reg,
            "Regularized weights ({}) should be smaller than unregularized ({})",
            w_reg,
            w_no_reg
        );
    }

    #[test]
    fn test_log_loss_positive() {
        let model = BinaryLogisticRegression::new(2);
        let x = vec![vec![1, -1], vec![-1, 1]];
        let y = vec![1, 0];
        let loss = model.log_loss(&x, &y);
        assert!(loss > 0.0, "Log loss should be positive");
    }

    #[test]
    fn test_cross_entropy_loss_positive() {
        let model = TernaryLogisticRegression::new(2);
        let x = vec![vec![1, -1], vec![-1, 1]];
        let y = vec![0, 1];
        let loss = model.cross_entropy_loss(&x, &y);
        assert!(loss > 0.0, "Cross-entropy should be positive");
    }

    #[test]
    fn test_binary_linear_predict() {
        let mut model = BinaryLogisticRegression::new(2);
        model.weights = vec![1.0, -1.0];
        model.bias = 0.5;
        let z = model.linear_predict(&[1, -1]);
        assert!((z - 2.5).abs() < 1e-10, "w·x + b = 1*1 + (-1)*(-1) + 0.5 = 2.5");
    }
}
