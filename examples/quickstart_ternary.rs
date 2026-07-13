//! Standalone Quick Start: Multinomial Classification (from README)
use ternary_logistic::{LogisticConfig, TernaryLogisticRegression};

fn main() {
    let x: Vec<Vec<i8>> = vec![
        vec![-1, -1],
        vec![-1, 0], // class 0
        vec![0, 0],
        vec![0, 1], // class 1
        vec![1, 1],
        vec![1, 0], // class 2
    ];
    let y: Vec<usize> = vec![0, 0, 1, 1, 2, 2];

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

    let probs = model.predict_proba(&[1, 1]);
    println!("probs = {:?}  (README says: sum to 1.0)", probs);
    let class = model.predict(&[-1, -1]);
    println!("predict([-1,-1]) = {}  (README says: 0)", class);
    let ce = model.cross_entropy_loss(&x, &y);
    println!("cross_entropy = {:.6}", ce);
}
