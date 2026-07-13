//! Standalone Quick Start: Binary Classification (from README)
use ternary_logistic::{BinaryLogisticRegression, LogisticConfig};

fn main() {
    let x: Vec<Vec<i8>> = vec![
        vec![-1, -1],
        vec![-1, 0],
        vec![0, -1], // class 0
        vec![1, 0],
        vec![1, 1],
        vec![1, -1], // class 1
    ];
    let y: Vec<u8> = vec![0, 0, 0, 1, 1, 1];

    let mut model = BinaryLogisticRegression::with_config(
        2,
        LogisticConfig {
            learning_rate: 0.5,
            max_iter: 2000,
            l2_penalty: 0.01,
            tol: 1e-10,
        },
    );
    model.fit(&x, &y);

    let prob = model.predict_proba(&[1, 1]);
    println!("P(Y=1 | [1,1]) = {:.6}  (README says: near 1.0)", prob);
    let label = model.predict(&[-1, -1]);
    println!("predict([-1,-1]) = {}  (README says: 0)", label);
    let loss = model.log_loss(&x, &y);
    println!(
        "log_loss = {:.6}  (README says: negative log-likelihood + L2)",
        loss
    );
    let acc = model.accuracy(&x, &y);
    println!("accuracy = {:.6}", acc);
}
