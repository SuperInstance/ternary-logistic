//! Standalone Quick Start: sigmoid and softmax (from README)
use ternary_logistic::{sigmoid, softmax};

fn main() {
    assert!((sigmoid(0.0) - 0.5).abs() < 1e-10);
    assert!((sigmoid(-100.0) - 0.0).abs() < 1e-6); // no overflow
    let probs = softmax(&[1.0, 2.0, 3.0]);
    assert!((probs.iter().sum::<f64>() - 1.0).abs() < 1e-10);
    println!("sigmoid(0.0) = {} ✓", sigmoid(0.0));
    println!("sigmoid(-100.0) = {:.10} (no overflow) ✓", sigmoid(-100.0));
    println!("softmax([1,2,3]) = {:?} (sums to 1.0) ✓", probs);
}
