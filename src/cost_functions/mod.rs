//! Cost function for single features
//!
//! This crate perform calculation of J(theta)

use std::f64::consts::E;
use std::io;
use std::io::{Error, ErrorKind};

/// # Cost function for multiple features (x\[1\], x\[2\], ..., x\[n\]
///
/// - X and y are the training sets.
/// - X is a 2D Vector represent multiple training sets
/// - theta is a vector that contains chosen numbers.
///
/// ## Implement the following matlab formula:
///
/// hx = 1 ./ (1 + exp(-(theta' * X')));
///
/// J = sum(-y .* log(hx') - (1 - y) .* log(1 - hx')) /
///         m + (lambda / (2 * m)) * sum(theta_without_first.^2);
///
pub fn get_cost(x: &[Vec<f64>], y: &[f64], theta: &[f64]) -> Result<f64, io::Error> {
    let num_feat = theta.len();
    let mut h_x: Vec<f64> = vec![]; // hypothesis - h(x)

    let mut sum = 0.0;
    let j_theta: f64 = 0.0; // The cost

    let num_train = if x.len() == y.len() {
        y.len()
    } else {
        return Err(Error::new(ErrorKind::Other, "Mis-match training sets"));
    };

    for i in x.iter().enumerate().take(num_train) {
        if i.1.len() != num_feat {
            panic!(
                "Missing matching number of elements in theta and X[{}]",
                i.0
            );
        }
    }

    // h(x) equation from MatLab
    // hx = 1 ./ (1 + exp(-(theta' * X')));
    for i in x.iter().enumerate().take(num_train) {
        for j in i.1.iter().enumerate().take(num_feat) {
            sum += *j.1 * theta[j.0];
        }
        h_x.push(1.0 / (1.0 + E.powf(-sum)));
    }

    // Cost function for logistical regression
    // J = sum(-y .* log(hx') - (1 - y) .* log(1 - hx')) /
    //      m + (lambda / (2 * m)) * sum(theta_without_first.^2);

    Ok(j_theta)
}
