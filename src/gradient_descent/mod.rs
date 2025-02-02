//! Gradient descent with multiple features
//!
//! This crate is a collection of functions to perform
//! calculation of gradient descent

use std::io;
use std::io::{Error, ErrorKind};

/// # Gradient descent
///
/// - X and y are the training sets.
/// - alpha is the learning rate
/// - theta is a chosen number.
///
/// ## Implement the following matlab formula:
///
/// theta(indx,:) = theta(indx,:) -
///                 alpha * ((((temp[] * X[]) - y[]) * X(:,indx))/m);
pub fn get_thetas(
    x_mtrx: &[Vec<f32>],
    y_vec: &[f32],
    alpha: f32,
    theta: &mut [f32],
    iterations: u32,
) -> Result<Vec<f32>, io::Error> {
    let num_train = y_vec.len(); // no of training sets
    let num_feat = theta.len(); // no of feature sets

    let mut sum: f32;
    let mut h_x = vec![0.0; num_train];

    if x_mtrx.len() != num_train {
        return Err(Error::new(ErrorKind::Other, "Mis-matching training sets"));
    }

    // Convert &[Vec<f32>] to &[&[f32]]
    // to speeds up the execution by a little
    let mut x_vec_slice: Vec<&[f32]> = Vec::with_capacity(num_train);

    for x_row in x_mtrx.iter().take(num_train) {
        x_vec_slice.push(&x_row[..]);
    }

    let x_slice = &x_vec_slice[..];

    for _ in 0..iterations {
        // Shadow h_x from vec to vec slice to speeds up the execution a bit
        let h_x = &mut h_x[..];

        // Logistic Regression Hypothesis
        // hx = 1 ./ (1 + exp(-(theta' * X')));
        for (i, x_row) in x_slice.iter().enumerate().take(num_train) {
            sum = 0.0;
            for j in 0..num_feat {
                sum += theta[j] * x_row[j];
            }
            h_x[i] = 1.0 / (1.0 + std::f32::consts::E.powf(-sum));
        }

        for (j, t) in theta.iter_mut().enumerate().take(num_feat) {
            sum = 0.0;

            for i in 0..num_train {
                sum += (h_x[i] - y_vec[i]) * x_slice[i][j];
            }

            *t -= alpha * sum / num_train as f32;
        }
    }

    Ok(theta.to_vec())
}
