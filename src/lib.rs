//! Implementation of logistic regression
use std::path::Path;
// use std::time::Instant;

pub mod cost_functions;
pub mod gradient_descent;
pub mod read_data;

const ITERATIONS: u32 = 6500; // number of iteration
const ALPHA: f32 = 0.01;       // the learning speed

// Sample run of linear regression
pub fn sample_run(input_file_path: &Path) {
    let (x, y) = match read_data::get_data(input_file_path) {
        Ok((x, y)) => (x, y),
        Err(e) => panic!("{}", e.get_ref().unwrap()),
    };

    let mut theta = vec![0.0; x[0].len()]; // set theta 0 and theta 1 to 0.0

    // set the learning rate = no of features / 10
    let alpha = ALPHA;

    match gradient_descent::get_thetas(&x, &y, alpha, &mut theta, ITERATIONS) {
        Ok(theta) => {
            println!("Found thetas using Gradient Descent with learning speed {} and {} number of iterations: {:?}", alpha, ITERATIONS, &theta[1..]);
        }
        Err(e) => panic!("{}", e.get_ref().unwrap()),
    }
}
