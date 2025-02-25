//! # Read data from file and store the value into vectors
//!
use std::fs::File;
use std::io::{self, BufRead, Error, ErrorKind};
use std::path::Path;

type DoubleVecF32 = Vec<Vec<f32>>;

pub fn get_data(path: &Path) -> Result<(DoubleVecF32, Vec<f32>), io::Error> {
    let lines = match File::open(path) {
        Ok(file) => io::BufReader::new(file).lines(),
        Err(ref error) if error.kind() == ErrorKind::NotFound => {
            return Err(Error::new(ErrorKind::NotFound, "File not found"));
        }
        Err(error) if error.kind() == ErrorKind::PermissionDenied => {
            return Err(Error::new(ErrorKind::PermissionDenied, "Permission denied"));
        }
        Err(_) => {
            return Err(Error::new(ErrorKind::Other, "Can not open file."));
        }
    };

    let mut y: Vec<f32> = vec![];
    let mut x_lines: Vec<String> = vec![];

    // Read the file line by line
    // split each line by the last ',' into two vectors of x_lines and y
    for line in lines {
        if let Some(data_tuple) = line.unwrap().rsplit_once(',') {
            x_lines.push(data_tuple.0.to_string());
            y.push(data_tuple.1.parse::<f32>().expect("Failed"));
        }
    }

    let mut features_str_vec: Vec<Vec<&str>> = vec![];

    for x_each_line in x_lines.iter() {
        features_str_vec.push(x_each_line.split(',').collect::<Vec<&str>>());
    }

    let mut x: DoubleVecF32 = vec![];

    for feature_str in features_str_vec.iter() {
        let mut feature_vec_f32: Vec<f32> = vec![1.0];  // x0 is 1.0

        for feature_f32 in feature_str.iter().map(|e| e.to_string().parse::<f32>()) {
            feature_vec_f32.push(feature_f32.unwrap());
        }
        x.push(feature_vec_f32.to_vec());
    }

    Ok((x, y))
}

pub fn get_data_flat_x(path: &Path) -> Result<(Vec<f32>, usize, Vec<f32>), io::Error> {
    let lines = match File::open(path) {
        Ok(file) => io::BufReader::new(file).lines(),
        Err(ref error) if error.kind() == ErrorKind::NotFound => {
            return Err(Error::new(ErrorKind::NotFound, "File not found"));
        }
        Err(error) if error.kind() == ErrorKind::PermissionDenied => {
            return Err(Error::new(ErrorKind::PermissionDenied, "Permission denied"));
        }
        Err(_) => {
            return Err(Error::new(ErrorKind::Other, "Can not open file."));
        }
    };

    let mut flatten_x: Vec<f32> = vec![];
    let mut y: Vec<f32> = vec![];

    // Read the file line by line
    // split each line by the last ',' into two vectors of flatten_x and y
    // parse string into f32
    for line in lines {
        if let Some(data_tuple) = line.unwrap().rsplit_once(',') {
            // add X_0
            flatten_x = [&flatten_x, &vec![1.0][..]].concat();

            // add rest of line of X
            flatten_x = [
                &flatten_x,
                &data_tuple
                    .0
                    .split(',')
                    .map(|e| e.to_string().parse::<f32>().ok().unwrap())
                    .collect::<Vec<f32>>()[..],
            ]
            .concat();

            // add result Y
            y.push(data_tuple.1.parse::<f32>().expect("Failed"));
        }
    }

    let num_feat = flatten_x.len() / y.len();

    Ok((flatten_x, num_feat, y))
}
