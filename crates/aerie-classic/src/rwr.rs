use faer::Mat;

pub fn random_walk_with_restart(
    p: &Mat<f64>,
    labels: &[Option<usize>],
    n_classes: usize,
    alpha: f64,
    tol: f64,
    max_iter: usize,
) -> (Mat<f64>, usize) {
    let n = p.nrows();
    let mut y_0: Mat<f64> = Mat::zeros(n, n_classes);

    for i in 0..labels.len() {
        if let Some(c) = labels[i] {
            y_0[(i, c)] = 1.0;
        }
    }

    let mut y = y_0.clone();
    let mut y_old = y_0.clone();
    let mut iteration_count: usize = 0;
    for _ in 0..max_iter {
        iteration_count += 1;
        // Converged unless a cell below moves by more than tol.
        let mut break_tf = false;

        y = alpha * p * &y_old + ((1.0 - alpha) * &y_0);

        'y_iterator: for row in 0..labels.len() {
            for col in 0..n_classes {
                if (y[(row, col)] - y_old[(row, col)]).abs() > tol {
                    break_tf = true;
                    break 'y_iterator;
                }
            }
        }

        y_old = y.clone();

        if !break_tf {
            break;
        }
    }

    (y, iteration_count)
}
