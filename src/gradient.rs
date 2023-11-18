use crate::features::Features;
use rayon::prelude::*;

pub fn compute_gradient(data: &mut Vec<Features>, params: &[f64], k: f64) -> Vec<f64> {
    data.par_iter()
        .map(|x| {
            let mut gradient = vec![0.0; params.len()];
            single_gradient(x, &mut gradient, &params, k);
            gradient
        })
        .reduce(
            || vec![0.0; params.len()],
            |mut a, b| {
                for (a, b) in a.iter_mut().zip(b.iter()) {
                    *a += b;
                }
                a
            },
        )
}

fn single_gradient(features: &Features, gradient: &mut [f64], params: &[f64], k: f64) {
    let e = eval(features, params);
    let s = sigmoid(e, k);
    let res = (features.wdl - s) * s * (1.0 - s);
    let mg = features.phase;
    let eg = 1.0 - features.phase;

    for (i, v) in &features.features {
        gradient[*i] += v * res * mg;
        gradient[*i + 384] += v * res * eg;
    }
}

pub fn eval(features: &Features, params: &[f64]) -> f64 {
    let mut mg = 0.0;
    let mut eg = 0.0;

    for (i, v) in &features.features {
        mg += params[*i] * v;
        eg += params[*i + 384] * v;
    }

    mg * features.phase + eg * (1.0 - features.phase)
}

pub fn sigmoid(x: f64, k: f64) -> f64 {
    1.0 / (1.0 + (-k * x / 400.0).exp())
}
