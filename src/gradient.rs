use crate::features::Features;

pub fn compute_gradient(data: &mut Vec<Features>, gradient: &mut [f64], params: &[f64], k: f64) {
    for features in data {
        single_gradient(features, gradient, &params, k);
    }
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
