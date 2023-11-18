use std::{env, fs};

use gradient::{compute_gradient, eval, sigmoid};
use init::PSTS;
use marlinformat::PackedBoard;

use crate::{features::Features, marlinformat::PieceType};

mod features;
mod gradient;
mod init;
mod marlinformat;

const EPOCHS: usize = 10000;

fn main() {
    let mut weights = vec![];

    for x in PSTS.iter().flatten() {
        weights.push(*x as f64);
    }

    let buf = fs::read(env::args().nth(1).unwrap()).unwrap();
    let mut dataset: Vec<_> = PackedBoard::read_many(&buf)
        .iter()
        .map(|x| Features::from_packed(x))
        .collect();

    let mut n = 0;
    let mut d = 0;
    let mut w = 0;
    let mut l = 0;

    for pos in &dataset {
        n += 1;
        if pos.wdl == 0.5 {
            d += 1;
        } else if pos.wdl == 0.0 {
            l += 1;
        } else {
            w += 1;
        }
    }

    dbg!(w, d, l, n);

    let k = optimal_k(&dataset, &weights);
    let rate = 200.0 / n as f64;

    println!("K = {k}");

    // let mut adagrad = vec![0.0; weights.len()];

    rayon::ThreadPoolBuilder::new()
        .num_threads(24)
        .build_global()
        .unwrap();

    for i in 1..=EPOCHS {
        let gradient = compute_gradient(&mut dataset, &weights, k);

        for i in 0..gradient.len() {
            // adagrad[i] += (2.0 * gradient[i]).powi(2);

            weights[i] += 2.0 * k * gradient[i] * rate;
        }

        println!("Epoch {i}: E = {}", eval_errors(&dataset, &weights, k));
    }

    use PieceType::*;
    for (offset, phase) in [(0, "MG"), (384, "EG")] {
        for ty in [Pawn, Knight, Bishop, Rook, Queen, King] {
            println!("#[rustfmt::skip]");
            println!(
                "const {phase}_{}: [i32; 64] = [",
                format!("{ty:?}").to_ascii_uppercase()
            );
            for rank in 0..8 {
                print!("    ");
                for file in 0..8 {
                    print!(
                        "{:>4.0}, ",
                        weights[offset + 64 * ty as usize + rank * 8 + file].round()
                    );
                }
                println!();
            }
            println!("];");
        }
    }
}

fn optimal_k(dataset: &[Features], weights: &[f64]) -> f64 {
    const GR: f64 = 1.61803399;

    let mut a = 0.0;
    let mut b = 100.0;

    let mut k1 = b - (b - a) / GR;
    let mut k2 = a + (b - a) / GR;

    while (b - a).abs() > 0.01 {
        let f1 = eval_errors(dataset, weights, k1);
        let f2 = eval_errors(dataset, weights, k2);
        if f1 < f2 {
            b = k2;
        } else {
            a = k1;
        }
        k1 = b - (b - a) / GR;
        k2 = a + (b - a) / GR;
    }

    (b + a) / 2.0
}

fn eval_errors(dataset: &[Features], weights: &[f64], k: f64) -> f64 {
    let mut total = 0.0;

    for i in 0..dataset.len() {
        total += (dataset[i].wdl - sigmoid(eval(&dataset[i], weights), k)).powi(2) as f64;
    }

    total / dataset.len() as f64
}
