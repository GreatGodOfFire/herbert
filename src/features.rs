use crate::marlinformat::PackedBoard;

pub struct Features {
    pub features: Vec<(usize, f64)>,
    pub wdl: f64,
    pub phase: f64,
}

impl Features {
    pub fn from_packed(board: &PackedBoard) -> Self {
        let pieces = board.unpack().0;

        let mut features = vec![];
        let mut phase = 0;
        for p in &pieces {
            phase += PHASE[p.ty as usize];
        }
        let phase = phase.min(24);

        for p in &pieces {
            if !p.color {
                let feature_num = p.sq as usize + p.ty as usize * 64;
                features.push((feature_num, 1.0));
            } else {
                let feature_num = p.sq as usize ^ 56 + p.ty as usize * 64;
                features.push((feature_num, -1.0));
            }
        }

        Self {
            features,
            wdl: board.wdl() as f64 / 2.0,
            phase: phase as f64 / 24.0,
        }
    }
}

const PHASE: [u8; 6] = [0, 1, 1, 2, 4, 0];
