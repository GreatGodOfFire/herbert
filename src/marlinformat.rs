use crate::features::Features;

#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PackedBoard {
    occupied: u64,
    pieces: u128,
    stm_ep: u8,
    halfmove: u8,
    fullmove: u16,
    eval: i16,
    wdl: u8,
    extra: u8,
}

unsafe impl bytemuck::Pod for PackedBoard {}
unsafe impl bytemuck::Zeroable for PackedBoard {}

#[derive(Clone, Copy)]
pub struct Piece {
    pub color: bool,
    pub ty: PieceType,
    pub sq: u8,
}

#[derive(Clone, Copy, Debug)]
pub enum PieceType {
    Pawn = 0,
    Knight,
    Bishop,
    Rook,
    Queen,
    King,
    UnmovedRook,
}

impl PackedBoard {
    pub fn unpack(&self) -> (Vec<Piece>, [[u8; 2]; 2], bool, i16, u8) {
        let mut occupied = self.occupied();

        let mut packed_pieces = self.pieces();

        let mut seen_king = [false; 2];
        let mut castling = [[64; 2]; 2];

        let mut pieces = vec![];

        while occupied != 0 {
            let sq = occupied.trailing_zeros() as u8;
            occupied &= occupied - 1;

            let mut ty = packed_pieces as usize & 0b111;
            let color = packed_pieces as u8 & 0b1000 != 0;
            packed_pieces >>= 4;

            if sq == 61 && ty == 0 {
                panic!();
            }

            if ty == PieceType::King as usize {
                seen_king[color as usize] = true;
            }

            if ty == PieceType::UnmovedRook as usize {
                ty = PieceType::Rook as usize;
                castling[color as usize][seen_king[color as usize] as usize] = sq;
            }

            let ty = match ty {
                0 => PieceType::Pawn,
                1 => PieceType::Knight,
                2 => PieceType::Bishop,
                3 => PieceType::Rook,
                4 => PieceType::Queen,
                5 => PieceType::King,
                _ => unreachable!(),
            };

            pieces.push(Piece { color, ty, sq });
        }

        (pieces, castling, self.stm(), self.eval(), self.wdl())
    }
}

impl PackedBoard {
    fn occupied(&self) -> u64 {
        u64::from_le(self.occupied)
    }
    fn pieces(&self) -> u128 {
        u128::from_le(self.pieces)
    }
    fn stm(&self) -> bool {
        self.stm_ep & 0b10000000 != 0
    }
    fn ep_square(&self) -> u8 {
        self.stm_ep & 0b1111111
    }
    fn halfmove(&self) -> u8 {
        self.halfmove
    }
    fn fullmove(&self) -> u16 {
        u16::from_le(self.fullmove)
    }
    fn eval(&self) -> i16 {
        i16::from_le(self.eval)
    }
    pub fn wdl(&self) -> u8 {
        self.wdl
    }
    fn extra(&self) -> u8 {
        self.extra
    }

    pub fn read_many(bytes: &[u8]) -> &[Self] {
        bytemuck::cast_slice(bytes)
    }
}
