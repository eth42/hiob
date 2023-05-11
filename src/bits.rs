use paste::paste;

use crate::progress::{MaybeSend, MaybeSync};

pub trait Bits: Clone+Copy+MaybeSend+MaybeSync {
	fn size() -> usize;
	fn get_bit(&self, i: usize) -> Option<bool> {
		if i >= Self::size() { None }
		else { Some(self.get_bit_unchecked(i)) }
	}
	fn get_bit_unchecked(&self, i: usize) -> bool;
	fn set_bit(&mut self, i: usize, b: bool) { if i < Self::size() { self.set_bit_unchecked(i, b) } }
	fn set_bit_unchecked(&mut self, i: usize, b: bool);
	fn count_bits(&self) -> usize;
	fn count_bits_range(&self, lo: usize, hi: usize) -> Option<usize> {
		if lo >= hi || hi > Self::size() { None }
		else { Some(self.count_bits_range_unchecked(lo, hi)) }
	}
	fn count_bits_range_unchecked(&self, lo: usize, hi: usize) -> usize;
	fn zeros() -> Self;
	fn ones() -> Self;
	fn hamming_dist(&self, other: &Self) -> usize;
	fn dot_prod(&self, other: &Self) -> usize;
	fn or(&self, other: &Self) -> Self;
	fn and(&self, other: &Self) -> Self;
	fn xor(&self, other: &Self) -> Self;
	fn not(&self) -> Self;
}

impl Bits for bool {
	fn size() -> usize { 1 }
	fn get_bit_unchecked(&self, _i: usize) -> bool { *self }
	fn set_bit_unchecked(&mut self, _i: usize, b: bool) { *self = b }
	fn count_bits(&self) -> usize { if *self {1} else {0} }
	fn count_bits_range_unchecked(&self, _lo: usize, _hi: usize) -> usize { if *self {1} else {0} }
	fn zeros() -> Self { false }
	fn ones() -> Self { true }
	fn hamming_dist(&self, other: &Self) -> usize { if self != other { 1 } else { 0 } }
	fn dot_prod(&self, other: &Self) -> usize { if self & other { 1 } else { 0 } }
	fn or(&self, other: &Self) -> Self { self | other }
	fn and(&self, other: &Self) -> Self {self & other }
	fn xor(&self, other: &Self) -> Self {self ^ other }
	fn not(&self) -> Self { !self }
}

pub trait BitMasked<const N_BITS: usize>: Sized {
	const BIT_MASKS: [Self; N_BITS];
	const INV_BIT_MASKS: [Self; N_BITS];
}
macro_rules! int_bits {
	($itype: ident, $n_bits: literal ) => {
		paste! {
			const fn [<bit_mask_arr_gen_ $itype>](inv: bool) -> [$itype; $n_bits] {
				let mut ret = [0 as $itype; $n_bits];
				let mut i=0;
				while i<$n_bits {
					ret[i] = (1 as $itype) << i;
					if inv { ret[i] = !ret[i]; }
					i += 1;
				}
				ret
			}
			impl BitMasked<$n_bits> for $itype {
				const BIT_MASKS: [$itype; $n_bits] = [<bit_mask_arr_gen_ $itype>](false);
				const INV_BIT_MASKS: [$itype; $n_bits] = [<bit_mask_arr_gen_ $itype>](true);
			}
		}
		impl Bits for $itype {
			fn size() -> usize { $n_bits }
			fn get_bit_unchecked(&self, i: usize) -> bool { unsafe { (self & Self::BIT_MASKS.get_unchecked(i)) > 0 } }
			fn set_bit_unchecked(&mut self, i: usize, b: bool) {
				unsafe {
					if b { *self |= Self::BIT_MASKS.get_unchecked(i); }
					else { *self &= Self::INV_BIT_MASKS.get_unchecked(i); }
				}
			}
			fn count_bits(&self) -> usize {
				self.count_ones() as usize
			}
			fn count_bits_range_unchecked(&self, lo: usize, hi: usize) -> usize {
				((self >> lo) << (lo+$n_bits-hi)).count_ones() as usize
			}
			fn zeros() -> Self { 0 as $itype }
			fn ones() -> Self { Self::BIT_MASKS.iter().sum() }
			fn hamming_dist(&self, other: &Self) -> usize { (self ^ other).count_ones() as usize }
			fn dot_prod(&self, other: &Self) -> usize { (self & other).count_ones() as usize }
			fn or(&self, other: &Self) -> Self { self | other }
			fn and(&self, other: &Self) -> Self { self & other }
			fn xor(&self, other: &Self) -> Self { self ^ other }
			fn not(&self) -> Self { self ^ Self::ones() }
		}
	};
}
int_bits!(i8, 7);
int_bits!(i16, 15);
int_bits!(i32, 31);
int_bits!(i64, 63);
int_bits!(i128, 127);
int_bits!(u8, 8);
int_bits!(u16, 16);
int_bits!(u32, 32);
int_bits!(u64, 64);
int_bits!(u128, 128);

