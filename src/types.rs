use num::{Float, FromPrimitive};
use std::{iter::Sum, ops::AddAssign};

use crate::bits::Bits;
#[cfg(feature="python")]
use crate::pydata::NumpyEquivalent;
#[cfg(feature="rust-hdf5")]
use {hdf5::H5Type};
use num::Zero;



macro_rules! param_struct {
	/* Matching e.g. SomeParams[Debug, Clone]<F: Float> {a: F = F::one()} */
	(
		$name:ident /* Name of the parameter struct */
		$([$($derived_type:ty),*])? /* Derived types */
		$(<$($generic_names:ident : $generic_types:path)*>)? /* Generics */
		{$($field_name:ident: $field_type:ty = $field_value:expr),*$(,)?} /* Fields */
	) => { paste::paste! {
		#[derive($($($derived_type,)*)?)]
		pub struct $name$(<$($generic_names: $generic_types),*>)? {
			$(pub $field_name: $field_type),*
		}
		impl$(<$($generic_names: $generic_types),*>)? $name$(<$($generic_names),*>)? {
			pub fn new() -> Self {
				Self {
					$($field_name: $field_value),*
				}
			}
			pub fn new_full($($field_name: Option<$field_type>,)*) -> Self {
				let mut ret = Self::new();
				$(
					if $field_name.is_some() { ret.$field_name = $field_name.unwrap(); }
				)*
				ret
			}
			$(
				pub fn [<with_ $field_name>](mut self, $field_name: $field_type) -> Self {
					self.$field_name = $field_name;
					self
				}
			)*
			$(
				pub fn [<maybe_with_ $field_name>](mut self, $field_name: Option<$field_type>) -> Self {
					if $field_name.is_some() {
						self = self.[<with_ $field_name>]($field_name.unwrap());
					}
					self
				}
			)*
		}
	}};
}
pub(crate) use param_struct;



pub trait VFMADotProd<const LANES: usize>: std::ops::Sub<Output=Self>+std::ops::Mul<Output=Self>+std::ops::AddAssign+std::iter::Sum+Clone+Copy+num::Zero {
	#[inline(always)]
	fn dot_prod(v1: &[Self], v2: &[Self], d: usize) -> Self {
		debug_assert!(LANES.count_ones() == 1); // must be power of two; compile time assertion
		debug_assert!(v1.len() == d && v2.len() == d); // bounds check
		let sd = d & !(LANES - 1);
		let mut vsum = [Self::zero(); LANES];
		for i in (0..sd).step_by(LANES) {
			let (vv, cc) = (&v1[i..(i + LANES)], &v2[i..(i + LANES)]);
			for j in 0..LANES {
				unsafe {
					let x = *vv.get_unchecked(j) * *cc.get_unchecked(j);
					// emulated
					// *vsum.get_unchecked_mut(j) = x.mul_add(x, *vsum.get_unchecked(j));
					// FMA
					*vsum.get_unchecked_mut(j) += x;
				}
			}
		}
		let mut sum = vsum.into_iter().sum::<Self>();
		if d > sd {
			sum += (sd..d)
			.map(|i| unsafe { *v1.get_unchecked(i) * *v2.get_unchecked(i) })
			.sum();
		}
		sum
	}
}
impl VFMADotProd<2> for f32 {
	#[inline(always)]
	fn dot_prod(v1: &[Self], v2: &[Self], d: usize) -> Self { <Self as VFMADotProd<4>>::dot_prod(v1, v2, d) }
}
impl VFMADotProd<4> for f32 {
	#[inline(always)]
	fn dot_prod(v1: &[Self], v2: &[Self], d: usize) -> Self {
		debug_assert!(v1.len() == d && v2.len() == d); // bounds check
		const LANES: usize = 4;
		unsafe {
			use std::arch::x86_64::*;
			_mm_prefetch(v1.get_unchecked(0) as *const Self as *const i8, _MM_HINT_T0);
			_mm_prefetch(v2.get_unchecked(0) as *const Self as *const i8, _MM_HINT_T0);
			let sd = d & !(LANES - 1);
			let mut vsum = _mm_setzero_ps();
			for i in (0..sd).step_by(LANES) {
				let next_i = i+LANES;
				if next_i < d {
					_mm_prefetch(v1.get_unchecked(next_i) as *const Self as *const i8, _MM_HINT_T0);
					_mm_prefetch(v2.get_unchecked(next_i) as *const Self as *const i8, _MM_HINT_T0);
				}
				let v1 = _mm_loadu_ps(v1.get_unchecked(i) as *const Self);
				let v2 = _mm_loadu_ps(v2.get_unchecked(i) as *const Self);
				vsum = _mm_fmadd_ps(v1, v2, vsum);
			}
			let sum = _mm_hadd_ps(vsum, vsum);
			let sum = _mm_hadd_ps(sum, sum);
			let mut sum = _mm_cvtss_f32(sum);
			if d > sd {
				sum += (sd..d)
				.map(|i| *v1.get_unchecked(i) * *v2.get_unchecked(i))
				.sum::<Self>();
			}
			sum
		}
	}
}
impl VFMADotProd<8> for f32 {
	#[inline(always)]
	fn dot_prod(v1: &[Self], v2: &[Self], d: usize) -> Self {
		debug_assert!(v1.len() == d && v2.len() == d); // bounds check
		const LANES: usize = 8;
		unsafe {
			use std::arch::x86_64::*;
			_mm_prefetch(v1.get_unchecked(0) as *const Self as *const i8, _MM_HINT_T0);
			_mm_prefetch(v2.get_unchecked(0) as *const Self as *const i8, _MM_HINT_T0);
			let sd = d & !(LANES - 1);
			let mut vsum = _mm256_setzero_ps();
			for i in (0..sd).step_by(LANES) {
				let next_i = i+LANES;
				if next_i < d {
					_mm_prefetch(v1.get_unchecked(next_i) as *const Self as *const i8, _MM_HINT_T0);
					_mm_prefetch(v2.get_unchecked(next_i) as *const Self as *const i8, _MM_HINT_T0);
				}
				let v1 = _mm256_loadu_ps(v1.get_unchecked(i) as *const Self);
				let v2 = _mm256_loadu_ps(v2.get_unchecked(i) as *const Self);
				vsum = _mm256_fmadd_ps(v1, v2, vsum);
			}
			let sum = _mm256_hadd_ps(vsum, vsum);
			let sum = _mm256_hadd_ps(sum, sum);
			let sum_low = _mm256_castps256_ps128(sum);
			let sum_high = _mm256_extractf128_ps(sum, 1);
			let final_sum = _mm_add_ps(sum_low, sum_high);
			let mut sum = _mm_cvtss_f32(final_sum);
			if d > sd {
				sum += (sd..d)
				.map(|i| *v1.get_unchecked(i) * *v2.get_unchecked(i))
				.sum::<Self>();
			}
			sum
		}
	}
}
impl VFMADotProd<16> for f32 {
	#[inline(always)]
	fn dot_prod(v1: &[Self], v2: &[Self], d: usize) -> Self { <Self as VFMADotProd<8>>::dot_prod(v1, v2, d) }
}
impl VFMADotProd<2> for f64 {
	#[inline(always)]
	fn dot_prod(v1: &[Self], v2: &[Self], d: usize) -> Self {
		debug_assert!(v1.len() == d && v2.len() == d); // bounds check
		const LANES: usize = 2;
		unsafe {
			use std::arch::x86_64::*;
			_mm_prefetch(v1.get_unchecked(0) as *const Self as *const i8, _MM_HINT_T0);
			_mm_prefetch(v2.get_unchecked(0) as *const Self as *const i8, _MM_HINT_T0);
			let sd = d & !(LANES - 1);
			let mut vsum = _mm_setzero_pd();
			for i in (0..sd).step_by(LANES) {
				let next_i = i+LANES;
				if next_i < d {
					_mm_prefetch(v1.get_unchecked(next_i) as *const Self as *const i8, _MM_HINT_T0);
					_mm_prefetch(v2.get_unchecked(next_i) as *const Self as *const i8, _MM_HINT_T0);
				}
				let v1 = _mm_loadu_pd(v1.get_unchecked(i) as *const Self);
				let v2 = _mm_loadu_pd(v2.get_unchecked(i) as *const Self);
				vsum = _mm_fmadd_pd(v1, v2, vsum);
			}
			let sum = _mm_hadd_pd(vsum, vsum);
			let mut sum = _mm_cvtsd_f64(sum);
			if d > sd {
				sum += (sd..d)
				.map(|i| *v1.get_unchecked(i) * *v2.get_unchecked(i))
				.sum::<Self>();
			}
			sum
		}
	}
}
impl VFMADotProd<4> for f64 {
	#[inline(always)]
	fn dot_prod(v1: &[Self], v2: &[Self], d: usize) -> Self {
		debug_assert!(v1.len() == d && v2.len() == d); // bounds check
		const LANES: usize = 4;
		unsafe {
			use std::arch::x86_64::*;
			_mm_prefetch(v1.get_unchecked(0) as *const Self as *const i8, _MM_HINT_T0);
			_mm_prefetch(v2.get_unchecked(0) as *const Self as *const i8, _MM_HINT_T0);
			let sd = d & !(LANES - 1);
			let mut vsum = _mm256_setzero_pd();
			for i in (0..sd).step_by(LANES) {
				let next_i = i+LANES;
				if next_i < d {
					_mm_prefetch(v1.get_unchecked(next_i) as *const Self as *const i8, _MM_HINT_T0);
					_mm_prefetch(v2.get_unchecked(next_i) as *const Self as *const i8, _MM_HINT_T0);
				}
				let v1 = _mm256_loadu_pd(v1.get_unchecked(i) as *const Self);
				let v2 = _mm256_loadu_pd(v2.get_unchecked(i) as *const Self);
				vsum = _mm256_fmadd_pd(v1, v2, vsum);
			}
			let sum = _mm256_hadd_pd(vsum, vsum);
			let sum_low = _mm256_castpd256_pd128(sum);
			let sum_high = _mm256_extractf128_pd(sum, 1);
			let final_sum = _mm_add_pd(sum_low, sum_high);
			let mut sum = _mm_cvtsd_f64(final_sum);
			if d > sd {
				sum += (sd..d)
				.map(|i| *v1.get_unchecked(i) * *v2.get_unchecked(i))
				.sum::<Self>();
			}
			sum
		}
	}
}
impl VFMADotProd<8> for f64 {
	#[inline(always)]
	fn dot_prod(v1: &[Self], v2: &[Self], d: usize) -> Self { <Self as VFMADotProd<4>>::dot_prod(v1, v2, d) }
}
impl VFMADotProd<16> for f64 {
	#[inline(always)]
	fn dot_prod(v1: &[Self], v2: &[Self], d: usize) -> Self { <Self as VFMADotProd<4>>::dot_prod(v1, v2, d) }
}
#[cfg(feature="half")]
impl VFMADotProd<2> for half::f16 {
	#[inline(always)]
	fn dot_prod(v1: &[Self], v2: &[Self], d: usize) -> Self { <Self as VFMADotProd<8>>::dot_prod(v1, v2, d) }
}
#[cfg(feature="half")]
impl VFMADotProd<4> for half::f16 {
	#[inline(always)]
	fn dot_prod(v1: &[Self], v2: &[Self], d: usize) -> Self { <Self as VFMADotProd<8>>::dot_prod(v1, v2, d) }
}
#[cfg(feature="half")]
impl VFMADotProd<8> for half::f16 {
	#[inline(always)]
	fn dot_prod(v1: &[Self], v2: &[Self], d: usize) -> Self {
		debug_assert!(v1.len() == d && v2.len() == d);
		const LANES: usize = 8; // 8 f16 -> 8 f32 in __m256
		unsafe {
			use std::arch::x86_64::*;
			_mm_prefetch(v1.as_ptr() as *const i8, _MM_HINT_T0);
			_mm_prefetch(v2.as_ptr() as *const i8, _MM_HINT_T0);
			let sd = d & !(LANES - 1);
			let mut vsum = _mm256_setzero_ps();
			for i in (0..sd).step_by(LANES) {
				let next_i = i + LANES;
				if next_i < d {
					_mm_prefetch(v1.as_ptr().add(next_i) as *const i8, _MM_HINT_T0);
					_mm_prefetch(v2.as_ptr().add(next_i) as *const i8, _MM_HINT_T0);
				}
				let a = _mm_loadu_si128(v1.as_ptr().add(i) as *const __m128i);
				let b = _mm_loadu_si128(v2.as_ptr().add(i) as *const __m128i);
				let a = _mm256_cvtph_ps(a);
				let b = _mm256_cvtph_ps(b);
				vsum = _mm256_fmadd_ps(a, b, vsum);
			}
			let lo = _mm256_castps256_ps128(vsum);
			let hi = _mm256_extractf128_ps(vsum, 1);
			let sum128 = _mm_add_ps(lo, hi);
			let sum128 = _mm_hadd_ps(sum128, sum128);
			let sum128 = _mm_hadd_ps(sum128, sum128);
			let mut sum = _mm_cvtss_f32(sum128);
			if d > sd {
				for i in sd..d {
					let x = v1.get_unchecked(i).to_f32() * v2.get_unchecked(i).to_f32();
					sum += x;
				}
			}
			half::f16::from_f32(sum)
		}
	}
}
#[cfg(feature="half")]
impl VFMADotProd<16> for half::f16 {
	#[inline(always)]
	fn dot_prod(v1: &[Self], v2: &[Self], d: usize) -> Self { <Self as VFMADotProd<8>>::dot_prod(v1, v2, d) }
}
#[test]
fn test_vfma_dot() {
	use rand::random;
	let d = 47;
	#[cfg(feature="half")]
	let v1_16: Vec<half::f16> = (0..d).map(|_| half::f16::from_f32(random())).collect();
	#[cfg(feature="half")]
	let v2_16: Vec<half::f16> = (0..d).map(|_| half::f16::from_f32(random())).collect();
	let v1_32: Vec<f32> = (0..d).map(|_| random()).collect();
	let v2_32: Vec<f32> = (0..d).map(|_| random()).collect();
	// let v1_16: Vec<f64> = v1_32.iter().cloned().map(|v| v as f16).collect();
	// let v2_16: Vec<f64> = v2_32.iter().cloned().map(|v| v as f16).collect();
	let v1_64: Vec<f64> = v1_32.iter().cloned().map(|v| v as f64).collect();
	let v2_64: Vec<f64> = v2_32.iter().cloned().map(|v| v as f64).collect();
	#[cfg(feature="half")]
	let true_dist_16: half::f16 = v1_16.iter().zip(v2_16.iter()).map(|(&a, &b)| a*b).sum();
	let true_dist_32: f32 = v1_32.iter().zip(v2_32.iter()).map(|(&a, &b)| a*b).sum();
	let true_dist_64: f64 = v1_64.iter().zip(v2_64.iter()).map(|(&a, &b)| a*b).sum();
	#[cfg(feature="half")]
	[
		<half::f16 as VFMADotProd<2>>::dot_prod,
		<half::f16 as VFMADotProd<4>>::dot_prod,
		<half::f16 as VFMADotProd<8>>::dot_prod,
		<half::f16 as VFMADotProd<16>>::dot_prod,
	].iter().zip(vec![2,4,8,16]).for_each(|(fun, lanes)| {
		let dist = fun(v1_16.as_slice(), v2_16.as_slice(), v1_16.len()) as half::f16;
		assert!((true_dist_16-dist).to_f32().abs() < 1e-3 * true_dist_16.to_f32(), "f16x{:?}: {:?} != {:?}", lanes, true_dist_16, dist);
	});
	[
		<f32 as VFMADotProd<2>>::dot_prod,
		<f32 as VFMADotProd<4>>::dot_prod,
		<f32 as VFMADotProd<8>>::dot_prod,
		<f32 as VFMADotProd<16>>::dot_prod,
	].iter().zip(vec![2,4,8,16]).for_each(|(fun, lanes)| {
		let dist = fun(v1_32.as_slice(), v2_32.as_slice(), v1_32.len());
		assert!((true_dist_32-dist).abs() < 1e-5, "f32x{:?}: {:?} != {:?}", lanes, true_dist_32, dist);
	});
	[
		<f64 as VFMADotProd<2>>::dot_prod,
		<f64 as VFMADotProd<4>>::dot_prod,
		<f64 as VFMADotProd<8>>::dot_prod,
		<f64 as VFMADotProd<16>>::dot_prod,
		].iter().zip(vec![2,4,8,16]).for_each(|(fun, lanes)| {
		let dist = fun(v1_64.as_slice(), v2_64.as_slice(), v1_64.len());
		assert!((true_dist_64-dist).abs() < 1e-10, "f64x{:?}: {:?} != {:?}", lanes, true_dist_64, dist);
	});
}



macro_rules! trait_combiner {
	($combination_name: ident) => {
		pub trait $combination_name {}
		impl<T> $combination_name for T {}
	};
	($combination_name: ident: $t: tt $(+ $ts: tt)*) => {
		pub trait $combination_name: $t $(+ $ts)* {}
		impl<T: $t $(+ $ts)*> $combination_name for T {}
	};
}
#[cfg(feature="rust-hdf5")]
#[cfg(feature="python")]
trait_combiner!(HIOBFloat: (VFMADotProd<2>)+(VFMADotProd<4>)+(VFMADotProd<8>)+(VFMADotProd<16>)+FromPrimitive+CachingNumpyEquivalent+H5Type+Float+Sum+AddAssign+MaybeSend+MaybeSync);
#[cfg(feature="rust-hdf5")]
#[cfg(not(feature="python"))]
trait_combiner!(HIOBFloat: (VFMADotProd<2>)+(VFMADotProd<4>)+(VFMADotProd<8>)+(VFMADotProd<16>)+FromPrimitive+H5Type+Float+Sum+AddAssign+MaybeSend+MaybeSync);
#[cfg(not(feature="rust-hdf5"))]
#[cfg(feature="python")]
trait_combiner!(HIOBFloat: (VFMADotProd<2>)+(VFMADotProd<4>)+(VFMADotProd<8>)+(VFMADotProd<16>)+FromPrimitive+CachingNumpyEquivalent+Float+Sum+AddAssign+MaybeSend+MaybeSync);
#[cfg(not(feature="rust-hdf5"))]
#[cfg(not(feature="python"))]
trait_combiner!(HIOBFloat: (VFMADotProd<2>)+(VFMADotProd<4>)+(VFMADotProd<8>)+(VFMADotProd<16>)+FromPrimitive+Float+Sum+AddAssign+MaybeSend+MaybeSync);
trait_combiner!(HIOBBits: Bits+Clone+MaybeSend+MaybeSync);


#[cfg(feature="parallel")]
trait_combiner!(MaybeSync: Sync);
#[cfg(not(feature="parallel"))]
trait_combiner!(MaybeSync);
#[cfg(feature="parallel")]
trait_combiner!(MaybeSend: Send);
#[cfg(not(feature="parallel"))]
trait_combiner!(MaybeSend);


#[cfg(feature="rust-hdf5")]
trait_combiner!(CachingH5Type: H5Type+Zero+Copy+Clone+Send+'static);
#[cfg(feature="python")]
trait_combiner!(CachingNumpyEquivalent: Zero+NumpyEquivalent+'static);

