use ndarray::{ArrayBase, Data, Ix1};
use crate::binarizer::HIOBFloat;

pub trait InnerProduct {
	fn prod<F: HIOBFloat>(a: &[F], b: &[F], d: usize) -> F;
	#[inline(always)]
	fn prod_arrs<F: HIOBFloat, D1: Data<Elem=F>, D2: Data<Elem=F>>(a: &ArrayBase<D1, Ix1>, b: &ArrayBase<D2, Ix1>) -> F {
		Self::prod(a.as_slice().unwrap(), b.as_slice().unwrap(), a.len())
	}
}

pub struct DotProduct {}
// #[cfg(not(feature="safe_arch"))]
impl InnerProduct for DotProduct {
	#[inline(always)]
	fn prod<F: HIOBFloat>(a: &[F], b: &[F], d: usize) -> F {
		const LANES: usize = 8;
		if LANES > 1 {
			assert!(LANES != 0 && (LANES & (LANES-1)) == 0); // must be power of two
			let sd = d & !(LANES-1);
			let mut vsum = [F::zero(); LANES];
			for i in (0..sd).step_by(LANES) {
				let (vv, cc) = (&a[i..(i+LANES)], &b[i..(i+LANES)]);
				for j in 0..LANES { unsafe {
					let (a, b) = (*vv.get_unchecked(j), *cc.get_unchecked(j));
					*vsum.get_unchecked_mut(j) = a.mul_add(b, *vsum.get_unchecked(j)); // FMA
				}};
			}
			let mut sum = vsum.iter().copied().sum::<F>();
			if d > sd {
				sum += (sd..d).map(|i| unsafe { *a.get_unchecked(i) * *b.get_unchecked(i) }).sum()
			}
			sum
		} else {
			(0..d).map(|i| unsafe { *a.get_unchecked(i) * *b.get_unchecked(i) }).sum()
		}
	}
}




// #[cfg(feature="safe_arch")]
// use safe_arch::*;
// #[cfg(feature="safe_arch")]
// impl InnerProduct for DotProduct {
// 	#[inline(always)]
// 	fn prod(a: &[f32], b: &[f32], d: usize) -> f32 {
// 		let mut sum = {
// 			#[cfg(any(target_feature="avx", target_feature="avx2"))]
// 			{zeroed_m256()}
// 			#[cfg(not(any(target_feature="avx", target_feature="avx2")))]
// 			{zeroed_m128()}
// 		};
// 		let mut i = 0;
// 		const STRIDE: usize = {
// 			#[cfg(any(target_feature="avx", target_feature="avx2"))]
// 			{8}
// 			#[cfg(not(any(target_feature="avx", target_feature="avx2")))]
// 			{4}
// 		};
// 		unsafe {
// 			while i + STRIDE <= d {
// 				let a_chunk = {
// 					#[cfg(any(target_feature="avx", target_feature="avx2"))]
// 					{set_m256(*a.get_unchecked(i), *a.get_unchecked(i+1), *a.get_unchecked(i+2), *a.get_unchecked(i+3), *a.get_unchecked(i+4), *a.get_unchecked(i+5), *a.get_unchecked(i+6), *a.get_unchecked(i+7))}
// 					#[cfg(not(any(target_feature="avx", target_feature="avx2")))]
// 					{set_m128(*a.get_unchecked(i), *a.get_unchecked(i+1), *a.get_unchecked(i+2), *a.get_unchecked(i+3))}
// 				};
// 				let b_chunk = {
// 					#[cfg(any(target_feature="avx", target_feature="avx2"))]
// 					{set_m256(*b.get_unchecked(i), *b.get_unchecked(i+1), *b.get_unchecked(i+2), *b.get_unchecked(i+3), *b.get_unchecked(i+4), *b.get_unchecked(i+5), *b.get_unchecked(i+6), *b.get_unchecked(i+7))}
// 					#[cfg(not(any(target_feature="avx", target_feature="avx2")))]
// 					{set_m128(*b.get_unchecked(i), *b.get_unchecked(i+1), *b.get_unchecked(i+2), *b.get_unchecked(i+3))}
// 				};
// 				#[cfg(target_feature="fma")]
// 				{sum = fused_mul_add_m256(a_chunk, b_chunk, sum);}
// 				#[cfg(not(target_feature="fma"))]
// 				{sum += a_chunk * b_chunk;}
// 				i += STRIDE;
// 			}
// 		}
// 		let mut sum_scalar = 0f32;
// 		while i < d {
// 			sum_scalar += a[i] * b[i];
// 			i += 1;
// 		}
// 		sum_scalar + sum.to_array().iter().sum::<f32>()
// 	}
// 	#[inline(always)]
// 	fn prod(a: &[f64], b: &[f64], d: usize) -> f64 {
// 		let mut sum = {
// 			#[cfg(any(target_feature="avx", target_feature="avx2"))]
// 			{zeroed_m256d()}
// 			#[cfg(not(any(target_feature="avx", target_feature="avx2")))]
// 			{zeroed_m128d()}
// 		};
// 		let mut i = 0;
// 		const STRIDE: usize = {
// 			#[cfg(any(target_feature="avx", target_feature="avx2"))]
// 			{4}
// 			#[cfg(not(any(target_feature="avx", target_feature="avx2")))]
// 			{2}
// 		};
// 		unsafe {
// 			while i + STRIDE <= d {
// 				let a_chunk = {
// 					#[cfg(any(target_feature="avx", target_feature="avx2"))]
// 					{set_m256d(*a.get_unchecked(i), *a.get_unchecked(i+1), *a.get_unchecked(i+2), *a.get_unchecked(i+3))}
// 					#[cfg(not(any(target_feature="avx", target_feature="avx2")))]
// 					{set_m128d(*a.get_unchecked(i), *a.get_unchecked(i+1))}
// 				};
// 				let b_chunk = {
// 					#[cfg(any(target_feature="avx", target_feature="avx2"))]
// 					{set_m256d(*b.get_unchecked(i), *b.get_unchecked(i+1), *b.get_unchecked(i+2), *b.get_unchecked(i+3))}
// 					#[cfg(not(any(target_feature="avx", target_feature="avx2")))]
// 					{set_m128d(*b.get_unchecked(i), *b.get_unchecked(i+1))}
// 				};
// 				#[cfg(target_feature="fma")]
// 				{sum = fused_mul_add_m256(a_chunk, b_chunk, sum);}
// 				#[cfg(not(target_feature="fma"))]
// 				{sum += a_chunk * b_chunk;}
// 				i += STRIDE;
// 			}
// 		}
// 		let mut sum_scalar = 0f64;
// 		while i < d {
// 			sum_scalar += a[i] * b[i];
// 			i += 1;
// 		}
// 		sum_scalar + sum.to_array().iter().sum::<f64>()
// 	}
// }