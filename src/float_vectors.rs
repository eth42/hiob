use ndarray::{ArrayBase, Data, Ix1};
use crate::types::HIOBFloat;

pub trait InnerProduct {
	fn prod<F: HIOBFloat>(a: &[F], b: &[F], d: usize) -> F;
	#[inline(always)]
	fn prod_arrs<F: HIOBFloat, D1: Data<Elem=F>, D2: Data<Elem=F>>(a: &ArrayBase<D1, Ix1>, b: &ArrayBase<D2, Ix1>) -> F {
		Self::prod(a.as_slice().unwrap(), b.as_slice().unwrap(), a.len())
	}
}
pub trait VectorDistance {
	fn dist<F: HIOBFloat>(a: &[F], b: &[F], d: usize) -> F;
	#[inline(always)]
	fn dist_arrs<F: HIOBFloat, D1: Data<Elem=F>, D2: Data<Elem=F>>(a: &ArrayBase<D1, Ix1>, b: &ArrayBase<D2, Ix1>) -> F {
		Self::dist(a.as_slice().unwrap(), b.as_slice().unwrap(), a.len())
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

pub struct SqEuclidean {}
impl VectorDistance for SqEuclidean {
	#[inline(always)]
	fn dist<F: HIOBFloat>(a: &[F], b: &[F], d: usize) -> F {
		const LANES: usize = 8;
		if LANES > 1 {
			assert!(LANES != 0 && (LANES & (LANES-1)) == 0); // must be power of two
			let sd = d & !(LANES-1);
			let mut vsum = [F::zero(); LANES];
			for i in (0..sd).step_by(LANES) {
				let (vv, cc) = (&a[i..(i+LANES)], &b[i..(i+LANES)]);
				for j in 0..LANES { unsafe {
					let (a, b) = (*vv.get_unchecked(j), *cc.get_unchecked(j));
					let diff = a - b;
					*vsum.get_unchecked_mut(j) = diff.mul_add(diff, *vsum.get_unchecked(j)); // FMA
				}};
			}
			let mut sum = vsum.iter().copied().sum::<F>();
			if d > sd {
				sum += (sd..d).map(|i| unsafe {
					let diff = *a.get_unchecked(i) - *b.get_unchecked(i);
					diff * diff
				}).sum()
			}
			sum
		} else {
			(0..d).map(|i| unsafe {
				let diff = *a.get_unchecked(i) - *b.get_unchecked(i);
				diff * diff
			}).sum()
		}
	}
}
pub struct Euclidean {}
impl VectorDistance for Euclidean {
	#[inline(always)]
	fn dist<F: HIOBFloat>(a: &[F], b: &[F], d: usize) -> F {
		num::Float::sqrt(SqEuclidean::dist(a, b, d))
	}
}


#[test]
fn test_dot_product() {
	use rand::Rng;
	use ndarray::Array1;
	type F = f64;
	let n_runs = 100;
	let rand_vec = |d: usize| (0..d).map(|_| rand::thread_rng().gen_range(-1e3..1e3)).collect::<Vec<F>>();
	(0..n_runs).for_each(|_| {
		/* Generate random vectors */
		let d = rand::thread_rng().gen_range(10..100);
		let a: Vec<F> = rand_vec(d);
		let b: Vec<F> = rand_vec(d);
		let aa: Array1<F> = a.clone().into();
		let bb: Array1<F> = b.clone().into();
		/* Compute dot product */
		let dot1 = DotProduct::prod(&a, &b, d);
		let dot2 = DotProduct::prod_arrs(&aa, &bb);
		let dot_ref = a.iter().zip(b.iter()).map(|(a, b)| a*b).sum::<F>();
		assert!((dot1 - dot_ref).abs() < 1e-6, "dot1: {}, dot_ref: {}", dot1, dot_ref);
		assert!((dot2 - dot_ref).abs() < 1e-6, "dot2: {}, dot_ref: {}", dot2, dot_ref);
	});
}
#[test]
fn test_sq_euc_dist() {
	use rand::Rng;
	use ndarray::Array1;
	type F = f64;
	let n_runs = 100;
	let rand_vec = |d: usize| (0..d).map(|_| rand::thread_rng().gen_range(-1e3..1e3)).collect::<Vec<F>>();
	(0..n_runs).for_each(|_| {
		/* Generate random vectors */
		let d = rand::thread_rng().gen_range(10..100);
		let a: Vec<F> = rand_vec(d);
		let b: Vec<F> = rand_vec(d);
		let aa: Array1<F> = a.clone().into();
		let bb: Array1<F> = b.clone().into();
		/* Compute dot product */
		let dist1 = SqEuclidean::dist(&a, &b, d);
		let dist2 = SqEuclidean::dist_arrs(&aa, &bb);
		let dist_ref = a.iter().zip(b.iter()).map(|(a, b)| (a-b)*(a-b)).sum::<F>();
		assert!((dist1 - dist_ref).abs() < 1e-6, "dist1: {}, dist_ref: {}", dist1, dist_ref);
		assert!((dist2 - dist_ref).abs() < 1e-6, "dist2: {}, dist_ref: {}", dist2, dist_ref);
	});
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