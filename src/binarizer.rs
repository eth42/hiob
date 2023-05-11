use std::{ops::{Mul, Div}, f64::consts::PI};


use num::{Float};
use ndarray::{Axis, Array2, Array1, ArrayBase, Ix1, ScalarOperand, Data, Ix2, ArrayView2, ArrayView1};
use rand::prelude::*;
#[cfg(feature="parallel")]
use rayon::iter::{ParallelIterator};

use crate::bit_vectors::{BitVector, BitVectorMut};
use crate::measures::{DotProduct, InnerProduct};
use crate::bits::{Bits};
use crate::progress::{named_range, named_par_iter, par_iter, MaybeSend, MaybeSync};

macro_rules! trait_combiner {
	($combination_name: ident) => {
		pub trait $combination_name {}
		impl<T> $combination_name for T {}
	};
	($combination_name: ident: $t: ident $(+ $ts: ident)*) => {
		pub trait $combination_name: $t $(+ $ts)* {}
		impl<T: $t $(+ $ts)*> $combination_name for T {}
	};
}

trait_combiner!(HIOBFloat: Float+ScalarOperand+MaybeSend+MaybeSync);
trait_combiner!(HIOBBits: Bits+Clone+MaybeSend+MaybeSync);

pub struct HIOB<F: HIOBFloat, B: HIOBBits> where Array1<B>: BitVectorMut {
	n_data: usize,
	n_dims: usize,
	n_bits: usize,
	scale: F,
	data: Array2<F>,
	centers: Array2<F>,
	data_bins: Array2<B>,
	data_bins_t: Array2<B>,
	product: DotProduct<F>,
	overlap_mat: Array2<usize>,
	sim_mat: Array2<f64>,
	sim_sums: Array1<f64>,
	update_parallel: bool,
	displace_parallel: bool,
}
impl<F: HIOBFloat, B: HIOBBits> HIOB<F, B> where Array1<B>: BitVectorMut {
	pub fn new(data_in: Array2<F>, n_bits: usize, scale: Option<F>, centers: Option<Array2<F>>) -> HIOB<F, B> {
		let n = data_in.shape()[0];
		let d = data_in.shape()[1];
		/* Calculate the number of instances of B to accommodate >=n_bits bits */
		let n_buckets = n_bits / B::size() + (if n_bits % B::size() > 0 {1} else {0});
		/* Calculate the number of instances of B to accommodate >=n bits */
		let n_buckets_t = n / B::size() + (if n % B::size() > 0 {1} else {0});
		let centers = if centers.is_some() {
			centers.unwrap()
		} else {
			/* Choose centers at random */
			let mut rand_centers = Array2::zeros([n_bits, d]);
			_idx_choice(n, n_bits)
			.into_iter().enumerate()
			.for_each(|(i_center, i_point)| rand_centers.row_mut(i_center).assign(&data_in.row(i_point)));
			rand_centers
		};
		/* Create instance */
		let mut ret = HIOB {
			n_data: n,
			n_dims: d,
			n_bits: n_bits,
			scale: if scale.is_some() { scale.unwrap() } else { F::one() },
			data: data_in,
			centers: centers,
			data_bins: Array2::from_elem([n, n_buckets], B::zeros()),
			data_bins_t: Array2::from_elem([n_bits, n_buckets_t], B::zeros()),
			product: DotProduct::new(),
			overlap_mat: Array2::from_elem([n_bits, n_bits], n),
			sim_mat: Array2::from_elem([n_bits, n_bits], 0.0),
			sim_sums: Array1::from_elem(n_bits, 0.0),
			update_parallel: false,
			displace_parallel: false,
		};
		/* Initialize the instance by computing all entries in the dyn prog matrices */
		named_range(n_bits, "Initializing binarization arrays")
		.for_each(|i| ret.update_bits(i));
		named_range(n_bits, "Initializing overlap array")
		.for_each(|i| ret.update_overlaps(i));
		ret
	}

	#[inline(always)]
	pub fn overlap_to_sim(&self, overlap: usize) -> f64 {
		((overlap as f64) / (self.n_data as f64) - 0.5).abs()
	}
	fn update_bits(&mut self, i_center: usize) {
		let c = self.centers.row(i_center);
		let mut cb = self.data_bins_t.row_mut(i_center);
		// (0..cb.len()).for_each(|i| cb[i] = B::zeros());
		par_iter(
			(0..cb.len())
			.zip(self.data.axis_chunks_iter(Axis(0), B::size()))
			.zip(self.data_bins.axis_chunks_iter_mut(Axis(0), B::size()))
			.map(|((i_bits, points), points_bin)|
				(i_bits, points, points_bin)
			)
		)
		.map(|(i_bits, points, mut points_bin)| {
			let mut bits = B::zeros();
			points.axis_iter(Axis(0)).zip(points_bin.axis_iter_mut(Axis(0)))
			.enumerate()
			.map(|(a,(b,c))| (a,b,c))
			.for_each(|(i_bit, point, mut point_bin)| {
				let bit = self.product.prod(&c, &point) >= F::zero();
				point_bin.set_bit_unchecked(i_center, bit);
				bits.set_bit_unchecked(i_bit, bit);
			});
			(i_bits, bits)
		})
		.collect::<Vec<(usize, B)>>()
		.into_iter()
		.for_each(|(i_bits, bits)| cb[i_bits] = bits);
		// par_iter(
		// 	self.data.axis_iter(Axis(0))
		// 	.zip(self.data_bins.axis_iter_mut(Axis(0)))
		// 	.enumerate()
		// )
		// .map(|(a,(b,c))| (a,b,c))
		// .map(|(i_point,point,mut point_bin)| {
		// 	let bit = self.product.prod(&c, &point) >= F::zero();
		// 	point_bin.set_bit(i_center, bit);
		// 	let item_idx = i_point / B::size();
		// 	let bit_idx = i_point % B::size();
		// 	let mut bit_mask = B::zeros();
		// 	bit_mask.set_bit(bit_idx, bit);
		// 	(item_idx, bit_mask)
		// })
		// .collect::<Vec<(usize, B)>>()
		// .into_iter()
		// .for_each(|(item_idx, bit_mask)| cb[item_idx] = cb[item_idx].or(&bit_mask));
	}
	fn update_overlaps(&mut self, i_center: usize) {
		self.sim_sums[i_center] = 0.0;
		let row_i = self.data_bins_t.row(i_center);
		par_iter(self.data_bins_t.axis_iter(Axis(0)).enumerate())
		.filter(|(j_center, _)| i_center != *j_center)
		.map(|(j_center, row_j)| {
			let overlap = if i_center == j_center { 0 } else {
				self.n_data - row_i.hamming_dist_same(&row_j)
			};
			let sim = self.overlap_to_sim(overlap);
			let old_sim = self.sim_mat[[i_center,j_center]];
			(j_center, overlap, sim, old_sim)
		})
		.collect::<Vec<(usize, usize, f64, f64)>>()
		.into_iter()
		.for_each(|(j_center, overlap, sim, old_sim)| {
			self.sim_sums[i_center] += sim;
			self.sim_sums[j_center] += sim - old_sim;
			self.overlap_mat[[i_center,j_center]] = overlap;
			self.overlap_mat[[j_center,i_center]] = overlap;
			self.sim_mat[[i_center,j_center]] = sim;
			self.sim_mat[[j_center,i_center]] = sim;
		});
	}

	fn vec_norm<D: Data<Elem=F>>(vec: &ArrayBase<D, Ix1>) -> F {
		vec.iter().map(|&v| v*v).reduce(|a,b| a+b).unwrap().sqrt()
	}
	fn displacement_vec(&self, i_center: usize, j_center: usize) -> Array1<F> {
		let frac_equal = 
			F::from(self.overlap_mat[[i_center,j_center]]).unwrap()
			/ F::from(self.n_data).unwrap();
		let frac_unequal = F::one() - frac_equal;
		let rot_angle = ((frac_equal-frac_unequal)/F::from(2).unwrap())*F::from(PI).unwrap();
		let ci = self.centers.row(i_center);
		let cj = self.centers.row(j_center);
		let displacement_vec = ci.mul(ci.dot(&cj)) - cj;
		let norm = HIOB::vec_norm(&displacement_vec);
		let displacement_vec = displacement_vec.div(norm);
		displacement_vec.mul(rot_angle.mul(self.scale).tan())
		// displacement_vec.mul(rot_angle.tan() / F::from(2).unwrap())
	}

	pub fn step(&mut self) {
		let mis = if !self.update_parallel {
			[
				/* Selection of largest mean similarity */
				// self.sim_sums.iter()
				// .enumerate()
				// .reduce(|(i,a),(j,b)| if a > b {(i,a)} else {(j,b)})
				// .unwrap().0
				/* Selection of largest total similarity */
				// self.overlap_mat.axis_iter(Axis(0))
				// .enumerate()
				// .map(|(i_row, row)| (
				// 	i_row,
				// 	row.iter()
				// 	.map(|&overlap| self.overlap_to_sim(overlap))
				// 	.reduce(|a,b| if a>b {a} else {b}).unwrap()
				// ))
				// .reduce(|(i,a),(j,b)| if a>b {(i,a)} else {(j,b)})
				// .unwrap().0
				_random_pair_value(_argmax2(&self.sim_mat))
			].to_vec()
		} else {
			(0..self.n_bits).into_iter().collect()
		};
		let mut new_centers: Array2<F> = Array2::zeros((mis.len(), self.n_dims));
		mis.iter().zip(new_centers.axis_iter_mut(Axis(0))).for_each(|(&mi, mut new_center)| {
			let mjs = if !self.displace_parallel {
				[
					// self.overlap_mat.row(mi)
					// .iter().enumerate()
					// .filter(|(mj,_)| mi != *mj)
					// .map(|(mj,&overlap)| (mj, self.overlap_to_sim(overlap)))
					// .reduce(|(i,a),(j,b)| if a>b {(i,a)} else {(j,b)})
					// .unwrap().0
					_argmax1(&self.sim_mat.row(mi))
				].to_vec()
			} else {
				(0..self.n_bits).into_iter().collect()
			};
			let total_displacement = mjs.into_iter()
			.filter(|mj| mi != *mj)
			.map(|mj| self.displacement_vec(mi, mj))
			.reduce(|u,v| u+v)
			.unwrap();
			new_center.assign(&(total_displacement + self.centers.row(mi)));
			let norm = HIOB::vec_norm(&new_center);
			new_center.assign(&new_center.div(norm));
		});
		mis.iter().zip(new_centers.axis_iter(Axis(0))).for_each(|(&mi, new_center)| {
			self.centers.row_mut(mi).assign(&new_center);
			self.update_bits(mi);
			self.update_overlaps(mi);
		});
	}
	pub fn run(&mut self, n_steps: usize) {
		named_range(n_steps, "Improving pivot positions")
		.for_each(|_| self.step());
	}

	pub fn binarize<D: Data<Elem=F>+MaybeSync>(&self, queries: &ArrayBase<D, Ix2>) -> Array2<B> {
		let n_queries = queries.shape()[0];
		/* Calculate the number of instances of B to accommodate >=n_queries bits */
		let n_buckets = self.n_bits / B::size() + (if self.n_bits % B::size() > 0 {1} else {0});
		let mut bins = Array2::from_elem([n_queries, n_buckets], B::zeros());
		let raw_iter = bins.axis_iter_mut(Axis(0)).enumerate();
		named_par_iter(raw_iter, "Binarizing queries")
		.for_each(|(i, mut bins_row)| {
			self.product.prods(&queries.row(i), &self.centers)
			.into_iter()
			.map(|v| v>=F::zero())
			.enumerate()
			.for_each(|(j,b)| bins_row.set_bit_unchecked(j, b));
		});
		bins
	}

	pub fn get_n_data(&self) -> usize { self.n_data }
	pub fn get_n_dims(&self) -> usize { self.n_dims }
	pub fn get_n_bits(&self) -> usize { self.n_bits }
	pub fn get_scale(&self) -> F { self.scale }
	pub fn set_scale(&mut self, scale: F) { self.scale = scale }
	pub fn get_data<'a>(&'a self) -> ArrayView2<'a, F> { self.data.view() }
	pub fn get_centers<'a>(&'a self) -> ArrayView2<'a, F> { self.centers.view() }
	pub fn get_data_bins<'a>(&'a self) -> ArrayView2<'a, B> { self.data_bins.view() }
	pub fn get_overlap_mat<'a>(&'a self) -> ArrayView2<'a, usize> { self.overlap_mat.view() }
	pub fn get_sim_mat<'a>(&'a self) -> ArrayView2<'a, f64> { self.sim_mat.view() }
	pub fn get_sim_sums<'a>(&'a self) -> ArrayView1<'a, f64> { self.sim_sums.view() }
	pub fn get_update_parallel(&self) -> bool { self.update_parallel }
	pub fn get_displace_parallel(&self) -> bool { self.displace_parallel }
	pub fn set_update_parallel(&mut self, b: bool) { self.update_parallel = b; }
	pub fn set_displace_parallel(&mut self, b: bool) { self.displace_parallel = b; }

}


/// Chooses pairwise different indices between 0 (inclusive) and max (exclusive).
/// 
/// # Arguments
/// * `max` - The upper limit for indices (exlucsive)
/// * `cnt` - The number of indices to return
/// 
/// # Return
/// Result is of type `v: Vec<usize>`
/// * `v` - A `Vec<usize>` in ascending order
fn _idx_choice(max: usize, cnt: usize) -> Vec<usize> {
	let mut rng = rand::thread_rng();
	let mut ret = Vec::new();
	ret.reserve_exact(cnt);
	(0..cnt).for_each(|i| {
		ret.push((rng.next_u64() as usize) % (max-i));
		(0..i).for_each(|j| {
			if ret[i] >= ret[j] {
				ret[i] += 1;
			}
		});
		ret.sort();
	});
	ret
}
fn _max1<T: PartialOrd, D: Data<Elem=T>>(vec: &ArrayBase<D, Ix1>) -> &T {
	vec.iter()
	.reduce(|a,b| if a >= b {a} else {b})
	.unwrap()
}
fn _min1<T: PartialOrd, D: Data<Elem=T>>(vec: &ArrayBase<D, Ix1>) -> &T {
	vec.iter()
	.reduce(|a,b| if a <= b {a} else {b})
	.unwrap()
}
fn _argmax1<T: PartialOrd, D: Data<Elem=T>>(vec: &ArrayBase<D, Ix1>) -> usize {
	vec.indexed_iter()
	.reduce(|(i,a),(j,b)| if a >= b {(i,a)} else {(j,b)})
	.unwrap().0
}
fn _argmin1<T: PartialOrd, D: Data<Elem=T>>(vec: &ArrayBase<D, Ix1>) -> usize {
	vec.indexed_iter()
	.reduce(|(i,a),(j,b)| if a <= b {(i,a)} else {(j,b)})
	.unwrap().0
}
fn _max2<T: PartialOrd, D: Data<Elem=T>>(mat: &ArrayBase<D, Ix2>) -> &T {
	mat.axis_iter(Axis(0))
	.enumerate()
	.map(|(i_row, row)| &mat[[i_row, _argmax1(&row)]])
	.reduce(|a,b| if a >= b {a} else {b})
	.unwrap()
}
fn _min2<T: PartialOrd, D: Data<Elem=T>>(mat: &ArrayBase<D, Ix2>) -> &T {
	mat.axis_iter(Axis(0))
	.enumerate()
	.map(|(i_row, row)| &mat[[i_row, _argmin1(&row)]])
	.reduce(|a,b| if a <= b {a} else {b})
	.unwrap()
}
fn _argmax2<T: PartialOrd, D: Data<Elem=T>>(mat: &ArrayBase<D, Ix2>) -> (usize,usize) {
	let sol = mat.axis_iter(Axis(0))
	.enumerate()
	.map(|(i_row, row)| {
		let max_col = _argmax1(&row);
		let max = &mat[[i_row, max_col]];
		(i_row, max_col, max)
	})
	.reduce(|(i_row,i_col,val_i),(j_row,j_col,val_j)| if val_i >= val_j {(i_row,i_col,val_i)} else {(j_row,j_col,val_j)})
	.unwrap();
	(sol.0, sol.1)
}
fn _argmin2<T: PartialOrd, D: Data<Elem=T>>(mat: &ArrayBase<D, Ix2>) -> (usize,usize) {
	let sol = mat.axis_iter(Axis(0))
	.enumerate()
	.map(|(i_row, row)| {
		let min_col = _argmin1(&row);
		let min = &mat[[i_row, min_col]];
		(i_row, min_col, min)
	})
	.reduce(|(i_row,i_col,val_i),(j_row,j_col,val_j)| if val_i <= val_j {(i_row,i_col,val_i)} else {(j_row,j_col,val_j)})
	.unwrap();
	(sol.0, sol.1)
}
fn _random_pair_value<A>(pair: (A,A)) -> A {
	if rand::random() {pair.0} else {pair.1}
}




#[test]
fn min_max_tests() {
	use ndarray_rand::rand::random;
	/* Array2 */
	let arr2: Array2<u16> = Array2::from_shape_simple_fn([100,100], random) % 0x8000 as u16;
	/* Test argmax and max */
	let true_max = arr2.iter().map(|&v|v).reduce(|a,b| if a>=b {a} else {b}).unwrap();
	let (i,j) = _argmax2(&arr2);
	let pred_max = *_max2(&arr2);
	assert!(true_max == arr2[[i,j]], "True max: {}, Via _argmax2: {}", true_max, arr2[[i,j]]);
	assert!(true_max == pred_max, "True max: {}, Via _max2: {}", true_max, pred_max);
	/* Test argmin and min */
	let true_min = arr2.iter().map(|&v|v).reduce(|a,b| if a<=b {a} else {b}).unwrap();
	let (i,j) = _argmin2(&arr2);
	let pred_min = *_min2(&arr2);
	assert!(true_min == arr2[[i,j]], "True min: {}, Via _argmin2: {}", true_min, arr2[[i,j]]);
	assert!(true_min == pred_min, "True min: {}, Via _min2: {}", true_min, pred_min);
	/* Array1 */
	let arr1: Array1<u16> = Array1::from_shape_simple_fn(100, random) % 0x8000 as u16;
	/* Test argmax and max */
	let true_max = arr1.iter().map(|&v|v).reduce(|a,b| if a>=b {a} else {b}).unwrap();
	let i = _argmax1(&arr1);
	let pred_max = *_max1(&arr1);
	assert!(true_max == arr1[i], "True max: {}, Via _argmax1: {}", true_max, arr1[i]);
	assert!(true_max == pred_max, "True max: {}, Via _max1: {}", true_max, pred_max);
	/* Test argmin and min */
	let true_min = arr1.iter().map(|&v|v).reduce(|a,b| if a<=b {a} else {b}).unwrap();
	let i = _argmin1(&arr1);
	let pred_min = *_min1(&arr1);
	assert!(true_min == arr1[i], "True min: {}, Via _argmin1: {}", true_min, arr1[i]);
	assert!(true_min == pred_min, "True min: {}, Via _min1: {}", true_min, pred_min);
}

