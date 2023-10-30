use std::{ops::{Sub, AddAssign, Add}, f64::consts::PI};
use std::iter::Sum;


use ndarray_rand::rand_distr::{Normal,Distribution};
use num::{Float};
#[cfg(feature="rust-hdf5")]
use hdf5::H5Type;
use ndarray::{Axis, Array2, Array1, ArrayBase, Ix1, Data, Ix2, ArrayView2, ArrayView1};
use rand::thread_rng;
// use rand::prelude::*;
#[cfg(feature="parallel")]
use rayon::iter::{ParallelIterator};

use crate::{
	bit_vectors::{BitVector, BitVectorMut},
	random::RandomPermutationGenerator,
	data::{MatrixDataSource},
};
use crate::float_vectors::{DotProduct, InnerProduct};
use crate::bits::{Bits};
use crate::progress::{named_range, par_iter, MaybeSend, MaybeSync};
#[cfg(feature="python")]
use {
	crate::{
		pydata::{CachingH5PyReader, CachingNumpyEquivalent},
		data::{AsyncMatrixDataSource},
	},
	ndarray::{Slice},
};

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
#[cfg(feature="rust-hdf5")]
#[cfg(feature="python")]
trait_combiner!(HIOBFloat: CachingNumpyEquivalent+H5Type+Float+Sum+AddAssign+MaybeSend+MaybeSync);
#[cfg(feature="rust-hdf5")]
#[cfg(not(feature="python"))]
trait_combiner!(HIOBFloat: H5Type+Float+Sum+AddAssign+MaybeSend+MaybeSync);
#[cfg(not(feature="rust-hdf5"))]
#[cfg(feature="python")]
trait_combiner!(HIOBFloat: CachingNumpyEquivalent+Float+Sum+AddAssign+MaybeSend+MaybeSync);
#[cfg(not(feature="rust-hdf5"))]
#[cfg(not(feature="python"))]
trait_combiner!(HIOBFloat: Float+Sum+AddAssign+MaybeSend+MaybeSync);
trait_combiner!(HIOBBits: Bits+Clone+MaybeSend+MaybeSync);

pub struct HIOB<F: HIOBFloat, B: HIOBBits> where Array1<B>: BitVectorMut {
	n_data: usize,
	n_dims: usize,
	n_bits: usize,
	scale: F,
	data: Array2<F>,
	centers: Array2<F>,
	data_bin_length: usize,
	// center_bin_length: usize,
	// data_bins: Array2<B>,
	data_bins_t: Array2<B>,
	overlap_mat: Array2<usize>,
	sim_mat: Array2<f64>,
	sim_sums: Array1<f64>,
	update_parallel: bool,
	displace_parallel: bool,
	ransac_pairs_per_bit: usize,
	ransac_sub_sample: usize,
	pi_half: F,
	displace_vec_cache: Array1<F>,
	calc_vec: Array1<F>,
	centers_calc_cache: Array2<F>,
	is_affine: bool,
	center_biases: Array1<F>,
	biases_calc_cache: Array1<F>,
	balance_regression_factor: F,
}
impl<F: HIOBFloat, B: HIOBBits> HIOB<F, B> where Array1<B>: BitVectorMut {
	pub fn new(
		data_in: Array2<F>,
		n_bits: usize,
		affine: bool,
		scale: Option<F>,
		centers: Option<Array2<F>>,
		center_biases: Option<Array1<F>>,
		balance_regression_factor: Option<F>,
		init_greedy: Option<bool>,
		init_ransac: Option<bool>,
		ransac_pairs_per_bit: Option<usize>,
		ransac_sub_sample: Option<usize>,
	) -> HIOB<F, B> {
		let n = data_in.shape()[0];
		let d = data_in.shape()[1];
		/* Calculate the number of instances of B to accommodate >=n_bits bits */
		let data_bin_length = n_bits / B::size() + (if n_bits % B::size() > 0 {1} else {0});
		/* Calculate the number of instances of B to accommodate >=n bits */
		let center_bin_length = n / B::size() + (if n % B::size() > 0 {1} else {0});
		let init_greedy = init_greedy.is_some() && init_greedy.unwrap();
		let init_ransac = init_ransac.is_some() && init_ransac.unwrap();
		let init_basic = !init_greedy && !init_ransac;
		let centers = if centers.is_some() {
			centers.unwrap()
		} else {
			if init_basic {
				/* Choose centers at random */
				let mut rand_centers = Array2::zeros([n_bits, d]);
				_idx_choice(n, n_bits)
				.into_iter().enumerate()
				.for_each(|(i_center, i_point)| rand_centers.row_mut(i_center).assign(&data_in.row(i_point)));
				rand_centers
			} else {
				Array2::zeros([n_bits, d])
			}
		};
		let center_biases = if center_biases.is_some() {
			center_biases.unwrap()
		} else {
			Array1::zeros([n_bits])
		};
		let centers_calc_cache = Array2::zeros([n_bits, d]);
		let biases_calc_cache = Array1::zeros([n_bits]);
		/* Create instance */
		let mut ret = HIOB {
			n_data: n,
			n_dims: d,
			n_bits: n_bits,
			scale: if scale.is_some() { scale.unwrap() } else { F::one() },
			data: data_in,
			centers: centers,
			data_bin_length: data_bin_length,
			// center_bin_length: center_bin_length,
			// data_bins: Array2::from_elem([n, data_bin_length], B::zeros()),
			data_bins_t: Array2::from_elem([n_bits, center_bin_length], B::zeros()),
			overlap_mat: Array2::from_elem([n_bits, n_bits], n),
			sim_mat: Array2::from_elem([n_bits, n_bits], 0.0),
			sim_sums: Array1::from_elem(n_bits, 0.0),
			update_parallel: false,
			displace_parallel: false,
			ransac_pairs_per_bit: ransac_pairs_per_bit.unwrap_or(200),
			ransac_sub_sample: ransac_sub_sample.unwrap_or(2000),
			pi_half: F::from(PI).unwrap()/F::from(2).unwrap(),
			displace_vec_cache: Array1::from_elem((d,), F::zero()),
			calc_vec: Array1::from_elem((d,),F::zero()),
			centers_calc_cache: centers_calc_cache,
			is_affine: affine,
			center_biases: center_biases,
			biases_calc_cache: biases_calc_cache,
			balance_regression_factor: balance_regression_factor.unwrap_or(F::zero()),
		};
		/* Initialize the instance by computing all entries in the dyn prog matrices */
		if init_greedy {
			ret.init_greedy();
		} else if init_ransac {
			ret.init_ransac();
		} else {
			ret.init_regular();
		}
		ret
	}

	fn init_regular(&mut self) {
		named_range(self.n_bits, "Initializing binarization arrays")
		.for_each(|i| self.update_bits(i));
		named_range(self.n_bits, "Initializing overlap array")
		.for_each(|i| self.update_overlaps(i));
		self.centers_calc_cache = self.centers.view().to_owned();
		self.biases_calc_cache = self.center_biases.view().to_owned();
	}
	fn init_greedy(&mut self) {
		const ATTEMPTS: usize = 20;
		(0..self.n_bits).for_each(|i_center| {
			let mut p1: usize = 0;
			let mut p2: usize = 0;
			let mut best_hamming = usize::MAX;
			for _ in 0..ATTEMPTS {
				p1 = rand::random::<usize>() % self.n_data;
				let row1 = self.binarize_single(&self.data.row(p1));
				// let row1 = self.data_bins.row(p1);
				let iter = par_iter(0..self.n_data)
				.filter(|p2| p1 != *p2)
				.map(|p2| {
					let row2 = self.binarize_single(&self.data.row(p2));
					// let row2 = self.data_bins.row(p2);
					let hamming = row1.hamming_dist_same(&row2);
					(p2, hamming)
				});
				#[cfg(feature="parallel")]
				let (p2_cand,hamming) = iter
				.reduce(|| (0 as usize, usize::MAX), |(p2a, ha), (p2b, hb)| {
					if ha < hb {(p2a,ha)} else {(p2b,hb)}
				});
				#[cfg(not(feature="parallel"))]
				let (p2_cand,hamming) = iter
				.reduce(|(p2a, ha), (p2b, hb)| {
					if ha < hb {(p2a,ha)} else {(p2b,hb)}
				})
				.unwrap();
				if hamming < best_hamming {
					p2 = p2_cand;
					best_hamming = hamming;
				}
				if best_hamming == 0 { break; }
			}
			let c = self.data.row(p1).sub(&self.data.row(p2));
			let cn = unsafe {Self::vec_norm(&c)};
			self.centers.row_mut(i_center).assign(&c.mapv(|v| v/cn));
			self.update_bits(i_center);
		});
		named_range(self.n_bits, "Initializing overlap array")
		.for_each(|i| self.update_overlaps(i));
	}
	fn init_ransac(&mut self) {
		let samples = _idx_choice(self.n_data, self.ransac_sub_sample);
		let n_buckets = self.ransac_sub_sample / B::size() + (if self.ransac_sub_sample % B::size() > 0 {1} else {0});
		let mut c_bit_vecs = Array2::from_elem([self.n_bits, n_buckets], B::zeros());
		(0..self.n_bits).for_each(|i_center| {
			let mut best_c: Array1<F> = Array1::from_elem(self.n_dims, F::zero());
			let mut best_bias: F = F::zero();
			let mut best_bit_vec: Array1<B> = Array1::from_elem(n_buckets, B::zeros());
			let mut worst_sim: f64 = f64::MAX;
			for _ in 0..self.ransac_pairs_per_bit {
				let p1 = rand::random::<usize>() % self.n_data;
				let mut p2 = rand::random::<usize>() % self.n_data;
				let mut c = self.data.row(p1).sub(&self.data.row(p2));
				let mut cn = unsafe { Self::vec_norm(&c) };
				while cn <= F::zero() {
					p2 = rand::random::<usize>() % self.n_data;
					c = self.data.row(p1).sub(&self.data.row(p2));
					cn = unsafe { Self::vec_norm(&c) };
				}
				c.mapv_inplace(|v| v/cn);
				let mut bit_vec = Array1::from_elem(n_buckets, B::zeros());
				let mut bias = F::zero();
				if !self.is_affine {
					par_iter(bit_vec.iter_mut().enumerate())
					.for_each(|(i_item, target)|
						(0..B::size()).for_each(|i_bit| {
							let i_pnt = i_item*B::size()+i_bit;
							if i_pnt < self.ransac_sub_sample {
								let prod = DotProduct::prod_arrs(
									&c,
									&self.data.row(unsafe { *samples.get_unchecked(i_pnt) })
								);
								let bit = prod >= F::zero();
								target.set_bit_unchecked(i_bit, bit);
							}
						})
					);
				} else {
					let mut dots = vec![F::zero(); self.ransac_sub_sample];
					par_iter(dots.iter_mut().zip(samples.iter()))
					.for_each(|(dot_target, idx)| {
						*dot_target = DotProduct::prod_arrs(
							&c,
							&self.data.row(*idx)
						);
					});
					// let mut sorted_dots: Vec<F> = dots.to_vec();
					// sorted_dots.sort_by(|a,b| a.partial_cmp(b).unwrap());
					// bias = sorted_dots[sorted_dots.len()/2];
					bias = dots[rand::random::<usize>() % dots.len()];
					par_iter(bit_vec.iter_mut().enumerate())
					.for_each(|(i_item, target)|
						(0..B::size()).for_each(|i_bit| {
							let i_pnt = i_item*B::size()+i_bit;
							if i_pnt < self.ransac_sub_sample {
								let prod = dots[i_pnt];
								let bit = prod >= bias;
								target.set_bit_unchecked(i_bit, bit);
							}
						})
					);
				}
				if i_center == 0 {
					best_c = c;
					best_bias = bias;
					best_bit_vec = bit_vec;
					break;
				}
				let local_worst_sim = (0..i_center).map(|j_center| {
					let dist = c_bit_vecs.row(j_center).hamming_dist_same(&bit_vec.view());
					let overlap = self.ransac_sub_sample - dist;
					let sim = ((overlap as f64) / (self.ransac_sub_sample as f64) - 0.5).abs();
					sim
				})
				.reduce(|a,b| if a>b {a} else {b})
				.unwrap();
				if local_worst_sim < worst_sim {
					best_c = c;
					best_bias = bias;
					best_bit_vec = bit_vec;
					worst_sim = local_worst_sim;
				}
			}
			self.centers.row_mut(i_center).assign(&best_c);
			self.center_biases[i_center] = best_bias;
			c_bit_vecs.row_mut(i_center).assign(&best_bit_vec);
		});
		self.init_regular();
	}

	#[inline(always)]
	pub fn overlap_to_sim(&self, overlap: usize) -> f64 {
		((overlap as f64) / (self.n_data as f64) - 0.5).abs()
	}
	fn update_bits(&mut self, i_center: usize) {
		let c = self.centers.row(i_center);
		let bias = self.center_biases[i_center];
		let mut cb = self.data_bins_t.row_mut(i_center);
		#[cfg(feature="parallel")]
		#[allow(non_snake_case)]
		let TOTAL_CHUNKS_LOWER: usize = (cb.shape()[0] / rayon::current_num_threads()).max(1);
		#[cfg(not(feature="parallel"))]
		const TOTAL_CHUNKS_LOWER: usize = 200;
		let n_blocks = (TOTAL_CHUNKS_LOWER+(B::size()-1))/B::size();
		if !self.is_affine {
			par_iter(
				cb.axis_chunks_iter_mut(Axis(0), n_blocks)
				.zip(self.data.axis_chunks_iter(Axis(0), n_blocks*B::size()))
			)
			.for_each(|(mut cb_block, data_block)| {
				cb_block.iter_mut()
				.zip(data_block.axis_chunks_iter(Axis(0), B::size()))
				.for_each(|(target, points)| {
					let mut bits = B::zeros();
					points.axis_iter(Axis(0))
					.enumerate()
					.for_each(|(i_bit, point)| {
						let bit = DotProduct::prod_arrs(&c, &point) >= F::zero();
						bits.set_bit_unchecked(i_bit, bit);
					});
					*target = bits;
				});
			});
		} else if self.balance_regression_factor <= F::zero() {
			let mut dots = vec![F::zero(); self.n_data];
			par_iter(
				cb.axis_chunks_iter_mut(Axis(0), n_blocks)
				.zip(
					dots.chunks_mut(n_blocks*B::size())
					.zip(self.data.axis_chunks_iter(Axis(0), n_blocks*B::size()))
				)
			)
			.for_each(|(mut cb_block, (dots_block, data_block))| {
				cb_block.iter_mut()
				.zip(
					dots_block.chunks_mut(B::size())
					.zip(data_block.axis_chunks_iter(Axis(0), B::size()))
				)
				.for_each(|(target, (dots_chunk, points))| {
					let mut bits = B::zeros();
					points.axis_iter(Axis(0))
					.enumerate()
					.for_each(|(i_bit, point)| {
						let dot = DotProduct::prod_arrs(&c, &point);
						dots_chunk[i_bit] = dot;
						let bit = dot >= bias;
						bits.set_bit_unchecked(i_bit, bit);
					});
					*target = bits;
				});
			});
		} else {
			let mut dots = vec![F::zero(); self.n_data];
			let mut dots_for_median = vec![F::zero(); self.n_data];
			par_iter(
				dots.chunks_mut(n_blocks*B::size())
				.zip(dots_for_median.chunks_mut(n_blocks*B::size()))
				.zip(self.data.axis_chunks_iter(Axis(0), n_blocks*B::size()))
			)
			.for_each(|((dots_block, dots_for_median_block), data_block)| {
				dots_block.iter_mut()
				.zip(dots_for_median_block.iter_mut())
				.zip(data_block.axis_iter(Axis(0)))
				.for_each(|((dots_val, dots_for_median_val), point)| {
					let dot = DotProduct::prod_arrs(&c, &point);
					*dots_val = dot;
					*dots_for_median_val = dot;
				});
			});
			/* Select median, i.e., balanced bias */
			let balance_index = self.n_data/2;
			unsafe {
				dots_for_median.select_nth_unstable_by(
					balance_index,
					|a,b| a.partial_cmp(b).unwrap_unchecked()
				);
			}
			let balanced_bias = dots_for_median[balance_index];
			/* Compute balance regressed bias */
			let bias = bias + self.balance_regression_factor * (balanced_bias - bias);
			self.center_biases[i_center] = bias;
			/* Update bits with regressed bias */
			par_iter(
				cb.axis_chunks_iter_mut(Axis(0), n_blocks)
				.zip(dots.chunks_mut(n_blocks*B::size()))
			)
			.for_each(|(mut cb_block, dots_block)| {
				cb_block.iter_mut()
				.zip(dots_block.chunks_mut(B::size()))
				.for_each(|(target, dots_chunk)| {
					let mut bits = B::zeros();
					dots_chunk.iter()
					.enumerate()
					.for_each(|(i_bit, dot)| {
						let bit = dot >= &bias;
						bits.set_bit_unchecked(i_bit, bit);
					});
					*target = bits;
				});
			});
		}
	}
	fn update_overlaps(&mut self, i_center: usize) {
		unsafe { *self.sim_sums.uget_mut(i_center) = 0.0; }
		let row_i = self.data_bins_t.row(i_center);
		par_iter(self.data_bins_t.axis_iter(Axis(0)).enumerate())
		.filter(|(j_center, _)| i_center != *j_center)
		.map(|(j_center, row_j)| unsafe {
			let overlap = if i_center == j_center { 0 } else {
				self.n_data - row_i.hamming_dist_same(&row_j)
			};
			let sim = self.overlap_to_sim(overlap);
			let old_sim = *self.sim_mat.uget([i_center,j_center]);
			(j_center, overlap, sim, old_sim)
		})
		.collect::<Vec<(usize, usize, f64, f64)>>()
		.iter()
		.for_each(|(j_center, overlap, sim, old_sim)| unsafe {
			let j_center = *j_center;
			let overlap = *overlap;
			let sim = *sim;
			let old_sim = *old_sim;
			*self.sim_sums.uget_mut(i_center) += sim;
			*self.sim_sums.uget_mut(j_center) += sim - old_sim;
			*self.overlap_mat.uget_mut([i_center,j_center]) = overlap;
			*self.overlap_mat.uget_mut([j_center,i_center]) = overlap;
			*self.sim_mat.uget_mut([i_center,j_center]) = sim;
			*self.sim_mat.uget_mut([j_center,i_center]) = sim;
		});
	}

	#[inline(always)]
	unsafe fn vec_norm<D: Data<Elem=F>>(vec: &ArrayBase<D, Ix1>) -> F {
		vec.iter().map(|&v| v*v).reduce(|a,b| a+b).unwrap_unchecked().sqrt()
	}
	#[inline]
	fn displacement_vec_in_cache(&mut self, i_center: usize, j_center: usize) {
		let frac_equal = unsafe { 
			F::from(*self.overlap_mat.uget([i_center,j_center])).unwrap_unchecked()
			/ F::from(self.n_data).unwrap_unchecked()
		};
		let frac_unequal = F::one() - frac_equal;
		let rot_angle = (frac_equal-frac_unequal)*self.pi_half;
		let factor = rot_angle.mul(self.scale).tan();
		let ci = self.centers.row(i_center);
		let cj = self.centers.row(j_center);
		let prod = DotProduct::prod_arrs(&ci, &cj);
		self.calc_vec.iter_mut()
		.zip(ci.iter().zip(cj.iter()))
		.for_each(|(target,(v1,v2))| *target = *v1 * prod - *v2);
		let norm = unsafe { HIOB::vec_norm(&self.calc_vec) };
		let factor = factor/norm;
		self.calc_vec.mapv_inplace(|v| v*factor);
	}
	#[inline]
	fn agg_displacement_vec_in_cache(&mut self, i_center: usize, j_centers: Vec<usize>) {
		self.displace_vec_cache.fill(F::zero());
		let bias_i = self.center_biases[i_center];
		j_centers.iter()
		.filter(|&j_center| i_center != *j_center)
		.for_each(|j_center| {
			self.displacement_vec_in_cache(i_center, *j_center);
			self.displace_vec_cache.add_assign(&self.calc_vec);
			if self.is_affine {
				/* Calculate intersection point a of current hyperplanes */
				let center_i = self.centers.row(i_center);
				let center_j = self.centers.row(*j_center);
				let bias_j = self.center_biases[*j_center];
				let center_dot = DotProduct::prod_arrs(&center_i, &center_j);
				/* We use the definition of:
				 * intersect = mult_i * center_i + mult_j * center_j
				 * s.t. intersect.dot(center_i) = bias_i
				 * and  intersect.dot(center_j) = bias_j */
				let mult_i = (bias_j * center_dot - bias_i) / (center_dot*center_dot - F::one());
				let mult_j = (bias_i * center_dot - bias_j) / (center_dot*center_dot - F::one());
				let intersect = center_i.mapv(|v| v * mult_i) + center_j.mapv(|v| v * mult_j);
				/* Now consider the vector to which center_i will be rotated
				 * to get the preferred differential in bias */
				let mut target_vec = center_i.add(&self.calc_vec);
				let norm = unsafe { HIOB::vec_norm(&target_vec) };
				target_vec.mapv_inplace(|v| v/norm);
				let target_bias = DotProduct::prod_arrs(&target_vec, &intersect);
				self.biases_calc_cache[i_center] += target_bias - bias_i;
			}
		});
	}

	pub fn step(&mut self) {
		let mis = if !self.update_parallel {
			vec![_random_pair_value(unsafe { _argmax2(&self.sim_mat) })]
		} else {
			(0..self.n_bits).collect()
		};
		mis.iter()
		.for_each(|&mi| {
			let mjs = if !self.displace_parallel {
				unsafe { vec![_argmax1(&self.sim_mat.row(mi))] }
			} else {
				(0..self.n_bits).collect()
			};
			self.agg_displacement_vec_in_cache(mi, mjs);
			let mut row = self.centers_calc_cache.row_mut(mi);
			row.add_assign(&self.displace_vec_cache);
			let norm = unsafe { HIOB::vec_norm(&row) };
			row.mapv_inplace(|v| v/norm);
		});
		mis.iter().for_each(|&i_center| {
			self.centers.row_mut(i_center).assign(&self.centers_calc_cache.row(i_center));
			self.center_biases[i_center] = self.biases_calc_cache[i_center];
		});
		// std::mem::swap(&mut self.centers, &mut self.centers_calc_cache);
		// std::mem::swap(&mut self.center_biases, &mut self.biases_calc_cache);
		mis.iter().for_each(|&mi| {
			self.update_bits(mi);
			self.update_overlaps(mi);
		});
	}
	pub fn run(&mut self, n_steps: usize) {
		named_range(n_steps, "Improving pivot positions")
		.for_each(|_| self.step());
	}

	pub fn binarize_single<D: Data<Elem=F>+MaybeSync>(&self, query: &ArrayBase<D, Ix1>) -> Array1<B> {
		let mut bins = Array1::from_elem(self.data_bin_length, B::zeros());
		bins.iter_mut().zip(
			self.centers.axis_chunks_iter(Axis(0), B::size())
			.zip(self.center_biases.axis_chunks_iter(Axis(0), B::size()))
		)
		.for_each(|(b, (lcenters, lbiases))| {
			lcenters.axis_iter(Axis(0))
			.zip(lbiases.iter())
			.enumerate()
			.for_each(|(i_bit, (center, bias))| {
				let bit = DotProduct::prod_arrs(&query, &center) >= *bias;
				b.set_bit(i_bit, bit);
			});
		});
		bins
	}
	pub fn binarize<D: Data<Elem=F>+MaybeSync>(&self, queries: &ArrayBase<D, Ix2>) -> Array2<B> {
		let n_queries = queries.shape()[0];
		let mut bins = Array2::from_elem([n_queries, self.data_bin_length], B::zeros());
		#[cfg(feature="parallel")]
		#[allow(non_snake_case)]
		let CHUNK_SIZE: usize = queries.shape()[0] / rayon::current_num_threads();
		#[cfg(not(feature="parallel"))]
		const CHUNK_SIZE: usize = 10;
		let raw_iter = bins.axis_chunks_iter_mut(Axis(0), CHUNK_SIZE).zip(queries.axis_chunks_iter(Axis(0), CHUNK_SIZE));
		par_iter(raw_iter)
		.for_each(|(mut bins_row_chunk, query_chunk)| {
			bins_row_chunk.axis_iter_mut(Axis(0)).zip(query_chunk.axis_iter(Axis(0)))
			.for_each(|(mut bins_row, query)| {
				bins_row.iter_mut()
				.zip(
					self.centers.axis_chunks_iter(Axis(0), B::size())
					.zip(self.center_biases.axis_chunks_iter(Axis(0), B::size()))
				)
				.for_each(|(b, (lcenters, lbiases))| {
					lcenters.axis_iter(Axis(0))
					.zip(lbiases.iter())
					.enumerate()
					.for_each(|(i_bit, (center, bias))| {
						let bit = DotProduct::prod_arrs(&query, &center) >= *bias;
						b.set_bit(i_bit, bit);
					});
				});
			});
		});
		bins
	}
	
	#[cfg(feature="python")]
	pub fn binarize_h5(&self, file: &str, dataset: &str, batch_size: usize) -> Result<Array2<B>, std::fmt::Error> {
		// let data_source = read_h5_dataset(file, dataset)?;
		let mut cached_source = CachingH5PyReader::new(file.to_string(), dataset.to_string());
		// let mut cached_source = CachingH5Reader::new(file.to_string(), dataset.to_string());
		let n_total = cached_source.n_rows();
		let mut ret = Array2::from_elem(
			[n_total, self.data_bin_length],
			B::zeros()
		);
		let mut lo = 0;
		let mut hi = batch_size.min(n_total);
		let mut next_data = cached_source.get_rows_slice(lo, hi);
		let mut cached = hi;
		while cached < n_total {
			let next_lo = cached;
			let next_hi = (cached+batch_size).min(n_total);
			assert!(cached_source.prepare_rows_slice(next_lo, next_hi).is_ok());
			cached += next_hi-next_lo;
			let next_bins = self.binarize(&next_data);
			ret.slice_axis_mut(Axis(0), Slice::from(lo..hi)).axis_iter_mut(Axis(0)).zip(next_bins.axis_iter(Axis(0)))
			.for_each(|(mut row_to, row_from)| row_to.assign(&row_from));
			next_data = cached_source.get_cached().unwrap();
			lo = next_lo;
			hi = next_hi;
		}
		let next_bins = self.binarize(&next_data);
		ret.slice_axis_mut(Axis(0), Slice::from(lo..hi)).axis_iter_mut(Axis(0)).zip(next_bins.axis_iter(Axis(0)))
		.for_each(|(mut row_to, row_from)| row_to.assign(&row_from));
		Ok(ret)
	}

	/* Getters and setters */
	pub fn get_n_data(&self) -> usize { self.n_data }
	pub fn get_n_dims(&self) -> usize { self.n_dims }
	pub fn get_n_bits(&self) -> usize { self.n_bits }
	pub fn get_scale(&self) -> F { self.scale }
	pub fn set_scale(&mut self, scale: F) { self.scale = scale }
	pub fn get_balance_regression_factor(&self) -> F { self.balance_regression_factor }
	pub fn set_balance_regression_factor(&mut self, balance_regression_factor: F) { self.balance_regression_factor = balance_regression_factor }
	pub fn get_data<'a>(&'a self) -> ArrayView2<'a, F> { self.data.view() }
	pub fn get_centers<'a>(&'a self) -> ArrayView2<'a, F> { self.centers.view() }
	pub fn get_is_affine(&self) -> bool { self.is_affine }
	pub fn get_center_biases<'a>(&'a self) -> ArrayView1<'a, F> { self.center_biases.view() }
	pub fn set_center<D: Data<Elem=F>>(&mut self, i_center: usize, center: &ArrayBase<D, Ix1>) {
		self.centers.row_mut(i_center).assign(center);
		self.update_bits(i_center);
	}
	pub fn set_bias(&mut self, i_center: usize, bias: F) {
		self.center_biases[i_center] = bias;
		self.update_bits(i_center);
	}
	pub fn set_center_bias<D: Data<Elem=F>>(&mut self, i_center: usize, center: &ArrayBase<D, Ix1>, bias: F) {
		self.centers.row_mut(i_center).assign(center);
		self.center_biases[i_center] = bias;
		self.update_bits(i_center);
	}
	// pub fn get_data_bins<'a>(&'a self) -> ArrayView2<'a, B> { self.data_bins.view() }
	pub fn get_overlap_mat<'a>(&'a self) -> ArrayView2<'a, usize> { self.overlap_mat.view() }
	pub fn get_sim_mat<'a>(&'a self) -> ArrayView2<'a, f64> { self.sim_mat.view() }
	pub fn get_sim_sums<'a>(&'a self) -> ArrayView1<'a, f64> { self.sim_sums.view() }
	pub fn get_update_parallel(&self) -> bool { self.update_parallel }
	pub fn get_displace_parallel(&self) -> bool { self.displace_parallel }
	pub fn set_update_parallel(&mut self, b: bool) { self.update_parallel = b; }
	pub fn set_displace_parallel(&mut self, b: bool) { self.displace_parallel = b; }

}


pub struct StochasticHIOB<F: HIOBFloat, B: HIOBBits, D: MatrixDataSource<F>> where Array1<B>: BitVectorMut {
	wrapped_hiob: HIOB<F,B>,
	data_source: D,
	perm_gen: RandomPermutationGenerator,
	sample_size: usize,
	its_per_sample: usize,
	current_it: usize,
	noise_std: Option<F>,
}
impl<F: HIOBFloat, B: HIOBBits, D: MatrixDataSource<F>> StochasticHIOB<F,B,D> where Array1<B>: BitVectorMut {
	pub fn new(
		data_source: D,
		sample_size: usize,
		its_per_sample: usize,
		n_bits: usize,
		affine: bool,
		perm_gen_rounds: Option<usize>,
		scale: Option<F>,
		centers: Option<Array2<F>>,
		center_biases: Option<Array1<F>>,
		balance_regression_factor: Option<F>,
		init_greedy: Option<bool>,
		init_ransac: Option<bool>,
		ransac_pairs_per_bit: Option<usize>,
		ransac_sub_sample: Option<usize>,
		noise_std: Option<F>,
	) -> Self {
		let scale = scale.unwrap_or(F::one());
		let mut perm_gen = RandomPermutationGenerator::new(data_source.n_rows(), perm_gen_rounds.unwrap_or(4));
		let mut initial_data = data_source.get_rows(&perm_gen.next_usizes(sample_size));
		if noise_std.is_some() {
			let mut rng = thread_rng();
			let normal: Normal<f64> = Normal::new(
				0.,
				noise_std.unwrap().to_f64().unwrap()
			).unwrap();
			initial_data.mapv_inplace(|v| v + F::from(normal.sample(&mut rng)).unwrap());
		}
		StochasticHIOB {
			wrapped_hiob: HIOB::new(
				initial_data,
				n_bits,
				affine,
				Some(scale),
				centers,
				center_biases,
				balance_regression_factor,
				init_greedy,
				init_ransac,
				ransac_pairs_per_bit,
				ransac_sub_sample
			),
			data_source,
			perm_gen: perm_gen,
			sample_size: sample_size,
			its_per_sample: its_per_sample,
			current_it: 0,
			noise_std: noise_std,
		}
	}

	pub fn step(&mut self) {
		if self.current_it >= self.its_per_sample {
			let mut new_sample = self.data_source.get_rows(&self.perm_gen.next_usizes(self.sample_size));
			if self.noise_std.is_some() {
				let mut rng = thread_rng();
				let normal: Normal<f64> = Normal::new(
					0.,
					self.noise_std.unwrap().to_f64().unwrap()
				).unwrap();
				new_sample.mapv_inplace(|v| v + F::from(normal.sample(&mut rng)).unwrap());
			}
			let mut new_wrapped_hiob = HIOB::new(
				new_sample,
				self.wrapped_hiob.n_bits,
				self.wrapped_hiob.is_affine,
				Some(self.wrapped_hiob.scale),
				Some(self.wrapped_hiob.centers.clone()),
				Some(self.wrapped_hiob.center_biases.clone()),
				Some(self.wrapped_hiob.balance_regression_factor.clone()),
				None, None, None, None
			);
			new_wrapped_hiob.set_update_parallel(self.wrapped_hiob.get_update_parallel());
			new_wrapped_hiob.set_displace_parallel(self.wrapped_hiob.get_displace_parallel());
			self.wrapped_hiob = new_wrapped_hiob;
			self.current_it = 0;
		}
		self.wrapped_hiob.step();
		self.current_it += 1;
	}
	pub fn run(&mut self, n_steps: usize) {
		named_range(n_steps, "Improving pivot positions")
		.for_each(|_| self.step());
	}

	pub fn binarize<D2: Data<Elem=F>+MaybeSync>(&self, queries: &ArrayBase<D2, Ix2>) -> Array2<B> {
		self.wrapped_hiob.binarize(queries)
	}

	#[cfg(feature="python")]
	pub fn binarize_h5(&self, file: &str, dataset: &str, batch_size: usize) -> Result<Array2<B>, std::fmt::Error> {
		self.wrapped_hiob.binarize_h5(file, dataset, batch_size)
	}
	
	pub fn get_sample_size(&self) -> usize { self.sample_size }
	pub fn set_sample_size(&mut self, value: usize) { self.sample_size = value; }
	pub fn get_its_per_sample(&self) -> usize { self.its_per_sample }
	pub fn set_its_per_sample(&mut self, value: usize) { self.its_per_sample = value; }
	pub fn get_n_samples(&self) -> usize { self.wrapped_hiob.get_n_data() }
	pub fn get_n_dims(&self) -> usize { self.wrapped_hiob.get_n_dims() }
	pub fn get_n_bits(&self) -> usize { self.wrapped_hiob.get_n_bits() }
	pub fn get_scale(&self) -> F { self.wrapped_hiob.get_scale() }
	pub fn set_scale(&mut self, scale: F) { self.wrapped_hiob.set_scale(scale) }
	pub fn get_balance_regression_factor(&self) -> F { self.wrapped_hiob.get_balance_regression_factor() }
	pub fn set_balance_regression_factor(&mut self, balance_regression_factor: F) { self.wrapped_hiob.set_balance_regression_factor(balance_regression_factor) }
	pub fn get_data<'a>(&'a self) -> ArrayView2<'a, F> { self.wrapped_hiob.get_data() }
	pub fn get_centers<'a>(&'a self) -> ArrayView2<'a, F> { self.wrapped_hiob.get_centers() }
	pub fn get_is_affine(&self) -> bool { self.wrapped_hiob.get_is_affine() }
	pub fn get_center_biases<'a>(&'a self) -> ArrayView1<'a, F> { self.wrapped_hiob.get_center_biases() }
	pub fn set_center<D2: Data<Elem=F>>(&mut self, i_center: usize, center: &ArrayBase<D2, Ix1>) { self.wrapped_hiob.set_center(i_center, center); }
	pub fn set_bias(&mut self, i_center: usize, bias: F) { self.wrapped_hiob.set_bias(i_center, bias); }
	pub fn set_center_bias<D2: Data<Elem=F>>(&mut self, i_center: usize, center: &ArrayBase<D2, Ix1>, bias: F) { self.wrapped_hiob.set_center_bias(i_center, center, bias); }
	// pub fn get_data_bins<'a>(&'a self) -> ArrayView2<'a, B> { self.wrapped_hiob.get_data_bins() }
	pub fn get_overlap_mat<'a>(&'a self) -> ArrayView2<'a, usize> { self.wrapped_hiob.get_overlap_mat() }
	pub fn get_sim_mat<'a>(&'a self) -> ArrayView2<'a, f64> { self.wrapped_hiob.get_sim_mat() }
	pub fn get_sim_sums<'a>(&'a self) -> ArrayView1<'a, f64> { self.wrapped_hiob.get_sim_sums() }
	pub fn get_update_parallel(&self) -> bool { self.wrapped_hiob.get_update_parallel() }
	pub fn get_displace_parallel(&self) -> bool { self.wrapped_hiob.get_displace_parallel() }
	pub fn set_update_parallel(&mut self, b: bool) { self.wrapped_hiob.set_update_parallel(b); }
	pub fn set_displace_parallel(&mut self, b: bool) { self.wrapped_hiob.set_displace_parallel(b); }
	pub fn get_noise_std(&self) -> Option<F> { self.noise_std }
	pub fn set_noise_std(&mut self, value: Option<F>) { self.noise_std = value; }

}


/* Helper functions */
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
	RandomPermutationGenerator::new(max, 4).next_usizes(cnt)
	// let mut rng = rand::thread_rng();
	// let mut ret = Vec::new();
	// ret.reserve_exact(cnt);
	// (0..cnt).for_each(|i| {
	// 	ret.push((rng.next_u64() as usize) % (max-i));
	// 	(0..i).for_each(|j| unsafe {
	// 		if ret.get_unchecked(i) >= ret.get_unchecked(j) {
	// 			*ret.get_unchecked_mut(i) += 1;
	// 		}
	// 	});
	// 	ret.sort();
	// });
	// ret
}
unsafe fn _max1<T: PartialOrd, D: Data<Elem=T>>(vec: &ArrayBase<D, Ix1>) -> &T {
	vec.iter()
	.reduce(|a,b| if a >= b {a} else {b})
	.unwrap_unchecked()
}
unsafe fn _min1<T: PartialOrd, D: Data<Elem=T>>(vec: &ArrayBase<D, Ix1>) -> &T {
	vec.iter()
	.reduce(|a,b| if a <= b {a} else {b})
	.unwrap_unchecked()
}
unsafe fn _argmax1<T: PartialOrd, D: Data<Elem=T>>(vec: &ArrayBase<D, Ix1>) -> usize {
	vec.indexed_iter()
	.reduce(|(i,a),(j,b)| if a >= b {(i,a)} else {(j,b)})
	.unwrap_unchecked().0
}
unsafe fn _argmin1<T: PartialOrd, D: Data<Elem=T>>(vec: &ArrayBase<D, Ix1>) -> usize {
	vec.indexed_iter()
	.reduce(|(i,a),(j,b)| if a <= b {(i,a)} else {(j,b)})
	.unwrap_unchecked().0
}
unsafe fn _max2<T: PartialOrd, D: Data<Elem=T>>(mat: &ArrayBase<D, Ix2>) -> &T {
	mat.axis_iter(Axis(0))
	.enumerate()
	.map(|(i_row, row)| unsafe { mat.uget([i_row, _argmax1(&row)]) })
	.reduce(|a,b| if a >= b {a} else {b})
	.unwrap_unchecked()
}
unsafe fn _min2<T: PartialOrd, D: Data<Elem=T>>(mat: &ArrayBase<D, Ix2>) -> &T {
	mat.axis_iter(Axis(0))
	.enumerate()
	.map(|(i_row, row)| unsafe { mat.uget([i_row, _argmin1(&row)]) })
	.reduce(|a,b| if a <= b {a} else {b})
	.unwrap_unchecked()
}
unsafe fn _argmax2<T: PartialOrd, D: Data<Elem=T>>(mat: &ArrayBase<D, Ix2>) -> (usize,usize) {
	let sol = mat.axis_iter(Axis(0))
	.enumerate()
	.map(|(i_row, row)| unsafe {
		let max_col = _argmax1(&row);
		let max = mat.uget([i_row, max_col]);
		(i_row, max_col, max)
	})
	.reduce(|(i_row,i_col,val_i),(j_row,j_col,val_j)| if val_i >= val_j {(i_row,i_col,val_i)} else {(j_row,j_col,val_j)})
	.unwrap_unchecked();
	(sol.0, sol.1)
}
unsafe fn _argmin2<T: PartialOrd, D: Data<Elem=T>>(mat: &ArrayBase<D, Ix2>) -> (usize,usize) {
	let sol = mat.axis_iter(Axis(0))
	.enumerate()
	.map(|(i_row, row)| unsafe {
		let min_col = _argmin1(&row);
		let min = mat.uget([i_row, min_col]);
		(i_row, min_col, min)
	})
	.reduce(|(i_row,i_col,val_i),(j_row,j_col,val_j)| if val_i <= val_j {(i_row,i_col,val_i)} else {(j_row,j_col,val_j)})
	.unwrap_unchecked();
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
	let (i,j) = unsafe { _argmax2(&arr2) };
	let pred_max = unsafe { *_max2(&arr2) };
	assert!(true_max == arr2[[i,j]], "True max: {}, Via _argmax2: {}", true_max, arr2[[i,j]]);
	assert!(true_max == pred_max, "True max: {}, Via _max2: {}", true_max, pred_max);
	/* Test argmin and min */
	let true_min = arr2.iter().map(|&v|v).reduce(|a,b| if a<=b {a} else {b}).unwrap();
	let (i,j) = unsafe { _argmin2(&arr2) };
	let pred_min = unsafe { *_min2(&arr2) };
	assert!(true_min == arr2[[i,j]], "True min: {}, Via _argmin2: {}", true_min, arr2[[i,j]]);
	assert!(true_min == pred_min, "True min: {}, Via _min2: {}", true_min, pred_min);
	/* Array1 */
	let arr1: Array1<u16> = Array1::from_shape_simple_fn(100, random) % 0x8000 as u16;
	/* Test argmax and max */
	let true_max = arr1.iter().map(|&v|v).reduce(|a,b| if a>=b {a} else {b}).unwrap();
	let i = unsafe { _argmax1(&arr1) };
	let pred_max = unsafe { *_max1(&arr1) };
	assert!(true_max == arr1[i], "True max: {}, Via _argmax1: {}", true_max, arr1[i]);
	assert!(true_max == pred_max, "True max: {}, Via _max1: {}", true_max, pred_max);
	/* Test argmin and min */
	let true_min = arr1.iter().map(|&v|v).reduce(|a,b| if a<=b {a} else {b}).unwrap();
	let i = unsafe { _argmin1(&arr1) };
	let pred_min = unsafe { *_min1(&arr1) };
	assert!(true_min == arr1[i], "True min: {}, Via _argmin1: {}", true_min, arr1[i]);
	assert!(true_min == pred_min, "True min: {}, Via _min1: {}", true_min, pred_min);
}
