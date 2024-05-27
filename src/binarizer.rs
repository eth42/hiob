use std::{ops::{Sub, AddAssign, Add}, f64::consts::PI};

use ndarray::s;
use ndarray_rand::rand_distr::{Normal,Distribution};
use ndarray_linalg::{Eigh, SVD, UPLO};
use ndarray_stats::CorrelationExt;
#[cfg(feature="rust-hdf5")]
use hdf5::H5Type;
use ndarray::{Axis, Array2, Array1, ArrayBase, Ix1, Data, Ix2, ArrayView2, ArrayView1};
use rand::thread_rng;
// use rand::prelude::*;
#[cfg(feature="parallel")]
use rayon::iter::ParallelIterator;

use crate::{
	bit_vectors::{BitVector, BitVectorMut}, bits::Bits, data::{DPSParams, DatasourcePermutationSampler, MatrixDataSource}, float_vectors::{DotProduct, InnerProduct, SqEuclidean, VectorDistance}, inversion::{cap_to_ball, SphericalInverterParams}, matrices::SymArgmaxMatrix, progress::{named_range, par_iter}, random::RandomPermutationGenerator, types::{HIOBBits, HIOBFloat, MaybeSync}, vec_math::vec_norm
};
#[cfg(feature="python")]
use {
	crate::{
		pydata::CachingH5PyReader,
		data::AsyncMatrixDataSource,
	},
	ndarray::Slice,
};

use paste::paste;
macro_rules! get_gen {
	() => {};
	/* @params value: int, ... */
	(@$attribute: ident $field: ident: $type: ty $(, $($rest: tt)+)?) => {
		paste! {
			pub fn [<get_ $field>](&self) -> $type { self.$attribute.[<get_ $field>]() }
		}
		$(get_gen!($($rest)+);)?
	};
	/* params value: int, ... */
	($attribute: ident $field: ident: $type: ty $(, $($rest: tt)+)?) => {
		paste! {
			pub fn [<get_ $field>](&self) -> $type { self.$attribute.$field }
		}
		$(get_gen!($($rest)+);)?
	};
	/* value: int, ... */
	($field: ident: $type: ty $(, $($rest: tt)+)?) => {
		paste! {
			pub fn [<get_ $field>](&self) -> $type { self.$field }
		}
		$(get_gen!($($rest)+);)?
	};
}
macro_rules! set_gen {
	() => {};
	/* @params value: int, ... */
	(@$attribute: ident $field: ident: $type: ty $(, $($rest: tt)+)?) => {
		paste! {
			pub fn [<set_ $field>](&mut self, $field: $type) { self.$attribute.[<set_ $field>]($field) }
		}
		$(set_gen!($($rest)+);)?
	};
	/* params value: int, ... */
	($attribute: ident $field: ident: $type: ty $(, $($rest: tt)+)?) => {
		paste! {
			pub fn [<set_ $field>](&mut self, $field: $type) { self.$attribute.$field = $field }
		}
		$(set_gen!($($rest)+);)?
	};
	/* value: int, ... */
	($field: ident: $type: ty $(, $($rest: tt)+)?) => {
		paste! {
			pub fn [<set_ $field>](&mut self, $field: $type) { self.$field = $field }
		}
		$(set_gen!($($rest)+);)?
	};
}
macro_rules! get_set_gen {
	($($rest: tt)+) => {
		get_gen!($($rest)+);
		set_gen!($($rest)+);
	};
}
macro_rules! get_view_gen {
	() => {};
	/* @ident params value: int, ... */
	(@$attribute: ident $field: ident: $dim: literal $type: ty $(, $($rest: tt)+)?) => {
		paste! {
			pub fn [<get_ $field>]<'a>(&'a self) -> [<ArrayView $dim>]<'a, $type> { self.$attribute.[<get_ $field>]() }
		}
		$(get_view_gen!($($rest)+);)?
	};
	/* param value: int, ... */
	($field: ident: $dim: literal $type: ty $(, $($rest: tt)+)?) => {
		paste! {
			pub fn [<get_ $field>]<'a>(&'a self) -> [<ArrayView $dim>]<'a, $type> { self.$field.view() }
		}
		$(get_view_gen!($($rest)+);)?
	};
}



/* Basic HIOB implementation */
crate::types::param_struct!(HIOBParams[Clone]<F: HIOBFloat> {
	affine: bool = false,
	scale: F = F::one(),
	centers: Option<Array2<F>> = None,
	center_biases: Option<Array1<F>> = None,
	balance_regression_factor: F = F::zero(),
	init_greedy: bool = false,
	init_ransac: bool = false,
	init_itq: bool = false,
	ransac_pairs_per_bit: usize = 200,
	ransac_sub_sample: usize = 2000,
	itq_central: bool = false,
	itq_iterations: usize = 50,
	update_parallel: bool = false,
	displace_parallel: bool = false,
});
pub struct HIOB<F: HIOBFloat, B: HIOBBits> where Array1<B>: BitVectorMut {
	n_data: usize,
	n_dims: usize,
	n_bits: usize,
	data: Array2<F>,
	params: HIOBParams<F>,
	centers: Array2<F>,
	data_bin_length: usize,
	// center_bin_length: usize,
	// data_bins: Array2<B>,
	data_bins_t: Array2<B>,
	overlap_mat: Array2<usize>,
	sim_mat: SymArgmaxMatrix<f64>,
	sim_sums: Array1<f64>,
	pi_half: F,
	displace_vec_cache: Array1<F>,
	calc_vec: Array1<F>,
	centers_calc_cache: Array2<F>,
	center_biases: Array1<F>,
	biases_calc_cache: Array1<F>,
}
impl<F: HIOBFloat, B: HIOBBits> HIOB<F, B> where Array1<B>: BitVectorMut {
	pub fn new(data_in: Array2<F>, n_bits: usize, params: HIOBParams<F>) -> HIOB<F, B> {
		let n = data_in.shape()[0];
		let d = data_in.shape()[1];
		/* Calculate the number of instances of B to accommodate >=n_bits bits */
		let data_bin_length = n_bits / B::size() + (if n_bits % B::size() > 0 {1} else {0});
		/* Calculate the number of instances of B to accommodate >=n bits */
		let center_bin_length = n / B::size() + (if n % B::size() > 0 {1} else {0});
		let (centers, center_biases) = if params.centers.is_some() {
			let centers = params.centers.as_ref().unwrap().view().to_owned();
			let biases = if params.center_biases.is_some() {
				params.center_biases.as_ref().unwrap().view().to_owned()
			} else {
				Array1::from_elem(n_bits, F::zero())
			};
			(centers,biases)
		} else if params.init_greedy {
			GreedyInitializer::new(None).init_hyperplanes(&data_in, n_bits, params.affine)
		} else if params.init_ransac {
			RansacInitializer::new(None, Some(params.ransac_pairs_per_bit), Some(params.ransac_sub_sample)).init_hyperplanes(&data_in, n_bits, params.affine)
		} else if params.init_itq {
			ITQInitializer::new(Some(params.itq_central), Some(params.itq_iterations), None).init_hyperplanes(&data_in, n_bits, params.affine)
		} else {
			RandomInitializer::new(None).init_hyperplanes(&data_in, n_bits, params.affine)
		};
		let centers_calc_cache = centers.view().to_owned();
		let biases_calc_cache = center_biases.view().to_owned();
		/* Create instance */
		let mut ret = HIOB {
			n_data: n,
			n_dims: d,
			n_bits: n_bits,
			data: data_in,
			params: params,
			centers: centers,
			center_biases: center_biases,
			data_bin_length: data_bin_length,
			// center_bin_length: center_bin_length,
			// data_bins: Array2::from_elem([n, data_bin_length], B::zeros()),
			data_bins_t: Array2::from_elem([n_bits, center_bin_length], B::zeros()),
			overlap_mat: Array2::from_elem([n_bits, n_bits], n),
			sim_mat: SymArgmaxMatrix::new(Array2::from_elem([n_bits, n_bits], 0.0)),
			sim_sums: Array1::from_elem(n_bits, 0.0),
			pi_half: F::from(PI).unwrap()/F::from(2).unwrap(),
			displace_vec_cache: Array1::from_elem((d,), F::zero()),
			calc_vec: Array1::from_elem((d,),F::zero()),
			centers_calc_cache: centers_calc_cache,
			biases_calc_cache: biases_calc_cache,
		};
		for i in 0..n_bits {
			ret.update_bits(i);
			ret.update_overlaps(i);
		}
		ret
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
		if !self.params.affine {
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
		} else if self.params.balance_regression_factor <= F::zero() {
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
			let bias = bias + self.params.balance_regression_factor * (balanced_bias - bias);
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
			let old_sim = *self.sim_mat.matrix.uget([i_center,j_center]);
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
			self.sim_mat.update_value_sym(i_center, j_center, sim);
		});
	}

	#[inline]
	fn displacement_vec_in_cache(&mut self, i_center: usize, j_center: usize) {
		let frac_equal = unsafe { 
			F::from(*self.overlap_mat.uget([i_center,j_center])).unwrap_unchecked()
			/ F::from(self.n_data).unwrap_unchecked()
		};
		let frac_unequal = F::one() - frac_equal;
		let rot_angle = (frac_equal-frac_unequal)*self.pi_half;
		let factor = num::Float::tan(rot_angle.mul(self.params.scale));
		let ci = self.centers.row(i_center);
		let cj = self.centers.row(j_center);
		let prod = DotProduct::prod_arrs(&ci, &cj);
		self.calc_vec.iter_mut()
		.zip(ci.iter().zip(cj.iter()))
		.for_each(|(target,(v1,v2))| *target = *v1 * prod - *v2);
		let norm = unsafe { vec_norm(&self.calc_vec) };
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
			if self.params.affine {
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
				let norm = unsafe { vec_norm(&target_vec) };
				target_vec.mapv_inplace(|v| v/norm);
				let target_bias = DotProduct::prod_arrs(&target_vec, &intersect);
				self.biases_calc_cache[i_center] += target_bias - bias_i;
			}
		});
	}

	pub fn step(&mut self) {
		let mis = if !self.params.update_parallel {
			vec![_random_pair_value(self.sim_mat.get_argmax())]
		} else {
			(0..self.n_bits).collect()
		};
		mis.iter()
		.for_each(|&mi| {
			let mjs = if !self.params.displace_parallel {
				vec![self.sim_mat.get_row_argmax(mi)]
			} else {
				(0..self.n_bits).collect()
			};
			self.agg_displacement_vec_in_cache(mi, mjs);
			let mut row = self.centers_calc_cache.row_mut(mi);
			row.add_assign(&self.displace_vec_cache);
			let norm = unsafe { vec_norm(&row) };
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
	get_gen!(
		n_data: usize, n_dims: usize, n_bits: usize,
		params affine: bool, params init_greedy: bool, params init_ransac: bool
	);
	get_set_gen!(
		params scale: F, params balance_regression_factor: F,
		params update_parallel: bool, params displace_parallel: bool
	);
	get_view_gen!(
		data: 2 F, centers: 2 F, center_biases: 1 F,
		// data_bins: 2 B,
		overlap_mat: 2 usize, sim_mat: 2 f64, sim_sums: 1 f64
	);
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
}



/* Stochastic HIOB implementation */
crate::types::param_struct!(StochasticHIOBParams[Clone]<F: HIOBFloat> {
	sample_size: usize = 1024,
	its_per_sample: usize = 200,
	inversive: bool = false,
	n_inverter_init_samples: usize = 2000,
	kernelized: bool = false,
	kernel_reshape: Vec<usize> = vec![0;0],
	kernel_width: usize = 3,
	perm_gen_rounds: usize = 4,
	noise_std: Option<F> = None,
	pre_noise: bool = true,
});
pub struct StochasticHIOB<F: HIOBFloat, B: HIOBBits, D: MatrixDataSource<F>> where Array1<B>: BitVectorMut {
	wrapped_hiob: HIOB<F,B>,
	data_sampler: DatasourcePermutationSampler<F,D>,
	current_it: usize,
	params: StochasticHIOBParams<F>,
}
impl<F: HIOBFloat, B: HIOBBits, D: MatrixDataSource<F>> StochasticHIOB<F,B,D> where Array1<B>: BitVectorMut {
	pub fn new(data_source: D, n_bits: usize, params: StochasticHIOBParams<F>, hiob_params: HIOBParams<F>) -> Self {
		let mut data_sampler = if params.inversive && params.kernelized {
			DatasourcePermutationSampler::new_inversive_kernel(
				data_source,
				DPSParams::new()
				.with_noise_std(params.noise_std.clone())
				.with_pre_noise(params.pre_noise)
				.with_perm_gen_rounds(Some(params.perm_gen_rounds))
				.with_n_invert_init_samples(params.n_inverter_init_samples)
				.with_reshape(Some(params.kernel_reshape.clone()))
				.with_kernel_width(Some(params.kernel_width)),
				SphericalInverterParams::new()
			)
		} else if params.inversive {
			DatasourcePermutationSampler::new_inversive(
				data_source,
				DPSParams::new()
				.with_noise_std(params.noise_std.clone())
				.with_pre_noise(params.pre_noise)
				.with_perm_gen_rounds(Some(params.perm_gen_rounds))
				.with_n_invert_init_samples(params.n_inverter_init_samples),
				SphericalInverterParams::new()
			)
		} else if params.kernelized {
			DatasourcePermutationSampler::new_kernel(
				data_source,
				DPSParams::new()
				.with_noise_std(params.noise_std.clone())
				.with_pre_noise(params.pre_noise)
				.with_perm_gen_rounds(Some(params.perm_gen_rounds))
				.with_reshape(Some(params.kernel_reshape.clone()))
				.with_kernel_width(Some(params.kernel_width)),
			)
		} else {
			DatasourcePermutationSampler::new(
				data_source,
				DPSParams::new()
				.with_noise_std(params.noise_std.clone())
				.with_pre_noise(params.pre_noise)
				.with_perm_gen_rounds(Some(params.perm_gen_rounds)),
			)
		};
		let initial_data = data_sampler.sample(params.sample_size);
		StochasticHIOB {
			wrapped_hiob: HIOB::new(
				initial_data,
				n_bits,
				hiob_params,
			),
			data_sampler: data_sampler,
			current_it: 0,
			params: params,
		}
	}

	pub fn step(&mut self) {
		if self.current_it >= self.params.its_per_sample {
			let new_sample = self.data_sampler.sample(self.params.sample_size);
			self.wrapped_hiob = HIOB::new(
				new_sample,
				self.wrapped_hiob.n_bits,
				self.wrapped_hiob.params.clone()
				.with_centers(Some(self.wrapped_hiob.centers.clone()))
				.with_center_biases(Some(self.wrapped_hiob.center_biases.clone()))
				.with_init_greedy(false)
				.with_init_ransac(false)
				.with_init_itq(false),
			);
			self.current_it = 0;
		}
		self.wrapped_hiob.step();
		self.current_it += 1;
	}
	pub fn run(&mut self, n_steps: usize) {
		named_range(n_steps, "Improving pivot positions")
		.for_each(|_| self.step());
	}

	pub fn get_inversive_balls(&self) -> Option<(Array2<F>, Array1<F>)> {
		if self.params.inversive {
			let normals = &self.wrapped_hiob.centers;
			let biases = &self.wrapped_hiob.center_biases;
			let d = normals.shape()[1];
			let mut centers = Array2::from_elem([self.wrapped_hiob.n_bits, d-1], F::zero());
			let mut sq_radii = Array1::from_elem(self.wrapped_hiob.n_bits, F::zero());
			let inverter = self.data_sampler.inverter.as_ref().unwrap();
			let scale = inverter.scale;
			let inverter_shift = inverter.shift.as_ref().map(|a| a.clone()).unwrap_or(Array1::from_elem(d-1, F::zero()));
			par_iter(
				centers.axis_iter_mut(Axis(0))
				.zip(sq_radii.iter_mut())
				.zip(normals.axis_iter(Axis(0)))
				.zip(biases.iter())
			)
			.for_each(|(((mut center, sq_radius), normal), bias)| {
				let (lcenter, lsq_radius) = cap_to_ball(&normal, *bias, scale);
				center.assign(&(lcenter+&inverter_shift));
				*sq_radius = lsq_radius;
			});
			Some((centers, sq_radii))
		} else { None }
	}

	fn binarize_balls<D2: Data<Elem=F>+MaybeSync>(&self, queries: &ArrayBase<D2, Ix2>) -> Array2<B> {
		let (centers, sq_radii) = self.get_inversive_balls().unwrap();
		let n_queries = queries.shape()[0];
		let mut bins = Array2::from_elem([n_queries, self.wrapped_hiob.data_bin_length], B::zeros());
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
					centers.axis_chunks_iter(Axis(0), B::size())
					.zip(sq_radii.axis_chunks_iter(Axis(0), B::size()))
				)
				.for_each(|(b, (lcenters, lsq_radii))| {
					lcenters.axis_iter(Axis(0))
					.zip(lsq_radii.iter())
					.enumerate()
					.for_each(|(i_bit, (center, sq_radius))| {
						let bit = SqEuclidean::dist_arrs(&query, &center) <= *sq_radius;
						b.set_bit(i_bit, bit);
					});
				});
			});
		});
		bins
	}

	pub fn binarize<D2: Data<Elem=F>+MaybeSync>(&self, queries: &ArrayBase<D2, Ix2>) -> Array2<B> {
		if self.params.inversive {
			// self.binarize_balls(queries)
			self.wrapped_hiob.binarize(&self.data_sampler.inverter.as_ref().unwrap().invert(queries))
		} else {
			self.wrapped_hiob.binarize(queries)
		}
	}

	#[cfg(feature="python")]
	pub fn binarize_h5(&self, file: &str, dataset: &str, batch_size: usize) -> Result<Array2<B>, std::fmt::Error> {
		self.wrapped_hiob.binarize_h5(file, dataset, batch_size)
	}
	
	get_gen!(
		@data_sampler n_data: usize, @wrapped_hiob n_dims: usize, @wrapped_hiob n_bits: usize,
		@wrapped_hiob affine: bool, @wrapped_hiob init_greedy: bool, @wrapped_hiob init_ransac: bool,
		params n_inverter_init_samples: usize, current_it: usize,
		params perm_gen_rounds: usize, params inversive: bool, params kernelized: bool,
		params kernel_width: usize
		/* , wrapped_hiob: HIOB<F,B> */
	);
	get_set_gen!(
		@wrapped_hiob scale: F, @wrapped_hiob balance_regression_factor: F,
		@wrapped_hiob update_parallel: bool, @wrapped_hiob displace_parallel: bool,
		params sample_size: usize, params its_per_sample: usize,
		params noise_std: Option<F>
	);
	get_view_gen!(
		@wrapped_hiob data: 2 F, @wrapped_hiob centers: 2 F, @wrapped_hiob center_biases: 1 F,
		// data_bins: 2 B,
		@wrapped_hiob overlap_mat: 2 usize, @wrapped_hiob sim_mat: 2 f64, @wrapped_hiob sim_sums: 1 f64
	);
	pub fn get_inverter_scale(&self) -> Option<F> { self.data_sampler.inverter.as_ref().map(|a| a.scale) }
	pub fn get_inverter_shift(&self) -> Option<Array1<F>> { self.data_sampler.inverter.as_ref().map(|a| a.shift.as_ref().unwrap().clone()) }
	pub fn get_kernel_reshape(&self) -> Vec<usize> { self.params.kernel_reshape.clone() }
	pub fn set_center<D2: Data<Elem=F>>(&mut self, i_center: usize, center: &ArrayBase<D2, Ix1>) { self.wrapped_hiob.set_center(i_center, center); }
	pub fn set_bias(&mut self, i_center: usize, bias: F) { self.wrapped_hiob.set_bias(i_center, bias); }
	pub fn set_center_bias<D2: Data<Elem=F>>(&mut self, i_center: usize, center: &ArrayBase<D2, Ix1>, bias: F) { self.wrapped_hiob.set_center_bias(i_center, center, bias); }

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
}
/// Chooses a random value from a pair of values.
/// 
/// # Arguments
/// * `pair` - A pair of values
/// 
/// # Return
/// * A random value from the pair with probability 50% each
/// 
fn _random_pair_value<A>(pair: (A,A)) -> A {
	if rand::random() {pair.0} else {pair.1}
}
/// Binarizes a single query vector.
/// 
/// # Arguments
/// * `centers` - The centers of the hyperplanes
/// * `center_biases` - The biases for the hyperplanes
/// * `query` - The query to binarize
/// 
/// # Return
/// * The binarized query as `Array1<B>`
fn _binarize_single<F: HIOBFloat, B: HIOBBits, D1: Data<Elem=F>, D2: Data<Elem=F>, D3: Data<Elem=F>>(centers: &ArrayBase<D1, Ix2>, center_biases: &ArrayBase<D2, Ix1>, query: &ArrayBase<D3, Ix1>) -> Array1<B> {
	let n_bits: usize = centers.shape()[0];
	let data_bin_length = n_bits / B::size() + (if n_bits % B::size() > 0 {1} else {0});
	let mut bins = Array1::from_elem(data_bin_length, B::zeros());
	bins.iter_mut().zip(
		centers.axis_chunks_iter(Axis(0), B::size())
		.zip(center_biases.axis_chunks_iter(Axis(0), B::size()))
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


/* HIOB plane initializer types */
trait HyperplaneInitializer {
	fn init_hyperplanes<F: HIOBFloat>(&self, data: &Array2<F>, n_bits: usize, affine: bool) -> (Array2<F>, Array1<F>);
}
/* Random initialization by choosing random directional vectors in data and balanced biases if affine */
struct RandomInitializer {
	n_bias_samples: usize
}
impl RandomInitializer {
	fn new(n_bias_samples: Option<usize>) -> Self {
		RandomInitializer {n_bias_samples: n_bias_samples.unwrap_or(100)}
	}
}
impl HyperplaneInitializer for RandomInitializer {
	fn init_hyperplanes<F: HIOBFloat>(&self, data: &Array2<F>, n_bits: usize, affine: bool) -> (Array2<F>, Array1<F>) {
		let n_data = data.shape()[0];
		let n_dims = data.shape()[1];
		let mut centers = Array2::zeros([n_bits, n_dims]);
		_idx_choice(n_data, n_bits).into_iter()
		.zip(_idx_choice(n_data, n_bits).into_iter())
		.enumerate()
		.for_each(|(i_center, (p1, p2))| {
			let dir = data.row(p1).sub(&data.row(p2));
			let dirn = unsafe {vec_norm(&dir)};
			centers.row_mut(i_center).assign(&dir.mapv(|v| v/dirn));
		});
		let mut center_biases = Array1::from_elem(n_bits, F::zero());
		if affine {
			(0..n_bits).for_each(|i_center| {
				let bias_sample_idx = _idx_choice(n_data, self.n_bias_samples);
				let dot_iter = par_iter(bias_sample_idx.iter())
				.map(|&i| DotProduct::prod_arrs(&data.row(i), &centers.row(i_center)));
				#[cfg(feature="parallel")]
				let dot_sum: F = dot_iter.reduce(|| F::zero(), |a,b| a+b );
				#[cfg(not(feature="parallel"))]
				let dot_sum: F = dot_iter.reduce(|a, b| a+b )
				.unwrap();
				center_biases[i_center] = dot_sum / F::from(self.n_bias_samples).unwrap();
			});
		}
		(centers, center_biases)
	}
}
/* Similar to random initialization but trying multiple attempts and choosing the one with the lowest hamming distance */
struct GreedyInitializer {
	n_attempts: usize,
}
impl GreedyInitializer {
	fn new(n_attempts: Option<usize>) -> Self {
		GreedyInitializer {
			n_attempts: n_attempts.unwrap_or(20),
		}
	}
}
impl HyperplaneInitializer for GreedyInitializer {
	fn init_hyperplanes<F: HIOBFloat>(&self, data: &Array2<F>, n_bits: usize, affine: bool) -> (Array2<F>, Array1<F>) {
		let mut centers = Array2::from_elem([n_bits, data.shape()[1]], F::zero());
		let mut center_biases = Array1::from_elem(n_bits, F::zero());
		let n_data = data.shape()[0];
		(0..n_bits).for_each(|i_center| {
			/* Indices for endpoints of normal vector creating hyperplane */
			let mut p1: usize = 0;
			let mut p2: usize = 0;
			let mut best_hamming = usize::MAX;
			for _ in 0..self.n_attempts {
				/* Choose first endpoint at random */
				p1 = rand::random::<usize>() % n_data;
				let row1: Array1<u32> = _binarize_single(&centers, &center_biases, &data.row(p1));
				let iter = par_iter(0..n_data)
				.filter(|p2| p1 != *p2)
				.map(|p2| {
					let row2 = _binarize_single(&centers, &center_biases, &data.row(p2));
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
			let c = data.row(p1).sub(&data.row(p2));
			let cn = unsafe {vec_norm(&c)};
			centers.row_mut(i_center).assign(&c.mapv(|v| v/cn));
			if affine {
				let mid_bias = DotProduct::prod_arrs(&centers.row(i_center), &data.row(p1));
				let mid_bias = mid_bias + DotProduct::prod_arrs(&centers.row(i_center), &data.row(p2));
				center_biases[i_center] = mid_bias / F::from(2).unwrap();
			}
		});
		(centers, center_biases)
	}
}
/* RANSAC Style initialization trying multiple pairs */
struct RansacInitializer {
	n_attempts: usize,
	ransac_pairs_per_bit: usize,
	ransac_sub_sample: usize,
}
impl RansacInitializer {
	fn new(n_attempts: Option<usize>, ransac_pairs_per_bit: Option<usize>, ransac_sub_sample: Option<usize>) -> Self {
		RansacInitializer {
			n_attempts: n_attempts.unwrap_or(20),
			ransac_pairs_per_bit: ransac_pairs_per_bit.unwrap_or(10),
			ransac_sub_sample: ransac_sub_sample.unwrap_or(100),
		}
	}
}
impl HyperplaneInitializer for RansacInitializer {
	fn init_hyperplanes<F: HIOBFloat>(&self, data: &Array2<F>, n_bits: usize, affine: bool) -> (Array2<F>, Array1<F>) {
		type B = u32;
		let n_data = data.shape()[0];
		let n_dims = data.shape()[1];
		let mut centers = Array2::from_elem([n_bits, n_dims], F::zero());
		let mut center_biases = Array1::from_elem(n_bits, F::zero());
		let samples = _idx_choice(n_data, self.ransac_sub_sample);
		let n_buckets = self.ransac_sub_sample / B::size() + (if self.ransac_sub_sample % B::size() > 0 {1} else {0});
		let mut c_bit_vecs = Array2::from_elem([n_bits, n_buckets], B::zeros());
		(0..n_bits).for_each(|i_center| {
			let mut best_c: Array1<F> = Array1::from_elem(n_dims, F::zero());
			let mut best_bias: F = F::zero();
			let mut best_bit_vec: Array1<B> = Array1::from_elem(n_buckets, B::zeros());
			let mut worst_sim: f64 = f64::MAX;
			for _ in 0..self.ransac_pairs_per_bit {
				let p1 = rand::random::<usize>() % n_data;
				let mut p2 = rand::random::<usize>() % n_data;
				let mut c = data.row(p1).sub(&data.row(p2));
				let mut cn = unsafe { vec_norm(&c) };
				while cn <= F::zero() {
					p2 = rand::random::<usize>() % n_data;
					c = data.row(p1).sub(&data.row(p2));
					cn = unsafe { vec_norm(&c) };
				}
				c.mapv_inplace(|v| v/cn);
				let mut bit_vec = Array1::from_elem(n_buckets, B::zeros());
				let mut bias = F::zero();
				if !affine {
					par_iter(bit_vec.iter_mut().enumerate())
					.for_each(|(i_item, target)|
						(0..B::size()).for_each(|i_bit| {
							let i_pnt = i_item*B::size()+i_bit;
							if i_pnt < self.ransac_sub_sample {
								let prod = DotProduct::prod_arrs(
									&c,
									&data.row(unsafe { *samples.get_unchecked(i_pnt) })
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
							&data.row(*idx)
						);
					});
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
			centers.row_mut(i_center).assign(&best_c);
			center_biases[i_center] = best_bias;
			c_bit_vecs.row_mut(i_center).assign(&best_bit_vec);
		});
		(centers, center_biases)
	}
}
/* Iterative Quantization initialization */
struct ITQInitializer {
	central: bool,
	n_iterations: usize,
	n_bias_samples: usize,
}
impl ITQInitializer {
	fn new(central: Option<bool>, n_iterations: Option<usize>, n_bias_samples: Option<usize>) -> Self {
		ITQInitializer {
			central: central.unwrap_or(false),
			n_iterations: n_iterations.unwrap_or(50),
			n_bias_samples: n_bias_samples.unwrap_or(100),
		}
	}
	fn make_random_rotation<F: HIOBFloat>(&self, n_dims: usize) -> Array2<F> {
		let mut rng = thread_rng();
		let normal: Normal<f64> = Normal::new(0., 1.).unwrap();
		/* Sample normally distributed matrix */
		let rand = Array2::from_shape_fn([n_dims, n_dims], |(_,_)| F::from(normal.sample(&mut rng)).unwrap());
		/* Compute covariance */
		let cov = rand.cov(F::zero()).unwrap();
		/* Compute eigenvectors */
		let (_, eigvecs) = cov.eigh(UPLO::Upper).unwrap();
		/* Return eigenvectors */
		eigvecs.mapv(|v| F::from(v).unwrap())
	}
}
impl HyperplaneInitializer for ITQInitializer {
	fn init_hyperplanes<F: HIOBFloat>(&self, data: &Array2<F>, n_bits: usize, affine: bool) -> (Array2<F>, Array1<F>) {
		let n_data = data.shape()[0];
		let n_dims = data.shape()[1];
		assert!(n_bits <= n_dims, "Number of bits must be less or equal to number of dimensions for iterative quantization");
		let data_cov = if self.central {
			let data_mean = data.mean_axis(Axis(0)).unwrap();
			(data-data_mean).t().cov(F::zero()).unwrap()
		} else {
			data.t().cov(F::zero()).unwrap()
		};
		let (_, eigvecs) = data_cov.eigh(UPLO::Upper).unwrap();
		/* Only keep n_bits largest eigenvectors */
		let eigvecs = eigvecs.slice(s![.., n_dims-n_bits..]);
		assert_eq!(eigvecs.shape(), [n_dims, n_bits]);
		let pca_embedded = data.dot(&eigvecs);
		/* Initialize with random rotation */
		let mut rot = self.make_random_rotation(n_bits);
		for _ in 0..self.n_iterations {
			/* Embed vectors */
			let embedded = pca_embedded.dot(&rot);
			/* Crop to positive values and rescale */
			let ux = embedded.mapv(|v| if v >= F::zero() {v} else {F::zero()} * F::from(2).unwrap() - F::one());
			/* Correlation with PCA */
			let c = ux.t().dot(&pca_embedded);
			/* Compute SVD of correlation */
			let (ub, _, ua) = c.svd(true, true).unwrap();
			/* Update random rotation */
			rot = ua.unwrap().dot(&ub.unwrap()).t().to_owned();
			assert_eq!(rot.shape(), [n_bits, n_bits]);
		}
		let centers = eigvecs.dot(&rot).t().to_owned();
		if affine {
			let mut center_biases = Array1::from_elem(n_bits, F::zero());
			(0..n_bits).for_each(|i_center| {
				let bias_sample_idx = _idx_choice(n_data, self.n_bias_samples);
				let dot_iter = par_iter(bias_sample_idx.iter())
				.map(|&i| DotProduct::prod_arrs(&data.row(i), &centers.row(i_center)));
				#[cfg(feature="parallel")]
				let dot_sum: F = dot_iter.reduce(|| F::zero(), |a,b| a+b );
				#[cfg(not(feature="parallel"))]
				let dot_sum: F = dot_iter.reduce(|a, b| a+b )
				.unwrap();
				center_biases[i_center] = dot_sum / F::from(self.n_bias_samples).unwrap();
			});
			(centers, center_biases)
		} else {
			(centers, Array1::from_elem(n_bits, F::zero()))
		}
	}
}



#[test]
fn itq_init_test() {
	use ndarray_rand::RandomExt;
	use ndarray_rand::rand_distr::Uniform;
	let data = Array2::random((1000, 20), Uniform::new(-1., 1.));
	let itq = ITQInitializer::new(Some(true), Some(50), Some(100));
	let (centers, biases) = itq.init_hyperplanes(&data, 16, true);
	assert_eq!(centers.shape(), [16, 20]);
	assert_eq!(biases.shape(), [16]);
}
