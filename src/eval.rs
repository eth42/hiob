use std::collections::HashSet;

use ndarray::{Data, Ix2, ArrayBase, Axis, Array2};
#[cfg(feature="parallel")]
use rayon::iter::ParallelIterator;

use crate::binarizer::{HIOBFloat, HIOBBits};
// use crate::bits::{Bits};
use crate::bit_vectors::{BitVector};
use crate::bits::Bits;
use crate::data::{MatrixDataSource, H5PyDataset};
use crate::progress::{named_par_iter, MaybeSync, par_iter};
use crate::measures::{DotProduct, InnerProduct};
use crate::heaps::{MaxHeap, MinHeap, GenericHeap};

pub struct BinarizationEvaluator {}
impl BinarizationEvaluator {

	pub fn new() -> Self { Self{} }

	pub fn brute_force_k_smallest_hamming<
		B: Bits,
		D1: Data<Elem=B>+MaybeSync,
		D2: Data<Elem=B>,
	> (
		&self,
		data: &ArrayBase<D1, Ix2>,
		queries: &ArrayBase<D2, Ix2>,
		k: usize
	) -> (Array2<usize>, Array2<usize>) {
		let mut nn_dists = Array2::zeros((queries.shape()[0], k));
		let mut nn_idxs = Array2::zeros((queries.shape()[0], k));
		let raw_iter = queries.axis_iter(Axis(0))
		.zip(nn_dists.axis_iter_mut(Axis(0)))
		.zip(nn_idxs.axis_iter_mut(Axis(0)));
		named_par_iter(raw_iter, "Computing neighbors")
		.map(|((a,b),c)| (a,b,c))
		.for_each(|(q, mut nn_dist, mut nn_idx)| {
			let mut heap = MaxHeap::<usize, usize>::new();
			heap.reserve(k);
			data.axis_iter(Axis(0))
			.enumerate()
			.for_each(|(i_row, row)| unsafe {
				let v = row.hamming_dist_same(&q);
				if heap.size() < k {
					heap.push(v, i_row);
				} else if heap.peek().unwrap_unchecked().0 > v {
					heap.pop();
					heap.push(v, i_row);
				}
			});
			heap.into_iter().zip((0..k).rev()).for_each(|((dist, idx), i_nn)| unsafe {
				*nn_dist.uget_mut(i_nn) = dist;
				*nn_idx.uget_mut(i_nn) = idx;
			})
		});
		(nn_dists, nn_idxs)
	}

	
	pub fn brute_force_k_largest_dot<
		F: HIOBFloat,
		D1: Data<Elem=F>+MaybeSync,
		D2: Data<Elem=F>,
	> (
		&self,
		data: &ArrayBase<D1, Ix2>,
		queries: &ArrayBase<D2, Ix2>,
		k: usize
	) -> (Array2<F>, Array2<usize>) {
		let prod = DotProduct::new();
		let mut nn_dots = Array2::zeros((queries.shape()[0], k));
		let mut nn_idxs = Array2::zeros((queries.shape()[0], k));
		let raw_iter = queries.axis_iter(Axis(0))
		.zip(nn_dots.axis_iter_mut(Axis(0)))
		.zip(nn_idxs.axis_iter_mut(Axis(0)));
		named_par_iter(raw_iter, "Computing neighbors")
		.map(|((a,b),c)| (a,b,c))
		.for_each(|(q, mut nn_dist, mut nn_idx)| {
			let mut heap = MinHeap::<F, usize>::new();
			heap.reserve(k);
			data.axis_iter(Axis(0))
			.enumerate()
			.for_each(|(i_row, row)| unsafe {
				let v = prod.prod(&row,&q);
				if heap.size() < k {
					heap.push(v, i_row);
				} else if heap.peek().unwrap_unchecked().0 < v {
					heap.pop();
					heap.push(v, i_row);
				}
			});
			heap.into_iter().zip((0..k).rev()).for_each(|((dist, idx), i_nn)| unsafe {
				*nn_dist.uget_mut(i_nn) = dist;
				*nn_idx.uget_mut(i_nn) = idx;
			})
		});
		(nn_dots, nn_idxs)
	}

	pub fn k_at_n_recall<
		F: HIOBFloat,
		B: HIOBBits,
		D1A: Data<Elem=F>+MaybeSync,
		D1B: Data<Elem=B>+MaybeSync,
		D2A: Data<Elem=F>,
		D2B: Data<Elem=B>,
	>(
		&self,
		data: &ArrayBase<D1A, Ix2>,
		data_bin: &ArrayBase<D1B, Ix2>,
		queries: &ArrayBase<D2A, Ix2>,
		queries_bin: &ArrayBase<D2B, Ix2>,
		k: usize,
		n: usize
	) -> f64 {
		let (_, dot_neighbors) = self.brute_force_k_largest_dot(data, queries, k);
		let (_, hamming_neighbors) = self.brute_force_k_smallest_hamming(data_bin, queries_bin, n);
		self.k_at_n_recall_prec_all(&dot_neighbors, &hamming_neighbors)
	}

	pub fn k_at_n_recall_prec_dot_neighbors<
		B: HIOBBits,
		D1: Data<Elem=B>+MaybeSync,
		D2: Data<Elem=B>,
		D3: Data<Elem=usize>,
	>(
		&self,
		data_bin: &ArrayBase<D1, Ix2>,
		queries_bin: &ArrayBase<D2, Ix2>,
		dot_neighbors: &ArrayBase<D3, Ix2>,
		n: usize
	) -> f64 {
		let (_, hamming_neighbors) = self.brute_force_k_smallest_hamming(data_bin, queries_bin, n);
		self.k_at_n_recall_prec_all(dot_neighbors, &hamming_neighbors)
	}

	pub fn k_at_n_recall_prec_hamming_neighbors<
		F: HIOBFloat,
		D1: Data<Elem=F>+MaybeSync,
		D2: Data<Elem=F>,
		D3: Data<Elem=usize>,
	>(
		&self,
		data: &ArrayBase<D1, Ix2>,
		queries: &ArrayBase<D2, Ix2>,
		hamming_neighbors: &ArrayBase<D3, Ix2>,
		k: usize
	) -> f64 {
		let (_, dot_neighbors) = self.brute_force_k_largest_dot(data, queries, k);
		self.k_at_n_recall_prec_all(&dot_neighbors, hamming_neighbors)
	}

	pub fn k_at_n_recall_prec_all<
		D1: Data<Elem=usize>,
		D2: Data<Elem=usize>,
	>(
		&self,
		true_neighbors: &ArrayBase<D1, Ix2>,
		pred_neighbors: &ArrayBase<D2, Ix2>,
	) -> f64 {
		let n_queries = true_neighbors.shape()[0];
		let k = true_neighbors.shape()[1];
		let raw_iter = true_neighbors.axis_iter(Axis(0))
		.zip(pred_neighbors.axis_iter(Axis(0)));
		let recall_cnt = named_par_iter(raw_iter, "Evaluating k@n-recall")
		.map(|(tr,pr)| {
			let set = tr.iter().map(|v| *v).collect::<HashSet<usize>>();
			pr.iter()
			.filter(|v| set.contains(*v))
			.count()
		})
		.sum::<usize>();
		recall_cnt as f64 / (n_queries * k) as f64
	}

	pub fn abstract_refine<
		F: HIOBFloat,
		TD: MatrixDataSource<F>+MaybeSync,
		TQ: Data<Elem=F>,
		TI: Data<Elem=usize>,
	>(
		&self,
		data: &TD,
		queries: &ArrayBase<TQ, Ix2>,
		hamming_ids: &ArrayBase<TI, Ix2>,
		k: usize,
		chunk_size: Option<usize>,
	) -> (Array2<F>, Array2<usize>) {
		/* Quick sanity check */
		assert!(k <= hamming_ids.shape()[1]);
		let chunk_size = chunk_size.unwrap_or(100);
		let n_queries = queries.shape()[0];
		let prod = DotProduct::<F>::new();
		let mut dot_prods = Array2::from_elem((n_queries,k), F::zero());
		let mut neighbor_ids = Array2::zeros((n_queries, k));
		let raw_iter = queries.axis_chunks_iter(Axis(0), chunk_size)
		.zip(hamming_ids.axis_chunks_iter(Axis(0), chunk_size))
		.zip(dot_prods.axis_chunks_iter_mut(Axis(0), chunk_size))
		.zip(neighbor_ids.axis_chunks_iter_mut(Axis(0), chunk_size));
		par_iter(raw_iter)
		.map(|(((a,b),c),d)| (a,b,c,d))
		.for_each(|(query_chunk, h_nn_ids_chunk, mut d_nn_dists_chunk, mut d_nn_ids_chunk)| {
			query_chunk.axis_iter(Axis(0))
			.zip(h_nn_ids_chunk.axis_iter(Axis(0)))
			.zip(d_nn_dists_chunk.axis_iter_mut(Axis(0)))
			.zip(d_nn_ids_chunk.axis_iter_mut(Axis(0)))
			.map(|(((a,b),c),d)| (a,b,c,d))
			.for_each(|(query, h_nn_ids, mut d_nn_dists, mut d_nn_ids)| {
				let candidates = data.get_rows(h_nn_ids.to_vec());
				let mut heap = MinHeap::<F, usize>::new();
				heap.reserve(k);
				candidates.axis_iter(Axis(0))
				.zip(h_nn_ids.iter())
				.for_each(|(row, &i_row)| unsafe {
					let v = prod.prod(&row,&query);
					if heap.size() < k {
						heap.push(v, i_row);
					} else if heap.peek().unwrap_unchecked().0 < v {
						heap.pop();
						heap.push(v, i_row);
					}
				});
				heap.into_iter().zip((0..k).rev()).for_each(|((dist, idx), i_nn)| unsafe {
					*d_nn_dists.uget_mut(i_nn) = dist;
					*d_nn_ids.uget_mut(i_nn) = idx;
				})
			})
		});
		(dot_prods, neighbor_ids)
	}

	pub fn refine<
		F: HIOBFloat,
		TD: Data<Elem=F>+MaybeSync,
		TQ: Data<Elem=F>,
		TI: Data<Elem=usize>,
	>(
		&self,
		data: &ArrayBase<TD, Ix2>,
		queries: &ArrayBase<TQ, Ix2>,
		hamming_ids: &ArrayBase<TI, Ix2>,
		k: usize,
		chunk_size: Option<usize>,
	) -> (Array2<F>, Array2<usize>) {
		self.abstract_refine(data, queries, hamming_ids, k, chunk_size)
	}

	pub fn refine_h5<
		F: HIOBFloat,
		TQ: Data<Elem=F>,
		TI: Data<Elem=usize>,
	>(
		&self,
		data_file: &str,
		data_dataset: &str,
		queries: &ArrayBase<TQ, Ix2>,
		hamming_ids: &ArrayBase<TI, Ix2>,
		k: usize,
		chunk_size: Option<usize>,
	) -> (Array2<F>, Array2<usize>) {
		let data = H5PyDataset::<F>::new(data_file, data_dataset);
		self.abstract_refine(&data, queries, hamming_ids, k, chunk_size)
	}

	pub fn refine_with_other_bin<
		B: HIOBBits,
		TD: Data<Elem=B>+MaybeSync,
		TQ: Data<Elem=B>,
		TI: Data<Elem=usize>,
	>(
		&self,
		data_bin: &ArrayBase<TD, Ix2>,
		queries_bin: &ArrayBase<TQ, Ix2>,
		hamming_ids: &ArrayBase<TI, Ix2>,
		k: usize,
		chunk_size: Option<usize>,
	) -> (Array2<usize>, Array2<usize>) {
		/* Quick sanity check */
		assert!(k <= hamming_ids.shape()[1]);
		let chunk_size = chunk_size.unwrap_or(100);
		let n_queries = queries_bin.shape()[0];
		let mut hamming_dists = Array2::zeros((n_queries,k));
		let mut neighbor_ids = Array2::zeros((n_queries, k));
		let raw_iter = queries_bin.axis_chunks_iter(Axis(0), chunk_size)
		.zip(hamming_ids.axis_chunks_iter(Axis(0), chunk_size))
		.zip(hamming_dists.axis_chunks_iter_mut(Axis(0), chunk_size))
		.zip(neighbor_ids.axis_chunks_iter_mut(Axis(0), chunk_size));
		par_iter(raw_iter)
		.map(|(((a,b),c),d)| (a,b,c,d))
		.for_each(|(query_chunk, h_nn_ids_chunk, mut d_nn_dists_chunk, mut d_nn_ids_chunk)| {
			query_chunk.axis_iter(Axis(0))
			.zip(h_nn_ids_chunk.axis_iter(Axis(0)))
			.zip(d_nn_dists_chunk.axis_iter_mut(Axis(0)))
			.zip(d_nn_ids_chunk.axis_iter_mut(Axis(0)))
			.map(|(((a,b),c),d)| (a,b,c,d))
			.for_each(|(query, h_nn_ids, mut d_nn_dists, mut d_nn_ids)| {
				let mut heap = MaxHeap::<usize, usize>::new();
				heap.reserve(k);
				h_nn_ids.iter()
				.map(|&i_row| (data_bin.row(i_row), i_row))
				.for_each(|(row, i_row)| unsafe {
					let v = row.hamming_dist_same(&query);
					if heap.size() < k {
						heap.push(v, i_row);
					} else if heap.peek().unwrap_unchecked().0 > v {
						heap.pop();
						heap.push(v, i_row);
					}
				});
				heap.into_iter().zip((0..k).rev()).for_each(|((dist, idx), i_nn)| unsafe {
					*d_nn_dists.uget_mut(i_nn) = dist;
					*d_nn_ids.uget_mut(i_nn) = idx;
				})
			})
		});
		(hamming_dists, neighbor_ids)
	}

	pub fn cascading_k_smallest_hamming<
		B: Bits,
		D1: Data<Elem=B>+MaybeSync,
		D2: Data<Elem=B>+MaybeSync,
	> (
		&self,
		data_bins: &Vec<ArrayBase<D1, Ix2>>,
		queries_bins: &Vec<ArrayBase<D2, Ix2>>,
		ks: &Vec<usize>,
		chunk_size: Option<usize>,
	) -> (Array2<usize>, Array2<usize>) {
		let chunk_size = chunk_size.unwrap_or(100);
		unsafe {
			let n_bins = data_bins.len();
			let n_queries = queries_bins.get_unchecked(0).shape()[0];
			let n_chunks = (n_queries + (chunk_size-1)) / chunk_size;
			let init_k = *ks.get_unchecked(0);
			let final_k = *ks.get_unchecked(ks.len()-1);
			let mut nn_dists = Array2::zeros((n_queries, *ks.last().unwrap_unchecked()));
			let mut nn_idxs = Array2::zeros((n_queries, *ks.last().unwrap_unchecked()));
			let raw_iter = (0..n_chunks)
			.zip(nn_dists.axis_chunks_iter_mut(Axis(0), chunk_size))
			.zip(nn_idxs.axis_chunks_iter_mut(Axis(0), chunk_size));
			named_par_iter(raw_iter, "Computing neighbors")
			.map(|((a,b),c)| (a,b,c))
			.for_each(|(q_id_chunk, mut nn_dist_chunk, mut nn_idx_chunk)| {
				let start_q_id = q_id_chunk * chunk_size;
				let end_q_id = ((q_id_chunk+1) * chunk_size).min(n_queries);
				(start_q_id..end_q_id)
				.zip(nn_dist_chunk.axis_iter_mut(Axis(0)))
				.zip(nn_idx_chunk.axis_iter_mut(Axis(0)))
				.map(|((a,b),c)| (a,b,c))
				.for_each(|(q_id, mut nn_dist, mut nn_idx)| {
					let init_q = queries_bins.get_unchecked(0).row(q_id);
					let mut heap = MaxHeap::<usize, usize>::new();
					heap.reserve(init_k);
					data_bins.get_unchecked(0).axis_iter(Axis(0))
					.enumerate()
					.for_each(|(i_row, row)| {
						let v = row.hamming_dist_same(&init_q);
						if heap.size() < init_k {
							heap.push(v, i_row);
						} else if heap.peek().unwrap_unchecked().0 > v {
							heap.pop();
							heap.push(v, i_row);
						}
					});
					(1..n_bins).for_each(|i_bin| {
						let q = queries_bins.get_unchecked(i_bin).row(q_id);
						let k = *ks.get_unchecked(i_bin);
						let data_bin = data_bins.get_unchecked(i_bin);
						let mut next_heap = MaxHeap::<usize, usize>::new();
						next_heap.reserve(k);
						let old_heap_size = heap.size();
						(0..old_heap_size)
						.map(|_| heap.pop().unwrap_unchecked())
						.for_each(|(old_v, i_row)| {
							let v = old_v + data_bin.row(i_row).hamming_dist_same(&q);
							if next_heap.size() < k {
								next_heap.push(v, i_row);
							} else if next_heap.peek().unwrap_unchecked().0 > v {
								next_heap.pop();
								next_heap.push(v, i_row);
							}
						});
						heap = next_heap;
					});
					heap.into_iter().zip((0..final_k).rev()).for_each(|((dist, idx), i_nn)| {
						*nn_dist.uget_mut(i_nn) = dist;
						*nn_idx.uget_mut(i_nn) = idx;
					})
				});
			});
			(nn_dists, nn_idxs)
		}
	}

}




#[test]
fn test_brute_force_k_nearest_dot() {
	/* Parameters */
	let n_data = 200;
	let n_dim = 10;
	let k_nn = 20;
	/* Additional imports */
	use ndarray_rand::rand::random;
	use ndarray::Slice;
	/* Initialize objects */
	let bin_eval = BinarizationEvaluator::new();
	let prod = DotProduct::new();
	let data: Array2<f32> = Array2::from_shape_simple_fn((n_data,n_dim), random);
	/* Compute true largest products */
	let products = prod.prods(&data.row(0), &data);
	let mut sorted_products = products.into_iter().collect::<Vec<f32>>();
	sorted_products.sort_by(|a,b| a.partial_cmp(b).unwrap());
	let sorted_products = sorted_products.split_at(n_data-k_nn).1.to_vec();
	/* Compute predicted largest products */
	let (_, nns) = bin_eval.brute_force_k_largest_dot(
		&data,
		&data.slice_axis(Axis(0), Slice::from(0..1)),
		k_nn
	);
	let mut sorted_nn_products = nns.into_iter().map(|i| prod.prod(&data.row(0), &data.row(i))).collect::<Vec<f32>>();
	sorted_nn_products.sort_by(|a,b| a.partial_cmp(b).unwrap());
	/* Assert products are the same */
	sorted_products.into_iter()
	.zip(sorted_nn_products.into_iter())
	.for_each(|(a,b)| assert!((a-b).abs() < 1e-6));
}

#[test]
fn test_brute_force_k_nearest_hamming() {
	/* Parameters */
	let n_data = 200;
	let n_dim = 10;
	let k_nn = 20;
	/* Additional imports */
	use ndarray_rand::rand::random;
	use ndarray::Slice;
	/* Initialize objects */
	let bin_eval = BinarizationEvaluator::new();
	let data: Array2<u16> = Array2::from_shape_simple_fn((n_data,n_dim), random);
	/* Compute true largest products */
	let distances = data.axis_iter(Axis(0))
	.map(|x| x.hamming_dist_same(&data.row(0)))
	.collect::<Vec<usize>>();
	let mut sorted_distances = distances;
	sorted_distances.sort_by(|a,b| a.partial_cmp(b).unwrap());
	let sorted_distances = sorted_distances.split_at(k_nn).0.to_vec();
	/* Compute predicted largest products */
	let (_, nns) = bin_eval.brute_force_k_smallest_hamming(
		&data,
		&data.slice_axis(Axis(0), Slice::from(0..1)),
		k_nn
	);
	let mut sorted_nn_distances = nns.into_iter().map(|i| data.row(i).hamming_dist_same(&data.row(0))).collect::<Vec<usize>>();
	sorted_nn_distances.sort_by(|a,b| a.partial_cmp(b).unwrap());
	/* Assert products are the same */
	sorted_distances.into_iter()
	.zip(sorted_nn_distances.into_iter())
	.for_each(|(a,b)| assert!(((a as isize)-(b as isize)).abs() < 1));
}

#[test]
fn test_k_at_n_recall() {
	/* Parameters */
	let n_data = 2000;
	let n_dim = 20;
	let k_nn = 20;
	let n_bits = 10;
	/* Additional imports */
	use ndarray_rand::rand::random;
	use ndarray::concatenate;
	use ndarray::Array1;
	/* Initialize objects */
	let bin_eval = BinarizationEvaluator::new();
	let data_non_nn: Array2<f32> = Array2::from_shape_simple_fn((n_data-k_nn,n_dim), random);
	let data_nn: Array2<f32> = Array2::from_shape_simple_fn((k_nn,n_dim), random) + n_dim as f32;
	let data = concatenate(Axis(0), &[data_nn.view(), data_non_nn.view()]).unwrap();
	let mut data_bin: Array2<u16> = Array2::from_shape_simple_fn((n_data, n_bits), random);
	(0..k_nn).into_iter().map(|i| i+k_nn/2).for_each(|i| data_bin.row_mut(i).assign(&Array1::zeros(n_bits)));
	let queries: Array2<f32> = Array2::ones((1, n_dim));
	let queries_bin: Array2<u16> = Array2::zeros((1, n_bits));
	let recall = bin_eval.k_at_n_recall(&data, &data_bin, &queries, &queries_bin, k_nn, k_nn);
	assert!((recall-0.5) < 0.01);
}

#[test]
fn test_k_at_n_recall_prec() {
	/* Parameters */
	let n_data = 2000;
	let k_nn = 200;
	let n_bits = 10;
	/* Additional imports */
	use ndarray_rand::rand::random;
	use ndarray::Array1;
	/* Initialize objects */
	let bin_eval = BinarizationEvaluator::new();
	let mut data_bin: Array2<u16> = Array2::from_shape_simple_fn((n_data, n_bits), random);
	(0..k_nn).into_iter().for_each(|i| data_bin.row_mut(i).assign(&Array1::zeros(n_bits)));
	let queries_bin: Array2<u16> = Array2::zeros((1, n_bits));
	let true_neighbors = (0..k_nn/2).into_iter().collect::<Vec<usize>>();
	let true_neighbors = Array2::from_shape_vec((1,true_neighbors.len()), true_neighbors).unwrap();
	let recall = bin_eval.k_at_n_recall_prec_dot_neighbors(&data_bin, &queries_bin, &true_neighbors, k_nn);
	assert!((recall-1.0) < 0.01);
	let random_neighbor = || {let r: usize = random(); r % n_data};
	let true_neighbors = Array2::from_shape_simple_fn((1, k_nn), random_neighbor);
	let recall = bin_eval.k_at_n_recall_prec_dot_neighbors(&data_bin, &queries_bin, &true_neighbors, k_nn);
	let true_recall = true_neighbors.into_iter().filter(|&i| i < k_nn).collect::<HashSet<usize>>().len() as f64 / k_nn as f64;
	assert!((recall-true_recall) < 0.01);
}

#[test]
fn test_k_at_n_recall_prec2() {
	/* Parameters */
	let n_data: usize = 2000;
	let n_queries: usize = 200;
	let k_nn: usize = 200;
	let n_nn: usize = 500;
	let n_bits: usize = 10;
	/* Additional imports */
	use ndarray_rand::rand::random;
	use ndarray::Slice;
	/* Initialize objects */
	let bin_eval = BinarizationEvaluator::new();
	let data_bin: Array2<u16> = Array2::from_shape_simple_fn((n_data, n_bits), random);
	let queries_bin: Array2<u16> = Array2::from_shape_simple_fn((n_queries, n_bits), random);
	let random_neighbor = || {let r: usize = random(); r % n_data};
	let true_neighbors: Array2<usize> = Array2::from_shape_simple_fn((n_queries, k_nn), random_neighbor);
	let total_recall = bin_eval.k_at_n_recall_prec_dot_neighbors(&data_bin, &queries_bin, &true_neighbors, n_nn);
	let agg_recall = (0..n_queries).map(|i| {
		let local_queries_bin = queries_bin.slice_axis(Axis(0), Slice::from(i..i+1));
		let local_true_neighbors = true_neighbors.slice_axis(Axis(0), Slice::from(i..i+1));
		bin_eval.k_at_n_recall_prec_dot_neighbors(&data_bin, &local_queries_bin, &local_true_neighbors, n_nn)
	})
	.sum::<f64>() / n_queries as f64;
	assert!((total_recall-agg_recall).abs() < 1e-6);
}

#[test]
fn test_cascading_k_nearest_hamming() {
	/* Parameters */
	let n_data = 2000;
	let n_queries = 100;
	/* Additional imports */
	use ndarray_rand::rand::random;
	/* Initialize objects */
	let bin_eval = BinarizationEvaluator::new();
	let mut datas: Vec<Array2<u16>> = Vec::new();
	datas.push(Array2::from_shape_simple_fn((n_data,1), random));
	datas.push(Array2::from_shape_simple_fn((n_data,2), random));
	datas.push(Array2::from_shape_simple_fn((n_data,4), random));
	let ks = vec![100, 50, 10];
	let mut queries: Vec<Array2<u16>> = Vec::new();
	queries.push(Array2::from_shape_simple_fn((n_queries,1), random));
	queries.push(Array2::from_shape_simple_fn((n_queries,2), random));
	queries.push(Array2::from_shape_simple_fn((n_queries,4), random));
	bin_eval.cascading_k_smallest_hamming(&datas, &queries, &ks, Some(10));
}
