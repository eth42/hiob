use crate::random::RandomPermutationGenerator;
use crate::{bit_vectors::BitVector, progress::par_iter};
use crate::heaps::{GenericHeap, MaxHeap};
use crate::bits::Bits;
use ndarray::{Array2, AssignElem, Axis};
#[cfg(feature="parallel")]
use rayon::iter::ParallelIterator;
#[cfg(feature="parallel")]
use std::marker::{Sync, Send};

struct BitHasher {
	bit_positions: Vec<usize>,
}
impl BitHasher {
	fn new(num_bits: usize, num_positions: usize) -> BitHasher {
		let mut bit_positions = Vec::with_capacity(num_positions);
		let mut rng = RandomPermutationGenerator::new(num_bits, 4);
		for _ in 0..num_positions {
			bit_positions.push(rng.next_usize());
		}
		BitHasher { bit_positions }
	}
	fn hash<V: BitVector, O: Bits>(&self, vec: &V) -> O {
		let mut ret = O::zeros();
		self.bit_positions.iter().enumerate()
		.for_each(|(i, &pos)| {
			ret.set_bit_unchecked(i, vec.get_bit_unchecked(pos))
		});
		ret
	}
}


use std::fmt::Debug;
use std::convert::{TryFrom,TryInto};
#[cfg(feature="parallel")]
pub trait Reference: TryFrom<usize>+TryInto<usize>+Clone+Send+Sync {
	fn as_usize(&self) -> usize;
	fn into_usize(self) -> usize;
	fn from_usize(v: usize) -> Self;
}
#[cfg(feature="parallel")]
impl<R: TryFrom<usize>+TryInto<usize>+Clone+Send+Sync> Reference for R where <R as TryFrom<usize>>::Error: Debug, <R as TryInto<usize>>::Error: Debug {
	fn as_usize(&self) -> usize {self.clone().into_usize()}
	fn into_usize(self) -> usize {self.try_into().unwrap()}
	fn from_usize(v: usize) -> Self {Self::try_from(v).unwrap()}
}
#[cfg(not(feature="parallel"))]
pub trait Reference: TryFrom<usize>+TryInto<usize>+Clone {
	fn as_usize(&self) -> usize;
	fn into_usize(self) -> usize;
	fn from_usize(v: usize) -> Self;
}
#[cfg(not(feature="parallel"))]
impl<R: TryFrom<usize>+TryInto<usize>+Clone> Reference for R where <R as TryFrom<usize>>::Error: Debug, <R as TryInto<usize>>::Error: Debug {
	fn as_usize(&self) -> usize {self.clone().into_usize()}
	fn into_usize(self) -> usize {self.try_into().unwrap()}
	fn from_usize(v: usize) -> Self {Self::try_from(v).unwrap()}
}

// impl Reference for u32 {}


pub struct MinHashSearcher<'a, I: Bits, R: Reference> {
	data: &'a Array2<I>,
	hashers: Vec<BitHasher>,
	inverted_index: Vec<Vec<Vec<R>>>,
}
impl<'a, I: Bits, R: Reference> MinHashSearcher<'a, I, R> {
	pub fn new(data: &'a Array2<I>, num_hashes: usize, num_positions: usize) -> Self {
		let num_bits = 64 * data.len_of(Axis(1));
		// println!("Num bits: {}, Num hashes: {}, Num positions: {}", num_bits, num_hashes, num_positions);
		let hashers: Vec<_> = (0..num_hashes).map(|_| BitHasher::new(num_bits, num_positions)).collect();
		let mut inverted_index: Vec<Vec<Vec<R>>> = vec![vec![Vec::new(); 1<<num_positions]; num_hashes];
		par_iter(hashers.iter().zip(inverted_index.iter_mut()))
		.for_each(|(hasher, inv_index)| {
			data.outer_iter().enumerate().for_each(|(i_data, row)| {
				let hash = hasher.hash::<_,usize>(&row);
				inv_index[hash].push(R::from_usize(i_data));
			});
		});
		// /* Sanity check that all numbers are included */
		// for inv_index in &inverted_index {
		// 	assert_eq!(
		// 		inv_index.iter().map(|v| v.len()).sum::<usize>(),
		// 		data.len_of(Axis(0)),
		// 	);
		// }
		MinHashSearcher { data, hashers, inverted_index }
	}
	pub fn query(&self, queries: &Array2<I>, k: usize, chunk_size: Option<usize>) -> (Array2<usize>, Array2<usize>) {
		let chunk_size = chunk_size.unwrap_or(100);
		let ext_k = k * self.hashers.len();
		unsafe {
			let n_queries = queries.len_of(Axis(0));
			let n_chunks = (n_queries + (chunk_size-1)) / chunk_size;
			let mut nn_dists = Array2::zeros((n_queries, k));
			let mut nn_idxs = Array2::zeros((n_queries, k));
			let raw_iter = (0..n_chunks)
			.zip(nn_dists.axis_chunks_iter_mut(Axis(0), chunk_size))
			.zip(nn_idxs.axis_chunks_iter_mut(Axis(0), chunk_size));
			par_iter(raw_iter)
			.map(|((a,b),c)| (a,b,c))
			.for_each(|(q_id_chunk, mut nn_dist_chunk, mut nn_idx_chunk)| {
				let mut heap: MaxHeap<usize,usize> = MaxHeap::<usize,usize>::new();
				heap.reserve(ext_k);
				let mut inclusion_set: std::collections::HashSet<usize> = std::collections::HashSet::with_capacity(ext_k);
				let mut insertion_idx_cache = vec![0usize; ext_k];
				let mut insertion_dist_cache = vec![0usize; ext_k];
				let start_q_id = q_id_chunk * chunk_size;
				let end_q_id = ((q_id_chunk+1) * chunk_size).min(n_queries);
				(start_q_id..end_q_id)
				.zip(nn_dist_chunk.axis_iter_mut(Axis(0)))
				.zip(nn_idx_chunk.axis_iter_mut(Axis(0)))
				.map(|((a,b),c)| (a,b,c))
				.for_each(|(q_id, mut nn_dist, mut nn_idx)| {
					let q = queries.row(q_id);
					heap.clear();
					self.hashers.iter().enumerate().map(|(i_hash, hasher)| {
						let hash: usize = hasher.hash(&q);
						self.inverted_index[i_hash][hash].iter()
					}).flatten()
					.for_each(|i_row| {
						let i_row: usize = i_row.as_usize();
						let row = self.data.row(i_row);
						let v = row.hamming_dist_same(&q);
						if heap.size() < ext_k {
							heap.push(v, i_row);
						} else if heap.peek().unwrap_unchecked().0 > v {
							heap.pop();
							heap.push(v, i_row);
						}
					});
					let heap_end_size: usize = heap.size();
					let mut i_nn = heap_end_size;
					while heap.size() > 0 {
						i_nn -= 1; /* Subtract first to avoid overflow in last iteration. */
						let (dist, idx) = heap.pop().unwrap_unchecked();
						insertion_dist_cache.get_unchecked_mut(i_nn).assign_elem(dist);
						insertion_idx_cache.get_unchecked_mut(i_nn).assign_elem(idx);
					}
					let mut i_nn = 0;
					let mut j_nn = 0;
					inclusion_set.clear();
					while j_nn < k && i_nn < heap_end_size {
						let next_idx = insertion_idx_cache.get_unchecked(i_nn).clone();
						if !inclusion_set.contains(&next_idx) {
							let next_dist = insertion_dist_cache.get_unchecked(i_nn).clone();
							inclusion_set.insert(next_idx);
							*nn_dist.uget_mut(j_nn) = next_dist;
							*nn_idx.uget_mut(j_nn) = next_idx;
							j_nn += 1;
						}
						i_nn += 1;
					}
				});
			});
			(nn_dists, nn_idxs)
		}
	}
}



#[test]
fn test_min_hash_search() {
	fn num_threads() -> usize {
		#[cfg(feature="parallel")]
		let result = rayon::current_num_threads();
		#[cfg(not(feature="parallel"))]
		let result = 1;
		result
	}	
	use crate::eval::BinarizationEvaluator;
	// use crate::num_threads;
	use std::fs::File;
	use std::io::Read;
	fn random_u64() -> u64 {
		let mut buf = [0u8; 8];
		let mut file = File::open("/dev/urandom").unwrap();
		file.read_exact(&mut buf).unwrap();
		u64::from_le_bytes(buf)
	}
	/* Create Array2 with random u64 */
	let n_nums = 6;
	let n_bits = n_nums*64;
	let k_neighbors = 30;
	let data = Array2::from_shape_fn((300_000, n_nums), |_| random_u64());
	let queries = Array2::from_shape_fn((10_000, n_nums), |_| random_u64());
	let chunk_size = (queries.len_of(Axis(0))+num_threads()*2-1)/(num_threads()*2);
	println!("Num bits: {}, Num data: {}, Num queries: {}", n_bits, data.len_of(Axis(0)), queries.len_of(Axis(0)));
	// let brute_force_timer = Timer::new();
	let (true_nn_dists, true_nn_idxs) = BinarizationEvaluator::new().brute_force_k_smallest_hamming(&data, &queries, k_neighbors, Some(chunk_size));
	// println!("Brute force time: {}", brute_force_timer.elapsed_str());
	// let searcher_build_timer = Timer::new();
	let searcher: MinHashSearcher<u64,u32> = MinHashSearcher::new(
		&data,
		12,
		5,
	);
	// println!("Searcher build time: {}", searcher_build_timer.elapsed_str());
	// let search_timer = Timer::new();
	let (_, nn_idxs) = searcher.query(&queries, k_neighbors, Some(chunk_size));
	// println!("Search time: {}", search_timer.elapsed_str());
	let recall = true_nn_idxs.axis_iter(Axis(0))
	.zip(nn_idxs.axis_iter(Axis(0)))
	.map(|(true_nn, est_nn)| {
		let true_hashset = true_nn.iter().collect::<std::collections::HashSet<_>>();
		let est_hashset = est_nn.iter().collect::<std::collections::HashSet<_>>();
		let n_correct = true_hashset.intersection(&est_hashset).count();
		n_correct as f64 / true_nn.len() as f64
	}).sum::<f64>() / queries.len_of(Axis(0)) as f64;
	println!("Recall: {}", recall);
	let mut dist_counts = vec![0; n_bits+1];
	true_nn_dists.iter().for_each(|dist| dist_counts[*dist as usize] += 1);
	println!("True dist counts: {:?}", dist_counts);
}


