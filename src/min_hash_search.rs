use crate::random::RandomPermutationGenerator;
use crate::{bit_vectors::BitVector, progress::par_iter};
use crate::heaps::{GenericHeap, MaxHeap};
use crate::bits::Bits;
use itertools::Itertools;
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

pub trait Bitsized {
	const BITS: usize;
}
use paste::paste;
macro_rules! impl_uint_bitsized {
	($($n:literal),*) => {
		paste! {
			$(
				impl Bitsized for [<u $n>] {
					const BITS: usize = $n;
				}
			)*
		}
	};
}
impl_uint_bitsized!(8,16,32,64,128);
impl Bitsized for bool {
	const BITS: usize = 1;
}
impl Bitsized for usize {
	const BITS: usize = usize::BITS as usize;
}

#[cfg(feature="parallel")]
pub trait Reference: Clone+Send+Sync+Bitsized {
	fn as_usize(&self) -> usize;
	fn into_usize(self) -> usize;
	fn from_usize(v: usize) -> Self;
}
#[cfg(not(feature="parallel"))]
pub trait Reference: Clone+Bitsized {
	fn as_usize(&self) -> usize;
	fn into_usize(self) -> usize;
	fn from_usize(v: usize) -> Self;
}
macro_rules! impl_reference {
	($($n:ty),*) => {
		paste! {
			$(
				impl Reference for $n {
					fn as_usize(&self) -> usize {self.clone() as usize}
					fn into_usize(self) -> usize {self as usize}
					fn from_usize(v: usize) -> Self {v as Self}
				}
			)*
		}
	};
}
impl_reference!(u8,u16,u32,u64,u128,usize);

// impl Reference for u32 {}


pub struct MinHashSearcher<'a, I: Bits, R: Reference> {
	data: &'a Array2<I>,
	hashers: Vec<BitHasher>,
	inverted_index: Vec<Vec<Vec<R>>>,
	hash_collision_table: Array2<usize>,
	max_hash_dist: usize,
}
impl<'a, I: Bits, R: Reference> MinHashSearcher<'a, I, R> {
	fn num_hashes_at_dist(num_positions: usize, dist: usize) -> usize {
		/* Rust equivalent of Pythons sum(binom(num_positions, v) for v in range(dist+1)) */
		/* Expressed as factorials: sum(fac(num_positions)/fac(v)/fac(num_positions-v) for v in range(dist+1)) */
		/* Equivalent: sum((num_positions-v+1) * ... * num_positions / 1 / ... / v) */
		(0..dist+1).map(|v| {
			let mut a = 1usize;
			let mut b = 1usize;
			(0..v).for_each(|i| {
				a *= num_positions - i;
				b *= i + 1;
			});
			a / b
		}).sum()
	}
	fn make_hash_collision_table(num_positions: usize, dist: usize) -> Array2<usize> {
		let rows = 1usize<<num_positions;
		let cols = Self::num_hashes_at_dist(num_positions, dist);
		let mut table = Array2::zeros((rows, cols));
		par_iter(table.axis_iter_mut(Axis(0)).enumerate()).for_each(|(hash, mut row)| {
			let mut offset = 0usize;
			(0..dist+1).for_each(|d| {
				(0..num_positions).combinations(d).for_each(|comb| {
					let mut other_hash = hash.clone();
					comb.iter().for_each(|&i| other_hash ^= 1<<i);
					row[offset] = other_hash;
					offset += 1;
				});
			});
			row.as_slice_mut().unwrap().sort_unstable();
		});
		table
	}
	pub fn new(data: &'a Array2<I>, num_hashes: usize, num_positions: usize) -> Self {
		Self::new_with_dist(data, num_hashes, num_positions, 0)
	}
	pub fn new_with_dist(data: &'a Array2<I>, num_hashes: usize, num_positions: usize, max_hash_dist: usize) -> Self {
		let num_bits = I::size() * data.len_of(Axis(1));
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
		let hash_collision_table = Self::make_hash_collision_table(num_positions, max_hash_dist);
		MinHashSearcher { data, hashers, inverted_index, hash_collision_table, max_hash_dist }
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
						(0..self.hash_collision_table.len_of(Axis(1))).map(move |col| {
							self.inverted_index[i_hash][self.hash_collision_table[[hash, col]]].iter()
						}).flatten()
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
	pub fn memory_footprint(&self) -> usize {
		std::mem::size_of_val(self)
		+ self.inverted_index.iter().map(|v| {
			v.iter().map(|v| {
				v.capacity() * R::BITS / 8
				+ std::mem::size_of_val(v)
			}).sum::<usize>()
			+ std::mem::size_of_val(v)
		}).sum::<usize>()
	}
	pub fn expected_size(n_data: usize, n_hashes: usize, n_pos: usize) -> usize {
		let flat_storage = n_data*n_hashes*R::BITS/8;
		let vec_base_size = std::mem::size_of::<Vec<R>>();
		let overhead = vec_base_size * (1 + n_hashes * (1 + (1<<n_pos)));
		flat_storage + overhead
	}
}



pub struct ChunkyMinHashSearcher<'a, I: Bits, R: Reference> {
	data: &'a Array2<I>,
	hashers: Vec<BitHasher>,
	/* inverted_index[i_hash][hash][i_batch][i_object] = reference */
	inverted_index: Vec<Vec<Vec<Vec<R>>>>,
	chunk_size: usize,
}
impl<'a, I: Bits, R: Reference> ChunkyMinHashSearcher<'a, I, R> {
	pub fn new(data: &'a Array2<I>, num_hashes: usize, num_positions: usize) -> Self {
		let num_bits = I::size() * data.len_of(Axis(1));
		let chunk_size = (1 as usize)<<R::BITS;
		let n_chunks = (data.len_of(Axis(0)) + chunk_size-1) / chunk_size;
		// println!("Num bits: {}, Num hashes: {}, Num positions: {}", num_bits, num_hashes, num_positions);
		let hashers: Vec<_> = (0..num_hashes).map(|_| BitHasher::new(num_bits, num_positions)).collect();
		let mut inverted_index: Vec<Vec<Vec<Vec<R>>>> = vec![vec![vec![Vec::new(); n_chunks]; 1<<num_positions]; num_hashes];
		par_iter(hashers.iter().zip(inverted_index.iter_mut()))
		.for_each(|(hasher, inv_index)| {
			data.axis_chunks_iter(Axis(0), chunk_size).enumerate().for_each(|(i_chunk, chunk)| {
				chunk.axis_iter(Axis(0)).enumerate().for_each(|(i_data, row)| {
					let hash = hasher.hash::<_,usize>(&row);
					inv_index[hash][i_chunk].push(R::from_usize(i_data));
				});
			});
			inv_index.iter_mut().for_each(|u| {
				u.iter_mut().for_each(|v| {
					v.shrink_to_fit();
				});
				u.shrink_to_fit();
			});
			inv_index.shrink_to_fit();
		});
		// /* Sanity check that all numbers are included */
		// for inv_index in &inverted_index {
		// 	assert_eq!(
		// 		inv_index.iter().map(|v| v.len()).sum::<usize>(),
		// 		data.len_of(Axis(0)),
		// 	);
		// }
		ChunkyMinHashSearcher { data, hashers, inverted_index, chunk_size }
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
						.enumerate()
						.map(|(i_chunk, chunk)| {
							let chunk_offset = i_chunk<<R::BITS;
							chunk.iter().map(move |i_data| i_data.as_usize() + chunk_offset)
						}).flatten()
					}).flatten()
					.for_each(|i_row| {
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
	pub fn memory_footprint(&self) -> usize {
		std::mem::size_of_val(self)
		+ self.inverted_index.iter().map(|v0| {
			v0.iter().map(|v1| {
				v1.iter().map(|v2| {
					v2.capacity() * R::BITS / 8
					+ std::mem::size_of_val(v2)
				}).sum::<usize>()
				+ std::mem::size_of_val(v1)
			}).sum::<usize>()
			+ std::mem::size_of_val(v0)
		}).sum::<usize>()
	}
	pub fn expected_size(n_data: usize, n_hashes: usize, n_pos: usize) -> usize {
		let flat_storage = n_data*n_hashes*R::BITS/8;
		let vec_base_size = std::mem::size_of::<Vec<R>>();
		let n_chunks = if R::BITS < <usize as Bitsized>::BITS {std::cmp::max(1,n_data / (1<<R::BITS))} else {1};
		let overhead = vec_base_size * (1 + n_hashes * (1 + (1<<n_pos) * (1 + n_chunks)));
		flat_storage + overhead
	}
	#[inline(always)]
	fn mul_chunk_size(v: usize) -> usize {
		v << R::BITS
	}
}



pub struct FlatChunkyMinHashSearcher<'a, I: Bits, R: Reference> {
	data: &'a Array2<I>,
	hashers: Vec<BitHasher>,
	/* inverted_index[
		i_hash * N
		+ i_chunk * chunk_size
		+ vec_offsets[
			i_hash * n_chunks * 2^bits
			+ i_chunk * 2^bits
			+ hash
		]
		+ i_object
	] = reference */
	inverted_index: Vec<R>,
	/* vec_offsets[
		i_hash * n_chunks * 2^bits
		+ i_chunk * 2^bits
		+ hash
	] = chunk-hash-offset */
	vec_offsets: Vec<R>,
	data_len: usize,
	num_positions: usize,
	n_chunks: usize,
	chunk_size: usize,
}
impl<'a, I: Bits, R: Reference> FlatChunkyMinHashSearcher<'a, I, R> {
	pub fn new(data: &'a Array2<I>, num_hashes: usize, num_positions: usize) -> Self {
		let num_bits = I::size() * data.len_of(Axis(1));
		let chunk_size = Self::mul_chunk_size(1);
		let data_len = data.len_of(Axis(0));
		let two_pow_pos = 1<<num_positions;
		let n_chunks = (data_len + chunk_size-1) / chunk_size;
		// println!("Num bits: {}, Num hashes: {}, Num positions: {}", num_bits, num_hashes, num_positions);
		let hashers: Vec<_> = (0..num_hashes).map(|_| BitHasher::new(num_bits, num_positions)).collect();
		/* Allocate memory for vecs */
		let mut inverted_index = Vec::<R>::with_capacity(num_hashes * data_len);
		let mut vec_offsets = Vec::<R>::with_capacity(num_hashes * n_chunks * two_pow_pos + 1);
		/* Set length of vecs, such that all indices are inside bounds */
		/* Note that fields that are not manually initialized contain random crap! */
		unsafe {
			inverted_index.set_len(inverted_index.capacity());
			vec_offsets.set_len(vec_offsets.capacity() - 1);
		}
		par_iter(
			hashers.iter()
			.zip(inverted_index.chunks_mut(data_len))
			.zip(vec_offsets.chunks_mut(n_chunks * two_pow_pos))
		).for_each(|((hasher, inv_idx_chunk), vec_offset_chunk)| {
			data.axis_chunks_iter(Axis(0), chunk_size)
			.zip(inv_idx_chunk.chunks_mut(chunk_size))
			.zip(vec_offset_chunk.chunks_mut(two_pow_pos))
			.for_each(|((chunk, inv_idx_chunk), vec_offset_chunk)| {
				// assert!(chunk.len_of(Axis(0)) == inv_idx_chunk.len());
				/* Compute hash values and their counts */
				/* TODO: `hash_counts` and `hashes` can be initialized globally to reduce allocations, but probably not worth it. */
				let mut hash_counts = vec![0usize; two_pow_pos];
				let hashes: Vec<usize> = chunk.axis_iter(Axis(0)).map(|row| {
					let hash = hasher.hash::<_,usize>(&row);
					hash_counts[hash] += 1;
					hash
				}).collect();
				/* Write offsets for each hash value into the global vec offsets vec */
				vec_offset_chunk[0] = R::from_usize(0);
				(1..two_pow_pos).for_each(|i| {
					let offset_usize = vec_offset_chunk[i-1].as_usize() + hash_counts[i-1];
					vec_offset_chunk[i] = R::from_usize(
						if offset_usize < (1<<R::BITS) { offset_usize } else { 0 }
					);
				});
				/* Write data indices into the global inverted index vec */
				hashes.iter().enumerate().for_each(|(i_data, &hash)| {
					hash_counts[hash] -= 1;
					inv_idx_chunk[vec_offset_chunk[hash].as_usize() + hash_counts[hash]] = R::from_usize(i_data);
				});
			});
		});
		/* Append a zero to have an end for slicing */
		vec_offsets.push(R::from_usize(0));
		FlatChunkyMinHashSearcher { data, hashers, inverted_index, vec_offsets, data_len, num_positions, n_chunks, chunk_size }
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
			(raw_iter)
			.map(|((a,b),c)| (a,b,c))
			.for_each(|(q_id_chunk, mut nn_dist_chunk, mut nn_idx_chunk)| {
				let mut heap: MaxHeap<usize,usize> = MaxHeap::<usize,usize>::new();
				heap.reserve(ext_k);
				let mut inclusion_set: std::collections::HashSet<usize> = std::collections::HashSet::with_capacity(ext_k);
				let mut insertion_idx_cache = vec![0usize; ext_k];
				let mut insertion_dist_cache = vec![0usize; ext_k];
				let start_q_id: usize = q_id_chunk * chunk_size;
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
						(0..self.n_chunks).map(move |i_chunk| {
							let (start, end) = self.idx_slice_limits(i_hash, i_chunk, hash);
							(start..end)
							.map(move |i_object| self.inverted_index[i_object].as_usize() + Self::mul_chunk_size(i_chunk))
						}).flatten()
					}).flatten()
					.for_each(|i_row| {
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
	pub fn memory_footprint(&self) -> usize {
		std::mem::size_of_val(self)
		+ self.inverted_index.capacity() * R::BITS / 8
		+ self.vec_offsets.capacity() * R::BITS / 8
	}
	pub fn expected_size(n_data: usize, n_hashes: usize, n_pos: usize) -> usize {
		let r_bytes = R::BITS / 8;
		let flat_storage = n_data*n_hashes*r_bytes;
		let vec_base_size = std::mem::size_of::<Vec<R>>();
		let n_chunks = if R::BITS < <usize as Bitsized>::BITS {std::cmp::max(1,n_data / (1<<R::BITS))} else {1};
		let overhead = n_hashes * n_chunks * (1<<n_pos) * r_bytes;
		flat_storage + overhead + 2 * vec_base_size + 4 * r_bytes
	}
	#[inline(always)]
	fn idx_slice_limits(&self, i_hash: usize, i_chunk: usize, hash: usize) -> (usize,usize) {
		let chunk_offset = i_hash * self.data_len + Self::mul_chunk_size(i_chunk);
		let hash_offset_idx = self.offset_idx(i_hash, i_chunk, hash);
		let start = chunk_offset + unsafe {self.vec_offsets.get_unchecked(hash_offset_idx)}.as_usize();
		let end = chunk_offset + unsafe {self.vec_offsets.get_unchecked(hash_offset_idx+1)}.as_usize();
		let end = end.max(start);
		(start, end)
	}
	#[inline(always)]
	fn idx_slice(&self, i_hash: usize, i_chunk: usize, hash: usize) -> &[R] {
		let (start, end) = self.idx_slice_limits(i_hash, i_chunk, hash);
		&self.inverted_index[start..end]
	}
	#[inline(always)]
	fn mut_idx_slice(&mut self, i_hash: usize, i_chunk: usize, hash: usize) -> &mut [R] {
		let (start, end) = self.idx_slice_limits(i_hash, i_chunk, hash);
		&mut self.inverted_index[start..end]
	}
	#[inline(always)]
	fn offset_idx(&self, i_hash: usize, i_chunk: usize, hash: usize) -> usize {
		self.mul_two_pow_pos(i_hash * self.n_chunks + i_chunk) + hash
	}
	#[inline(always)]
	fn mul_chunk_size(v: usize) -> usize {
		v << R::BITS
	}
	#[inline(always)]
	fn mul_two_pow_pos(&self, v: usize) -> usize {
		v << self.num_positions
	}
}


#[cfg(test)]
struct Timer {
	start: std::time::Instant,
}
#[cfg(test)]
impl Timer {
	fn new() -> Self {
		Timer{start: std::time::Instant::now()}
	}
	fn elapsed_s(&self) -> f64 {
		self.start.elapsed().as_secs_f64()
	}
	fn elapsed_str(&self) -> String {
		time_format(self.start.elapsed().as_secs_f64())
	}
}
#[cfg(test)]
fn time_format(seconds: f64) -> String {
	let ms = ((seconds%1f64)*1000f64).floor();
	let s = (seconds%60f64).floor();
	let m = ((seconds/60f64)%60f64).floor();
	let h = (seconds/3600f64).floor();
	match (s < 1f64, m < 1f64, h < 1f64) {
		(true, _, _) => format!("{:.0}ms", ms),
		(false, true, _) => format!("{:.0}s{:03.0}ms", s, ms),
		(false, false, true) => format!("{:.0}m{:02.0}s{:03.0}ms", m, s, ms),
		(false, false, false) => format!("{:.0}h{:02.0}m{:02.0}s{:03.0}ms", h, m, s, ms),
	}
}


#[test]
fn function_test_flatchunky() {
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
	let data = Array2::from_shape_fn((300_000, n_nums), |_| random_u64());
	let searcher: FlatChunkyMinHashSearcher<u64,u8> = FlatChunkyMinHashSearcher::new(
		&data,
		12,
		5,
	);
	searcher.hashers.iter().enumerate().for_each(|(i_hash, hasher)| {
		(0..searcher.n_chunks).for_each(|i_chunk| {
			(0..1<<searcher.num_positions).for_each(|hash| {
				let idx_slice = searcher.idx_slice(i_hash, i_chunk, hash);
				idx_slice.iter()
				.map(|v| v.clone().as_usize() + i_chunk*searcher.chunk_size)
				.for_each(|v| {
					assert_eq!(
						hash, hasher.hash(&data.row(v)),
						"i_hash: {}, i_chunk: {}, hash: {}, v: {}, offset: {}",
						i_hash, i_chunk, hash, v, searcher.vec_offsets[searcher.offset_idx(i_hash, i_chunk, hash)]);
				});
			});
		});
	});
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
	let brute_force_timer = Timer::new();
	let (_true_nn_dists, true_nn_idxs) = BinarizationEvaluator::new().brute_force_k_smallest_hamming(&data, &queries, k_neighbors, Some(chunk_size));
	println!("Brute force time: {}", brute_force_timer.elapsed_str());
	vec![
		(12usize,5usize,0usize),
		(13usize,9usize,1usize),
		(14usize,9usize,1usize),
	].into_iter().for_each(|(num_hashes, num_positions, max_hash_dist)| {
		let searcher_build_timer = Timer::new();
		let searcher: MinHashSearcher<u64,u32> = MinHashSearcher::new_with_dist(
			&data, num_hashes, num_positions, max_hash_dist,
		);
		println!("Searcher parameters: {:?}", (num_hashes, num_positions, max_hash_dist));
		println!("Searcher build time: {}", searcher_build_timer.elapsed_str());
		println!("Memory footprint: {}", searcher.memory_footprint());
		let search_timer = Timer::new();
		let (_, nn_idxs) = searcher.query(&queries, k_neighbors, Some(chunk_size));
		println!("Search time: {}", search_timer.elapsed_str());
		let recall = true_nn_idxs.axis_iter(Axis(0))
		.zip(nn_idxs.axis_iter(Axis(0)))
		.map(|(true_nn, est_nn)| {
			let true_hashset = true_nn.iter().collect::<std::collections::HashSet<_>>();
			let est_hashset = est_nn.iter().collect::<std::collections::HashSet<_>>();
			let n_correct = true_hashset.intersection(&est_hashset).count();
			n_correct as f64 / true_nn.len() as f64
		}).sum::<f64>() / queries.len_of(Axis(0)) as f64;
		println!("Recall: {}", recall);
	});
	// let mut dist_counts = vec![0; n_bits+1];
	// _true_nn_dists.iter().for_each(|dist| dist_counts[*dist as usize] += 1);
	// println!("True dist counts: {:?}", dist_counts);
}

#[test]
fn test_chunky_min_hash_search() {
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
	let brute_force_timer = Timer::new();
	let (true_nn_dists, true_nn_idxs) = BinarizationEvaluator::new().brute_force_k_smallest_hamming(&data, &queries, k_neighbors, Some(chunk_size));
	println!("Brute force time: {}", brute_force_timer.elapsed_str());

	let searcher_build_timer = Timer::new();
	let searcher: ChunkyMinHashSearcher<u64,u32> = ChunkyMinHashSearcher::new(
		&data,
		12,
		5,
	);
	println!("Searcher build time: {}", searcher_build_timer.elapsed_str());
	println!("Memory footprint: {} {}", searcher.memory_footprint(), ChunkyMinHashSearcher::<'_,u64,u32>::expected_size(data.len_of(Axis(0)), 12, 5));
	let search_timer = Timer::new();
	let (_, nn_idxs) = searcher.query(&queries, k_neighbors, Some(chunk_size));
	println!("Search time: {}", search_timer.elapsed_str());
	let recall = true_nn_idxs.axis_iter(Axis(0))
	.zip(nn_idxs.axis_iter(Axis(0)))
	.map(|(true_nn, est_nn)| {
		let true_hashset = true_nn.iter().collect::<std::collections::HashSet<_>>();
		let est_hashset = est_nn.iter().collect::<std::collections::HashSet<_>>();
		let n_correct = true_hashset.intersection(&est_hashset).count();
		n_correct as f64 / true_nn.len() as f64
	}).sum::<f64>() / queries.len_of(Axis(0)) as f64;
	println!("Recall: {}", recall);
	
	let searcher_build_timer = Timer::new();
	let searcher: ChunkyMinHashSearcher<u64,u16> = ChunkyMinHashSearcher::new(
		&data,
		12,
		5,
	);
	println!("Searcher build time: {}", searcher_build_timer.elapsed_str());
	println!("Memory footprint: {} {}", searcher.memory_footprint(), ChunkyMinHashSearcher::<'_,u64,u16>::expected_size(data.len_of(Axis(0)), 12, 5));
	let search_timer = Timer::new();
	let (_, nn_idxs) = searcher.query(&queries, k_neighbors, Some(chunk_size));
	println!("Search time: {}", search_timer.elapsed_str());
	let recall = true_nn_idxs.axis_iter(Axis(0))
	.zip(nn_idxs.axis_iter(Axis(0)))
	.map(|(true_nn, est_nn)| {
		let true_hashset = true_nn.iter().collect::<std::collections::HashSet<_>>();
		let est_hashset = est_nn.iter().collect::<std::collections::HashSet<_>>();
		let n_correct = true_hashset.intersection(&est_hashset).count();
		n_correct as f64 / true_nn.len() as f64
	}).sum::<f64>() / queries.len_of(Axis(0)) as f64;
	println!("Recall: {}", recall);
	
	// let searcher_build_timer = Timer::new();
	// let searcher: FlatChunkyMinHashSearcher<u64,u16> = FlatChunkyMinHashSearcher::new(
	// 	&data,
	// 	12,
	// 	5,
	// );
	// println!("Searcher build time: {}", searcher_build_timer.elapsed_str());
	// println!("Memory footprint: {} {}", searcher.memory_footprint(), FlatChunkyMinHashSearcher::<'_,u64,u16>::expected_size(data.len_of(Axis(0)), 12, 5));
	// let search_timer = Timer::new();
	// let (_, nn_idxs) = searcher.query(&queries, k_neighbors, Some(chunk_size));
	// println!("Search time: {}", search_timer.elapsed_str());
	// let recall = true_nn_idxs.axis_iter(Axis(0))
	// .zip(nn_idxs.axis_iter(Axis(0)))
	// .map(|(true_nn, est_nn)| {
	// 	let true_hashset = true_nn.iter().collect::<std::collections::HashSet<_>>();
	// 	let est_hashset = est_nn.iter().collect::<std::collections::HashSet<_>>();
	// 	let n_correct = true_hashset.intersection(&est_hashset).count();
	// 	n_correct as f64 / true_nn.len() as f64
	// }).sum::<f64>() / queries.len_of(Axis(0)) as f64;
	// println!("Recall: {}", recall);

	
	let searcher_build_timer = Timer::new();
	let searcher: ChunkyMinHashSearcher<u64,u8> = ChunkyMinHashSearcher::new(
		&data,
		12,
		5,
	);
	println!("Searcher build time: {}", searcher_build_timer.elapsed_str());
	println!("Memory footprint: {} {}", searcher.memory_footprint(), ChunkyMinHashSearcher::<'_,u64,u8>::expected_size(data.len_of(Axis(0)), 12, 5));
	let search_timer = Timer::new();
	let (_, nn_idxs) = searcher.query(&queries, k_neighbors, Some(chunk_size));
	println!("Search time: {}", search_timer.elapsed_str());
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


#[test]
pub fn test_vec_size() {
	println!("Vec<usize> 10: {}", std::mem::size_of_val(&vec![0usize; 10]));
	println!("Vec<usize> 20: {}", std::mem::size_of_val(&vec![0usize; 20]));
	println!("Vec<Vec<usize>> 10x20: {}", std::mem::size_of_val(&vec![vec![0usize; 20];10]));
}
