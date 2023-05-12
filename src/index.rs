use ndarray::{Data, ArrayBase, Ix1, Ix2, Array1, Array2, Axis};
use probability::{distribution::{Binomial}, prelude::Distribution};

#[cfg(feature="parallel")]
use rayon::iter::ParallelIterator;

use crate::{
	bits::Bits,
	bit_vectors::BitVector, progress::{par_iter, MaybeSend, MaybeSync},
	heaps::{GenericHeap,MinHeap,MaxHeap}
};


/* Truncated Hamming indeX */
pub struct THX<B: Bits, D: Data<Elem=B>+MaybeSend+MaybeSync, const FANOUT: usize> {
	data: ArrayBase<D, Ix2>,
	total_bits: usize,
	n_bits_per_layer: usize,
	height: usize,
	n_nodes: usize,
	quantile_bounds: [usize; FANOUT],
	nodes: Vec<THXNode<FANOUT>>,
}
impl<B: Bits, D: Data<Elem=B>+MaybeSend+MaybeSync, const FANOUT: usize> THX<B, D, FANOUT> {
	fn compute_n_nodes_for_height(height: usize) -> usize {
		/* Formula for the number of nodes in a complete tree with
		 * h (height) = height and
		 * d (node degree) = FANOUT is 
		 * (d^h-1)/(d-1) */
		(FANOUT.pow(height as u32 + 1)-1) / (FANOUT-1)
	}
	pub fn compute_n_nodes(total_bits: usize, n_bits_per_layer: usize) -> usize {
		let height = (total_bits+n_bits_per_layer-1)/n_bits_per_layer;
		Self::compute_n_nodes_for_height(height)
	}

	#[allow(unused)]
	pub fn new(data: ArrayBase<D, Ix2>, n_bits_per_layer: usize) -> THX<B, D, FANOUT> {
		let total_bits = data.row(0).size();
		let height = (total_bits+n_bits_per_layer-1)/n_bits_per_layer;
		let n_nodes = Self::compute_n_nodes_for_height(height);
		let nodes = vec![THXNode::<FANOUT>::new(); n_nodes];
		let mut ret = THX {
			data: data,
			total_bits,
			n_bits_per_layer: n_bits_per_layer,
			height: height,
			n_nodes: n_nodes,
			quantile_bounds: [0; FANOUT],
			nodes: nodes,
		};
		ret.init();
		ret
	}

	fn init(&mut self) {
		(0..self.n_nodes)
		.for_each(|i_node| unsafe {
			if !self.is_leaf(i_node) {
				let depth = self.nodes.get_unchecked(i_node).depth;
				(0..FANOUT).for_each(|i_child| {
					let i_child = self.child_of(i_node, i_child);
					let child = self.nodes.get_unchecked_mut(i_child);
					child.depth = depth+1;
				});
				self.nodes.get_unchecked_mut(i_node).children_counts = Some([0;FANOUT]);
				self.nodes.get_unchecked_mut(i_node).first_bit = depth * self.n_bits_per_layer;
			}
		});
		self.initialize_quantile_bounds();
		self.build_node(0, (0..self.data.shape()[0]).collect());
	}
	fn initialize_quantile_bounds(&mut self) {
		/* Initializing quantile bounds (exclusive upper interval limits/inclusive lower limits for next interval).
		 * We here assume, that the distribution of bits is roughly 50% per bit and entirely independent.
		 * That would give a distribution extremely similar to the binomial distribution.
		 * Quantiles are chosen accordingly. */
		 let binomial = Binomial::new(self.n_bits_per_layer, 0.5);
		 let mut i = 0;
		 let mut bit_count = 0;
		 while i < FANOUT {
			 let target_percentile = (i+1) as f64 / FANOUT as f64;
			 while bit_count < self.n_bits_per_layer && binomial.distribution(bit_count as f64) < target_percentile {
				 bit_count += 1;
			 }
			 unsafe { *self.quantile_bounds.get_unchecked_mut(i) = bit_count; }
			 i += 1;
		 }
	}
	fn build_node(&mut self, node: usize, data_idx: Vec<usize>) {
		unsafe {
			if self.nodes.get_unchecked(node).depth == self.height {
				/* This is a leaf */
				self.nodes.get_unchecked_mut(node).stored_objects = Some(data_idx);
				self.nodes.get_unchecked_mut(node).is_leaf = true;
			} else {
				/* Get bit counts for the specified range */
				let bit_counts = self.get_counts(&data_idx, self.nodes.get_unchecked(node).first_bit);
				self.nodes.get_unchecked_mut(node).children_counts = Some([0; FANOUT]);
				/* For each child collect relevant indices and build child nodes */
				(0..FANOUT)
				.for_each(|i_child| {
					let lo = if i_child==0 {0} else {*self.quantile_bounds.get_unchecked(i_child-1)};
					let hi = *self.quantile_bounds.get_unchecked(i_child);
					let subset_indices: Vec<usize> = bit_counts.iter()
					.enumerate()
					.filter(|(_, &cnt)| lo <= cnt && cnt < hi)
					.map(|(i, _)| *data_idx.get_unchecked(i))
					.collect();
					*self.nodes.get_unchecked_mut(node).children_counts.as_mut().unwrap_unchecked().get_unchecked_mut(i_child) = subset_indices.len();
					self.build_node(self.child_of(node, i_child), subset_indices);
				});
			}
		}
	}
	fn get_counts(&self, data_idx: &Vec<usize>, first_bit: usize) -> Vec<usize> {
		let last_bit_exclusive = (first_bit+self.n_bits_per_layer).min(self.total_bits);
		let mut ret = vec![0; data_idx.len()];
		par_iter(data_idx.iter().zip(ret.iter_mut()))
		.for_each(|(&i, target)| {
			*target = self.data.row(i)
			.count_bits_range_unchecked(first_bit, last_bit_exclusive);
		});
		ret
	}


	pub fn query_approx_single<D2: Data<Elem=B>>(&self, query: &ArrayBase<D2, Ix1>, k_neighbors: usize) -> (Array1<usize>, Array1<usize>) {
		/* A min heap for the refinement of nodes in the tree structure.
		 * The priority/key corresponds to the lower bound of distances
		 * of elements in that part of the tree to the query object. */
		let mut refinement_heap: MinHeap<usize, usize> = MinHeap::new();
		refinement_heap.push(0, 0);
		let mut nn_heap: MaxHeap<usize, usize> = MaxHeap::new();
		nn_heap.reserve(k_neighbors);
		let mut upper_bound = usize::MAX;
		// let mut visited_nodes = 0;
		unsafe {
			while refinement_heap.size() > 0 {
				// visited_nodes += 1;
				/* Get the next best candidate node to search for neighbors */
				let (lower_bound, i_node) = refinement_heap.pop().unwrap_unchecked();
				/* If the lower bound for distances is larger than the current largest neighbor distance, we can return */
				if nn_heap.size() >= k_neighbors && lower_bound >= upper_bound { break; }
				/* If the current node is a leaf, test neighbors, otherwise refine with children */
				if self.is_leaf(i_node) {
					let object_idx = self.nodes.get_unchecked(i_node).stored_objects.as_ref().unwrap_unchecked();
					object_idx.iter()
					.map(|&i_vec| (i_vec, self.data.row(i_vec).hamming_dist_same(&query.view())))
					.for_each(|(i_vec, dist)| {
						if nn_heap.size() < k_neighbors {
							nn_heap.push(dist, i_vec);
							if nn_heap.size() == k_neighbors { upper_bound = dist; }
						} else if nn_heap.peek().unwrap_unchecked().0 > dist {
							nn_heap.pop();
							nn_heap.push(dist, i_vec);
							upper_bound = dist;
						}
					});
				} else {
					let first_bit = self.nodes.get_unchecked(i_node).first_bit;
					let last_bit_exclusive = (first_bit+self.n_bits_per_layer).min(self.total_bits);
					let bit_count = query.view().count_bits_range_unchecked(first_bit, last_bit_exclusive);
					(0..FANOUT).for_each(|i_child| {
						let lo = if i_child==0 {0} else {*self.quantile_bounds.get_unchecked(i_child-1)};
						let hi = *self.quantile_bounds.get_unchecked(i_child);
						let i_child = self.child_of(i_node, i_child);
						if lo <= bit_count && bit_count < hi {
							refinement_heap.push(lower_bound+0, i_child);
						} else if bit_count < lo {
							refinement_heap.push(lower_bound+lo-bit_count, i_child);
						} else {
							refinement_heap.push(lower_bound+bit_count-hi, i_child);
						}
					});
				}
			}
		}
		// println!("Visited {:>3} nodes", visited_nodes);
		let mut ret_dists = Array1::zeros(k_neighbors);
		let mut ret_idx = Array1::zeros(k_neighbors);
		(0..k_neighbors).rev().for_each(|i| unsafe {
			let (dist, idx) = nn_heap.pop().unwrap_unchecked();
			*ret_dists.uget_mut(i) = dist;
			*ret_idx.uget_mut(i) = idx;
		});
		(ret_dists, ret_idx)
	}
	pub fn query_approx<D2: Data<Elem=B>>(&self, queries: &ArrayBase<D2, Ix2>, k_neighbors: usize) -> (Array2<usize>, Array2<usize>) {
		let mut ret_dists: Array2<usize> = Array2::zeros([queries.shape()[0], k_neighbors]);
		let mut ret_idx: Array2<usize> = Array2::zeros([queries.shape()[0], k_neighbors]);
		par_iter(
			queries.axis_iter(Axis(0))
			.zip(ret_dists.axis_iter_mut(Axis(0)))
			.zip(ret_idx.axis_iter_mut(Axis(0)))
		)
		.map(|((a,b),c)| (a,b,c))
		.for_each(|(query, mut target_dist, mut target_idx)| {
			let (dists, idx) = self.query_approx_single(&query, k_neighbors);
			target_dist.assign(&dists);
			target_idx.assign(&idx);
		});
		(ret_dists, ret_idx)
	}

	pub fn query_range_approx_single<D2: Data<Elem=B>>(&self, query: &ArrayBase<D2, Ix1>, max_dist: usize) -> (Vec<usize>, Vec<usize>) {
		/* A min heap for the refinement of nodes in the tree structure.
		 * The priority/key corresponds to the lower bound of distances
		 * of elements in that part of the tree to the query object. */
		let mut refinement_heap: MinHeap<usize, usize> = MinHeap::new();
		refinement_heap.push(0, 0);
		let mut nn_heap: MaxHeap<usize, usize> = MaxHeap::new();
		// let mut visited_nodes = 0;
		unsafe {
			while refinement_heap.size() > 0 {
				// visited_nodes += 1;
				/* Get the next best candidate node to search for neighbors */
				let (lower_bound, i_node) = refinement_heap.pop().unwrap_unchecked();
				/* If the lower bound for distances is larger than the maximum distance, we can return */
				if lower_bound > max_dist { break; }
				/* If the current node is a leaf, test neighbors, otherwise refine with children */
				if self.is_leaf(i_node) {
					let object_idx = self.nodes.get_unchecked(i_node).stored_objects.as_ref().unwrap_unchecked();
					object_idx.iter()
					.map(|&i_vec| (i_vec, self.data.row(i_vec).hamming_dist_same(&query.view())))
					.filter(|(_, dist)| *dist <= max_dist)
					.for_each(|(i_vec, dist)| nn_heap.push(dist, i_vec));
				} else {
					let first_bit = self.nodes.get_unchecked(i_node).first_bit;
					let last_bit_exclusive = (first_bit+self.n_bits_per_layer).min(self.total_bits);
					let bit_count = query.view().count_bits_range_unchecked(first_bit, last_bit_exclusive);
					(0..FANOUT).for_each(|i_child| {
						let lo = if i_child==0 {0} else {*self.quantile_bounds.get_unchecked(i_child-1)};
						let hi = *self.quantile_bounds.get_unchecked(i_child);
						let i_child = self.child_of(i_node, i_child);
						if lo <= bit_count && bit_count < hi {
							refinement_heap.push(lower_bound+0, i_child);
						} else if bit_count < lo {
							refinement_heap.push(lower_bound+lo-bit_count, i_child);
						} else {
							refinement_heap.push(lower_bound+bit_count-hi, i_child);
						}
					});
				}
			}
		}
		// println!("Visited {:>3} nodes", visited_nodes);
		let heap_size = nn_heap.size();
		let mut ret_dists = vec![0; heap_size];
		let mut ret_idx = vec![0; heap_size];
		(0..heap_size).rev().for_each(|i| unsafe {
			let (dist, idx) = nn_heap.pop().unwrap_unchecked();
			*ret_dists.get_unchecked_mut(i) = dist;
			*ret_idx.get_unchecked_mut(i) = idx;
		});
		(ret_dists, ret_idx)
	}
	pub fn query_range_approx<D2: Data<Elem=B>>(&self, queries: &ArrayBase<D2, Ix2>, max_dist: usize) -> (Vec<Vec<usize>>, Vec<Vec<usize>>) {
		let mut ret_dists = vec![vec![]; queries.shape()[0]];
		let mut ret_idx = vec![vec![]; queries.shape()[0]];
		par_iter(
			queries.axis_iter(Axis(0))
			.zip(ret_dists.iter_mut())
			.zip(ret_idx.iter_mut())
		)
		.map(|((a,b),c)| (a,b,c))
		.for_each(|(query, target_dist, target_idx)| {
			let (dists, idx) = self.query_range_approx_single(&query, max_dist);
			*target_dist = dists;
			*target_idx = idx;
		});
		(ret_dists, ret_idx)
	}


	#[allow(unused)]
	fn is_root(&self, node: usize) -> bool {
		node == 0
	}
	#[allow(unused)]
	fn is_leaf(&self, node: usize) -> bool {
		self.child_of(node, 0) >= self.n_nodes
	}
	#[allow(unused)]
	fn child_of(&self, node: usize, i_child: usize) -> usize {
		node*FANOUT+i_child+1
	}
	#[allow(unused)]
	fn parent_of(&self, node: usize) -> usize {
		(node-1)/FANOUT
	}
	#[allow(unused)]
	fn child_index_of(&self, node: usize) -> usize {
		(node-1) % FANOUT
	}
	#[allow(unused)]
	fn depth_of(&self, node: usize) -> usize {
		self.nodes[node].depth
	}

	pub fn get_n_nodes(&self) -> usize {
		self.n_nodes
	}
	pub fn get_height(&self) -> usize {
		self.height
	}
}

#[derive(Clone)]
struct THXNode<const FANOUT: usize> {
	first_bit: usize,
	is_leaf: bool,
	children_counts: Option<[usize; FANOUT]>,
	stored_objects: Option<Vec<usize>>,
	depth: usize
}
impl<const FANOUT: usize> THXNode<FANOUT> {
	pub fn new() -> THXNode<FANOUT> {
		Self {
			first_bit: 0,
			is_leaf: false,
			children_counts: None,
			stored_objects: None,
			depth: 0
		}
	}
}




#[test]
fn test_thx_init() {
	use ndarray_rand::rand::random;
	use ndarray::{Array2};
	type T = u16;
	const FANOUT: usize = 8;
	let arr2: Array2<T> = Array2::from_shape_simple_fn([20000,16], random) % T::MAX;
	let thx: THX<_,_,FANOUT> = THX::new(arr2, 128);
	let mut work: Vec<usize> = vec![0];
	while work.len() > 0 {
		let i_node = work.pop().unwrap();
		let node = &thx.nodes[i_node];
		if node.is_leaf {
			println!("{:}{:}{:}", "--".repeat(node.depth), if node.depth==0 {""} else {" "}, node.stored_objects.as_ref().unwrap().len());
		} else {
			println!("{:}{:}{:?}", "--".repeat(node.depth), if node.depth==0 {""} else {" "}, node.children_counts.as_ref().unwrap());
			(0..FANOUT).rev().for_each(|i_child| work.push(thx.child_of(i_node, i_child)));
		}
	}
	let mut i_node = 0;
	while !thx.nodes[i_node].is_leaf {
		i_node = thx.child_of(i_node, 0);
	}
	thx.nodes[i_node].stored_objects.as_ref().unwrap().iter().for_each(|i_row| {
		assert!(thx.data.row(*i_row).count_bits_range_unchecked(0, thx.n_bits_per_layer) < thx.quantile_bounds[0]);
	});
	println!("{:?}", thx.quantile_bounds);
	thx.nodes.iter().filter(|node| node.is_leaf).enumerate().for_each(|(i_leaf, leaf)| {
		let stored_indices = leaf.stored_objects.as_ref().unwrap();
		let sum_of_bits: usize = stored_indices.iter()
		.map(|&i_row| thx.data.row(i_row))
		.map(|row| row.count_bits())
		.sum();
		let average_of_bits: f64 = sum_of_bits as f64 / stored_indices.len() as f64;
		println!("Leaf {:}: {:.4}", i_leaf, average_of_bits);
	});
}

#[test]
fn test_thx_query() {
	use ndarray_rand::rand::random;
	use ndarray::{Array2};
	type T = u32;
	const FANOUT: usize = 4;
	let arr2: Array2<T> = Array2::from_shape_simple_fn([200000,32], random) % T::MAX;
	let thx: THX<_,_,FANOUT> = THX::new(arr2, 256);
	println!("Total nodes: {:>3}", thx.n_nodes);
	let q2: Array2<T> = Array2::from_shape_simple_fn([5,32], random) % T::MAX;
	let (_dists, _idx) = thx.query_approx(&q2, 10);
	// println!("{:?}", idx);
	// println!("{:?}", dists);
}