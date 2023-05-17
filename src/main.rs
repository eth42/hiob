mod bit_vectors;
mod measures;
mod binarizer;
mod bits;
mod progress;
mod eval;
mod pybridge;
mod index;
mod heaps;
mod random;
mod data;

use std::io::{self, Write};

use bits::Bits;
use eval::BinarizationEvaluator;
use hdf5::H5Type;
use ndarray::{Axis, Array, Array2};
use ndarray_rand::RandomExt;
use ndarray_rand::rand_distr::StandardNormal;
use rand::prelude::*;

use crate::measures::{InnerProduct,DotProduct};
use crate::binarizer::{HIOB, HIOBFloat, HIOBBits};
use crate::bit_vectors::{BitVector};
// use crate::progress::{named_range};

fn _print_hiob_oberlaps<F: HIOBFloat, B: HIOBBits>(hiob: &HIOB<F,B>) {
	hiob.get_overlap_mat().axis_iter(Axis{0: 0}).for_each(|row| {
		row.iter().for_each(|v| {
			print!("{0:>3} ", v);
		});
		println!("");
	});
}
fn _print_hiob_sim_sums<F: HIOBFloat, B: HIOBBits>(hiob: &HIOB<F,B>) {
	hiob.get_sim_sums().iter().for_each(|v| print!("{0:>8.5} ", v-0.5));
	println!("");
}

fn _read_h5_file<F: H5Type>(file: &str, dataset_name: &str) -> Result<Array2<F>, hdf5::Error> {
	print!("Opening file... "); _=io::stdout().flush();
	let file_handle = hdf5::File::open(file)?;
	println!("done");
	print!("Accessing dataset... "); _=io::stdout().flush();
	let dataset = file_handle.dataset(dataset_name)?;
	println!("done");
	print!("Reading into Array2... "); _=io::stdout().flush();
	let data = dataset.read_2d::<F>()?;
	println!("done");
	Ok(data)
}
fn _write_h5_file<F: H5Type>(file: &str, dataset_name: &str, arr: &Array2<F>) -> Result<(), hdf5::Error> {
	let file_handle = hdf5::File::create(file)?;
	file_handle.new_dataset_builder().with_data(arr).create(dataset_name)?;
	Ok(())
}

#[allow(dead_code)]
fn load_laion_data<F: H5Type>(size: &str) -> Result<Array2<F>, hdf5::Error> {
	_read_h5_file(
		format!("/home/thordsen/tmp/sisap23challenge/data/laion2B-en-clip768v2-n={:}.h5",size).as_str(),
		"emb"
	)
}
#[allow(dead_code)]
fn load_laion_neighbors(size: &str) -> Result<Array2<usize>, hdf5::Error> {
	let arr = _read_h5_file(
		format!("/home/thordsen/tmp/sisap23challenge/nns/laion2B-en-public-gold-standard-v2-{:}.h5",size).as_str(),
		"knns"
	)?;
	Ok(arr - 1)
}
#[allow(dead_code)]
fn load_laion_queries<F: H5Type>() -> Result<Array2<F>, hdf5::Error> {
	_read_h5_file(
		"/home/thordsen/tmp/sisap23challenge/data/public-queries-10k-clip768v2.h5",
		"emb"
	)
}
#[allow(dead_code)]
fn safe_laion_data_bins<B: Bits+H5Type>(data_bin: &Array2<B>, size: &str) -> Result<(), hdf5::Error> {
	_write_h5_file(
		format!("/home/thordsen/tmp/sisap23challenge/bins/hiob_d_n={:}.h5", size).as_str(),
		"hamming",
		data_bin
	)
}
#[allow(dead_code)]
fn safe_laion_queries_bins<B: Bits+H5Type>(queries_bin: &Array2<B>, size: &str) -> Result<(), hdf5::Error> {
	_write_h5_file(
		format!("/home/thordsen/tmp/sisap23challenge/bins/hiob_q_n={:}.h5", size).as_str(),
		"hamming",
		queries_bin
	)
}

#[allow(dead_code)]
fn main() {
	if false { _test_bit_vectors(); }
	if false { _test_random(); }
	if true {
		let size = "100K";
		/* Load H5 data from disk */
		let data = load_laion_data::<f32>(size).unwrap();
		let true_neighbors = load_laion_neighbors(size).unwrap();
		let queries = load_laion_queries::<f32>().unwrap();
		/* Create HIOB and train it */
		let hiob: HIOB<f32, u64> = _test_gen_hiob(&data, 1024, 60000);
		/* Binarizes data and queries and store on disk */
		let data_bin = hiob.binarize(&data);
		safe_laion_data_bins(&data_bin, size).unwrap();
		let queries_bin = hiob.binarize(&queries);
		safe_laion_queries_bins(&queries_bin, size).unwrap();
		/* Evaluate the binarization */
		let eval = BinarizationEvaluator::new();
		let k = 10;
		let print_k_at_n = |n: usize| println!(
			"{:}@{:}-recall: {:>6.2}%", k, n,
			100.0 * eval.k_at_n_recall_prec_dot_neighbors(&data_bin, &queries_bin, &true_neighbors, n)
		);
		(0..20).map(|i| (i+1)*10).for_each(|i| print_k_at_n(i));
	}
}

fn _test_bit_vectors() {
	let mut rng = rand::thread_rng();
	let rand_vec1 = (0..5).map(|_| rng.next_u64()).collect::<Vec<u64>>();
	let rand_vec2 = (0..5).map(|_| rng.next_u64()).collect::<Vec<u64>>();
	rand_vec1.iter_bits().for_each(|b| print!("{}", if b {"1"} else {"0"}));
	println!("");
	rand_vec2.iter_bits().for_each(|b| print!("{}", if b {"1"} else {"0"}));
	println!("");
	println!("{}", rand_vec1.hamming_dist(&rand_vec2));
	println!("{}", rand_vec1.hamming_dist_same(&rand_vec2));
	let lvec: Vec<u64> = (0..3).map(|_| rng.next_u64()).collect::<Vec<u64>>();
	lvec.iter_bits().for_each(|b| print!("{}", if b {"1"} else {"0"}));
	println!("");
	lvec.iter().for_each(|v| print!("{}", format!("{:064b}", v).chars().rev().collect::<String>()));
	println!("");
}

fn _test_random() {
	let mut data: Array2<f32> = Array::random((100000,150), StandardNormal{});
	let prod = DotProduct::new();
	let norms = (0..data.shape()[0])
	.map(|i| data.row(i))
	.map(|row| prod.prod(&row,&row).sqrt())
	.collect::<Vec<f32>>();
	(0..data.shape()[0]).for_each(|i| data.row_mut(i).mapv_inplace(|v| v/norms[i]));
	let mut hiob: HIOB<f32, u64> = HIOB::new(data.clone(), 100, None, None, None, None, None, None);

	// _print_hiob_oberlaps(&hiob);
	_print_hiob_sim_sums(&hiob);
	hiob.run(2000);
	_print_hiob_sim_sums(&hiob);
	// _print_hiob_oberlaps(&hiob);
	
	// let hiob_bins = hiob.get_data_bins();
	// let data_bins = hiob.binarize(&data);
	// named_range(data.shape()[0], "Verifying binarization")
	// .map(|i| (data_bins.row(i), hiob_bins.row(i)))
	// .for_each(|(u,v)| {
	// 	assert_eq!(0, u.hamming_dist(&v));
	// 	assert_eq!(0, v.hamming_dist(&u));
	// 	assert_eq!(0, u.hamming_dist_same(&v));
	// 	assert_eq!(0, v.hamming_dist_same(&u));
	// });
}

fn _test_h5_file<F: HIOBFloat+H5Type, B: HIOBBits>(data_file: &str, data_set: &str, n_bits: usize, n_steps: usize) -> HIOB<F, B> {
	let data = _read_h5_file(data_file, data_set).unwrap();
	_test_gen_hiob(&data, n_bits, n_steps)
}

fn _test_gen_hiob<F: HIOBFloat, B: HIOBBits>(data: &Array2<F>, n_bits: usize, n_steps: usize) -> HIOB<F, B> {
	print!("Initializing HIOB... ");
	#[cfg(feature = "progressbars")]
	println!("");
	_=io::stdout().flush();
	let mut hiob: HIOB<F, B> = HIOB::new(data.to_owned(), n_bits, None, None, None, None, None, None);
	println!("done");
	
	_print_hiob_sim_sums(&hiob);
	hiob.run(n_steps);
	_print_hiob_sim_sums(&hiob);

	hiob
}
