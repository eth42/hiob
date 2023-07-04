#![allow(dead_code)]


pub mod bit_vectors;
pub mod float_vectors;
// mod measures;
pub mod binarizer;
pub mod bits;
pub mod progress;
pub mod eval;
pub mod index;
pub mod heaps;
pub mod random;
pub mod data;
#[cfg(feature="python")]
pub mod pydata;

use {
	ndarray::{Axis, Array, Array2},
	ndarray_rand::RandomExt,
	ndarray_rand::rand_distr::StandardNormal,
};


#[allow(unused)]
fn main() {
	#[cfg(feature="python")]
	manual_benchmark();
}


#[allow(dead_code)]
#[cfg(feature="python")]
fn manual_benchmark() {
  use std::ops::{DivAssign};
	use pydata::H5PyDataset;
	use crate::data::MatrixDataSource;
	use crate::binarizer::StochasticHIOB;
	let data_file = "pytest/sisap23challenge/data/clip768v2/300K/dataset.h5";
	let data_set = "emb";
	let data_loader: H5PyDataset<f32> = H5PyDataset::new(data_file, data_set);
	println!("Loading data");
	let data = data_loader.get_rows_slice(0, data_loader.n_rows());
	println!("Initializing random centers");
	let mut init_centers: Array2<f32> = Array::random((1024,data_loader.n_cols()), StandardNormal{});
	init_centers.axis_iter_mut(Axis(0))
	.for_each(|mut row| {
		let norm = row.iter().map(|&v| v*v).reduce(|a,b| a+b).unwrap().sqrt();
		row.div_assign(norm);
	});
	println!("Building HIOB");
	let mut hiob: StochasticHIOB<f32, u64, Array2<f32>> = StochasticHIOB::new(
		data.clone(),
		10_000,
		128,
		1024,
		None,
		Some(0.1),
		Some(init_centers),
		Some(false),
		Some(false),
		None,
		None,
		None,
	);
	println!("Training HIOB");
	hiob.run(10_000);
	// println!("Binarizing");
	// hiob.binarize(&data);
}
