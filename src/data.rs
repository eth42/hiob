#[cfg(feature="rust-hdf5")]
use {
	hdf5::H5Type,
	crate::progress::{par_iter, MaybeSend},
	num::Zero,
	ndarray::s,
	futures::{prelude::*, executor::block_on},
	std::{pin::Pin, marker::PhantomData},
};
#[cfg(feature="rust-hdf5")]
#[cfg(feature="parallel")]
use rayon::iter::ParallelIterator;
use ndarray::{Array1, Array2, Slice, Data, ArrayBase, Ix2, Axis};

pub trait MatrixDataSource<T> {
	fn n_rows(&self) -> usize;
	fn n_cols(&self) -> usize;
	fn get_row(&self, i_row: usize) -> Array1<T>;
	fn get_rows(&self, i_rows: &Vec<usize>) -> Array2<T>;
	fn get_rows_slice(&self, i_row_from: usize, i_row_to: usize) -> Array2<T>;
}
pub trait AsyncMatrixDataSource<T>: MatrixDataSource<T> {
	fn prepare_rows(&mut self, i_rows: Vec<usize>) -> Result<(), ()>;
	fn prepare_rows_slice(&mut self, i_row_from: usize, i_row_to: usize) -> Result<(), ()>;
	fn get_cached(&mut self) -> Option<Array2<T>>;
}

impl<T: Copy+Clone, D: Data<Elem=T>> MatrixDataSource<T> for ArrayBase<D, Ix2> {
	fn n_rows(&self) -> usize {
		self.shape()[0]
	}
	fn n_cols(&self) -> usize {
		self.shape()[1]
	}
	fn get_row(&self, i_row: usize) -> Array1<T> {
		self.row(i_row).into_owned()
	}
	fn get_rows(&self, i_rows: &Vec<usize>) -> Array2<T> {
		Array2::from_shape_fn(
			(i_rows.len(), self.n_cols()),
			|(i,j)| self[[i_rows[i], j]]
		)
	}
	fn get_rows_slice(&self, i_row_from: usize, i_row_to: usize) -> Array2<T> {
		self.slice_axis(Axis(0), Slice::from(i_row_from..i_row_to.min(self.n_rows()))).to_owned()
	}
}

#[cfg(feature="rust-hdf5")]
impl<T: hdf5::H5Type+Copy+Zero+MaybeSend> MatrixDataSource<T> for hdf5::Dataset {
	fn n_rows(&self) -> usize {
		self.shape()[0]
	}

	fn n_cols(&self) -> usize {
		self.shape()[1]
	}

	fn get_row(&self, i_row: usize) -> Array1<T> {
		self.read_slice_1d(s![i_row, ..]).unwrap()
	}

	fn get_rows(&self, i_rows: Vec<usize>) -> Array2<T> {
		let n_rows = i_rows.len();
		let n_cols = <hdf5::Dataset as MatrixDataSource<T>>::n_cols(&self);
		let mut ret: Array2<T> = Array2::from_elem([n_rows, n_cols], T::zero());
		par_iter(
			ret.axis_iter_mut(Axis(0))
			.zip(i_rows.into_iter())
		)
		.for_each(|(mut target, i_row)| {
			target.assign(&self.get_row(i_row));
		});
		ret
		// let ret_vec: Vec<Array1<T>> = i_rows.into_iter()
		// .map(|i_row| self.get_row(i_row))
		// .collect();
		// let shape: (usize, usize) = (n_rows, <hdf5::Dataset as MatrixDataSource<T>>::n_cols(&self));
		// Array2::from_shape_fn(
		// 	shape,
		// 	|(i,j)| unsafe { *ret_vec.get_unchecked(i).uget(j) }
		// )
	}

	fn get_rows_slice(&self, i_row_from: usize, i_row_to: usize) -> Array2<T> {
		let n_rows = <hdf5::Dataset as MatrixDataSource<T>>::n_rows(&self);
		self.read_slice_2d(s![i_row_from..i_row_to.min(n_rows),..]).unwrap()
	}
}





#[cfg(feature="rust-hdf5")]
pub trait CachingH5Type: H5Type+Zero+Copy+Clone+Send+'static {}
#[cfg(feature="rust-hdf5")]
impl<T: H5Type+Zero+Copy+Clone+Send+'static> CachingH5Type for T {}

#[cfg(feature="rust-hdf5")]
pub struct CachingH5Reader<T: CachingH5Type> {
	_phantom: PhantomData<T>,
	has_active_query: bool,
	query_is_range: bool,
	file_name: String,
	dataset_name: String,
	dataset: hdf5::Dataset,
	cache_future: Option<Pin<Box<dyn Future<Output=Array2<T>>>>>
}
#[cfg(feature="rust-hdf5")]
impl<T: CachingH5Type> CachingH5Reader<T> {
	#[allow(unused)]
	pub fn new(file_name: String, dataset_name: String) -> Self {
		let dataset = read_h5_dataset(file_name.as_str(), dataset_name.as_str());
		Self {
			_phantom: PhantomData,
			has_active_query: false,
			query_is_range: false,
			file_name: file_name,
			dataset_name: dataset_name,
			dataset: dataset.unwrap(),
			cache_future: None
		}
	}
	async fn load_rows(file_name: String, dataset_name: String, idx: Vec<usize>) -> Array2<T> {
		let data = read_h5_dataset(file_name.as_str(), dataset_name.as_str());
		data.unwrap().get_rows(idx)
	}
	async fn load_rows_slice(file_name: String, dataset_name: String, start: usize, end: usize) -> Array2<T> {
		let data = read_h5_dataset(file_name.as_str(), dataset_name.as_str());
		data.unwrap().get_rows_slice(start, end)
	}
}
#[cfg(feature="rust-hdf5")]
impl<T: CachingH5Type> MatrixDataSource<T> for CachingH5Reader<T> {
	fn n_rows(&self) -> usize {
		<hdf5::Dataset as MatrixDataSource<T>>::n_rows(&self.dataset)
	}
	fn n_cols(&self) -> usize {
		<hdf5::Dataset as MatrixDataSource<T>>::n_cols(&self.dataset)
	}
	fn get_row(&self, i_row: usize) -> Array1<T> {
		self.dataset.get_row(i_row)
	}
	fn get_rows(&self, i_rows: Vec<usize>) -> Array2<T> {
		self.dataset.get_rows(i_rows)
	}
	fn get_rows_slice(&self, i_row_from: usize, i_row_to: usize) -> Array2<T> {
		self.dataset.get_rows_slice(i_row_from, i_row_to)
	}
}
#[cfg(feature="rust-hdf5")]
impl<T: CachingH5Type> AsyncMatrixDataSource<T> for CachingH5Reader<T> {
	fn prepare_rows(&mut self, idx: Vec<usize>) -> Result<(), ()> {
		if self.has_active_query {
			Err(())
		} else {
			self.query_is_range = false;
			self.has_active_query = true;
			self.cache_future = Some(Box::pin(Self::load_rows(self.file_name.clone(), self.dataset_name.clone(), idx)));
			Ok(())
		}
	}
	fn prepare_rows_slice(&mut self, start: usize, end: usize) -> Result<(), ()> {
		if self.has_active_query {
			Err(())
		} else {
			self.query_is_range = true;
			self.has_active_query = true;
			self.cache_future = Some(Box::pin(Self::load_rows_slice(self.file_name.clone(), self.dataset_name.clone(), start, end)));
			Ok(())
		}
	}
	fn get_cached(&mut self) -> Option<Array2<T>> {
		if self.cache_future.is_none() || !self.has_active_query {
			None
		} else {
			let future_arr = unsafe{self.cache_future.take().unwrap_unchecked()};
			let arr = block_on(future_arr);
			self.has_active_query = false;
			Some(arr)
		}
	}
}


#[cfg(feature="rust-hdf5")]
pub fn read_h5_dataset(file: &str, dataset: &str) -> Result<hdf5::Dataset, hdf5::Error> {
	let file = hdf5::File::open(file)?;
	file.dataset(dataset)
}
#[cfg(feature="rust-hdf5")]
#[allow(unused)]
pub fn store_h5_dataset<T: hdf5::H5Type>(file: &str, dataset: &str, data: &Array2<T>) -> Result<(), hdf5::Error>{
	let file = hdf5::File::create(file)?;
	let dataset_builder = file.new_dataset_builder();
	dataset_builder.with_data(data).create(dataset)?;
	Ok(())
}


#[test]
#[cfg(feature="rust-hdf5")]
fn benchmark_access_dataset_time() {
	use std::time::{SystemTime, UNIX_EPOCH};
	let file = "/home/thordsen/tmp/sisap23challenge/data/laion2B-en-clip768v2-n=100K.h5";
	let current_millis = || SystemTime::now().duration_since(UNIX_EPOCH).unwrap().as_millis();
	let n_its = 100;
	let start = current_millis();
	(0..n_its).for_each(|_| {
		_ = read_h5_dataset(file, "emb");
	});
	let end = current_millis();
	println!("{:?}", (end-start)/n_its);
	let n_its = 5;
	let start = current_millis();
	(0..n_its).for_each(|_| {
		let data = read_h5_dataset(file, "emb").unwrap();
		_ = <hdf5::Dataset as MatrixDataSource<f32>>::get_rows_slice(&data, 0, 100_000);
	});
	let end = current_millis();
	println!("{:?}", (end-start)/n_its);
}

