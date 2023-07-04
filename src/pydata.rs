#[cfg(feature="half")]
use half::f16;
use ndarray::{Array1,Array2};
use std::{pin::Pin, marker::PhantomData};
use futures::{prelude::*, executor::block_on};

use crate::data::{MatrixDataSource, AsyncMatrixDataSource};


pub trait NumpyEquivalent: numpy::Element {
	fn numpy_name() -> &'static str;
}
macro_rules! make_numpy_equivalent {
	($(($rust_types: ty, $numpy_names: literal)),*) => {
		$(make_numpy_equivalent!($rust_types, $numpy_names);)*
	};
	($rust_type: ty, $numpy_name: literal) => {
		impl NumpyEquivalent for $rust_type {
			fn numpy_name() -> &'static str {
				$numpy_name
			}
		}
	};
}
make_numpy_equivalent!(
	(f32, "float32"), (f64, "float64"),
	(bool, "bool_"),
	(u8, "uint8"), (u16, "uint16"),	(u32, "uint32"), (u64, "uint64")
);
#[cfg(feature="half")]
make_numpy_equivalent!((f16, "float16"));

pub struct H5PyDataset<T: NumpyEquivalent> {
	_phantom: PhantomData<T>,
	file: String,
	dataset: String,
	n_rows: usize,
	n_cols: usize
}
impl<T: NumpyEquivalent> H5PyDataset<T> {
	pub fn new(file: &str, dataset: &str) -> Self {
		let result: Result<_,pyo3::PyErr> = pyo3::Python::with_gil(|py| {
			let locals = pyo3::types::PyDict::new(py);
			locals.set_item("h5py", py.import("h5py")?)?;
			locals.set_item("data", py.eval(
				format!("h5py.File(\"{:}\")[\"{:}\"]", file, dataset).as_str(),
				None,
				Some(&locals)
			)?)?;
			let (n_rows, n_cols): (usize, usize) = py.eval(
				"data.shape",
				None,
				Some(&locals)
			)?.extract()?;
			Ok((n_rows, n_cols))
		});
		let (n_rows, n_cols) = result.unwrap();
		Self{
			_phantom: PhantomData,
			file: file.to_string(),
			dataset: dataset.to_string(),
			n_rows: n_rows,
			n_cols: n_cols
		}
	}
}
impl<T: NumpyEquivalent> MatrixDataSource<T> for H5PyDataset<T> {
	fn n_rows(&self) -> usize { self.n_rows }
	fn n_cols(&self) -> usize { self.n_cols }
	fn get_row(&self, i_row: usize) -> Array1<T> {
		let row: Result<_,pyo3::PyErr> = pyo3::Python::with_gil(|py| {
			let locals = pyo3::types::PyDict::new(py);
			locals.set_item("h5py", py.import("h5py")?)?;
			locals.set_item("np", py.import("numpy")?)?;
			locals.set_item("data", py.eval(
				format!("h5py.File(\"{:}\")[\"{:}\"]", self.file.as_str(), self.dataset.as_str()).as_str(),
				None,
				Some(&locals)
			)?)?;
			locals.set_item("i", i_row)?;
			let row_obj = py.eval(
				format!("data[i].astype(np.{:})", T::numpy_name()).as_str(),
				None,
				Some(&locals)
			)?;
			let row: &numpy::PyArray1<T> = row_obj.downcast()?;
			Ok(row.to_owned_array())
		});
		row.unwrap()
	}
	fn get_rows(&self, i_rows: &Vec<usize>) -> Array2<T> {
		let row: Result<_,pyo3::PyErr> = pyo3::Python::with_gil(|py| {
			let locals = pyo3::types::PyDict::new(py);
			locals.set_item("h5py", py.import("h5py")?)?;
			locals.set_item("np", py.import("numpy")?)?;
			locals.set_item("data", py.eval(
				format!("h5py.File(\"{:}\")[\"{:}\"]", self.file.as_str(), self.dataset.as_str()).as_str(),
				None,
				Some(&locals)
			)?)?;
			locals.set_item("idx", i_rows)?;
			let row_obj = py.eval(
				format!("data[np.sort(idx)].astype(np.{:})", T::numpy_name()).as_str(),
				None,
				Some(&locals)
			)?;
			let row: &numpy::PyArray2<T> = row_obj.downcast()?;
			Ok(row.to_owned_array())
		});
		row.unwrap()
	}
	fn get_rows_slice(&self, i_row_from: usize, i_row_to: usize) -> Array2<T> {
		let row: Result<_,pyo3::PyErr> = pyo3::Python::with_gil(|py| {
			let locals = pyo3::types::PyDict::new(py);
			locals.set_item("h5py", py.import("h5py")?)?;
			locals.set_item("np", py.import("numpy")?)?;
			locals.set_item("data", py.eval(
				format!("h5py.File(\"{:}\")[\"{:}\"]", self.file.as_str(), self.dataset.as_str()).as_str(),
				None,
				Some(&locals)
			)?)?;
			locals.set_item("start", i_row_from)?;
			locals.set_item("end", i_row_to)?;
			let row_obj = py.eval(
				format!("data[start:end].astype(np.{:})", T::numpy_name()).as_str(),
				None,
				Some(&locals)
			)?;
			let row: &numpy::PyArray2<T> = row_obj.downcast()?;
			Ok(row.to_owned_array())
		});
		row.unwrap()
	}
}

pub trait CachingNumpyEquivalent: NumpyEquivalent+'static {}
impl<T: NumpyEquivalent+'static> CachingNumpyEquivalent for T {}

pub struct CachingH5PyReader<T: CachingNumpyEquivalent> {
	_phantom: PhantomData<T>,
	has_active_query: bool,
	query_is_range: bool,
	file_name: String,
	dataset_name: String,
	dataset: H5PyDataset<T>,
	cache_future: Option<Pin<Box<dyn Future<Output=Array2<T>>>>>
}
impl<T: CachingNumpyEquivalent> CachingH5PyReader<T> {
	pub fn new(file_name: String, dataset_name: String) -> Self {
		let dataset = H5PyDataset::<T>::new(file_name.as_str(), dataset_name.as_str());
		Self {
			_phantom: PhantomData,
			has_active_query: false,
			query_is_range: false,
			file_name: file_name,
			dataset_name: dataset_name,
			dataset: dataset,
			cache_future: None
		}
	}
	async fn load_rows(file_name: String, dataset_name: String, idx: Vec<usize>) -> Array2<T> {
		let data = H5PyDataset::<T>::new(file_name.as_str(), dataset_name.as_str());
		data.get_rows(&idx)
	}
	async fn load_rows_slice(file_name: String, dataset_name: String, start: usize, end: usize) -> Array2<T> {
		let data = H5PyDataset::<T>::new(file_name.as_str(), dataset_name.as_str());
		data.get_rows_slice(start, end)
	}
}
impl<T: CachingNumpyEquivalent> MatrixDataSource<T> for CachingH5PyReader<T> {
	fn n_rows(&self) -> usize {
		<H5PyDataset<T> as MatrixDataSource<T>>::n_rows(&self.dataset)
	}
	fn n_cols(&self) -> usize {
		<H5PyDataset<T> as MatrixDataSource<T>>::n_cols(&self.dataset)
	}
	fn get_row(&self, i_row: usize) -> Array1<T> {
		self.dataset.get_row(i_row)
	}
	fn get_rows(&self, i_rows: &Vec<usize>) -> Array2<T> {
		self.dataset.get_rows(i_rows)
	}
	fn get_rows_slice(&self, i_row_from: usize, i_row_to: usize) -> Array2<T> {
		self.dataset.get_rows_slice(i_row_from, i_row_to)
	}
}
impl<T: CachingNumpyEquivalent> AsyncMatrixDataSource<T> for CachingH5PyReader<T> {
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



#[test]
fn test_h5py_binding() {
	use std::time::{SystemTime, UNIX_EPOCH};
	let current_millis = || SystemTime::now().duration_since(UNIX_EPOCH).unwrap().as_millis();
	let file = "/home/thordsen/tmp/sisap23challenge/data/laion2B-en-clip768v2-n=100K.h5";
	let dataset = "emb";
	let n_its = 5;
	let start = current_millis();
	(0..n_its).for_each(|_| {
		let data: H5PyDataset<f32> = H5PyDataset::new(file, dataset);
		_ = data.get_rows_slice(0, 100_000);
	});
	let end = current_millis();
	println!("{:?}", (end-start)/n_its);
}