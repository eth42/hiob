#[cfg(feature="half")]
use half::f16;
use ndarray::{Array1,Array2};
use pyo3::types::PyDict;
use std::{pin::Pin, marker::PhantomData};
use futures::{prelude::*, executor::block_on};

use crate::data::{MatrixDataSource, AsyncMatrixDataSource};
use crate::types::CachingNumpyEquivalent;


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

macro_rules! with_h5py_dataset {
	($py: ident, $globals: expr, $locals: ident, $file: expr, $dataset: expr, $code: expr) => {{
		let $locals = PyDict::new($py);
		$locals.set_item("file", $py.eval(
			format!("h5py.File(\"{:}\",\"r\",rdcc_nbytes=0,rdcc_nslots=0,rdcc_w0=1)", $file).as_str(),
			Some($globals),
			Some($locals),
		)?)?;
		$locals.set_item("data", $py.eval(
			format!("file[\"{:}\"]", $dataset).as_str(),
			Some($globals),
			Some($locals),
		)?)?;
		let result = $code;
		$py.run("file.close()",Some($globals),Some($locals))?;
		$py.run("del data",Some($globals),Some($locals))?;
		$py.run("del file",Some($globals),Some($locals))?;
		$locals.clear();
		$py.run("gc.collect()",Some($globals),Some($locals))?;
		result
	}};
	(closure $py: ident, $locals: ident, $file: expr, $dataset: expr, $code: expr) => {{
		let result: Result<_,pyo3::PyErr> = pyo3::Python::with_gil(|$py| {
			let $locals = PyDict::new($py);
			$locals.set_item("h5py", $py.import("h5py")?)?;
			$locals.set_item("np", $py.import("numpy")?)?;
			$locals.set_item("gc", $py.import("gc")?)?;
			$locals.set_item("file", $py.eval(
				format!("h5py.File(\"{:}\",\"r\",rdcc_nbytes=0,rdcc_nslots=0,rdcc_w0=1)", $file).as_str(),
				None,
				Some($locals),
			)?)?;
			$locals.set_item("data", $py.eval(
				format!("file[\"{:}\"]", $dataset).as_str(),
				None,
				Some($locals),
			)?)?;
			let result = $code;
			$py.run("file.close()",None,Some($locals))?;
			$locals.del_item("data")?;
			$locals.del_item("file")?;
			$locals.keys().iter().for_each(|v|{let _=$locals.del_item(v);});
			$locals.clear();
			result
		});
		result.unwrap()
	}};
}
pub struct H5PyDataset<T: NumpyEquivalent+num::Zero> {
	_phantom: PhantomData<T>,
	file: String,
	dataset: String,
	n_rows: usize,
	n_cols: usize
}
impl<T: NumpyEquivalent+num::Zero> H5PyDataset<T> {
	pub fn new(file: &str, dataset: &str) -> Self {
		let result: Result<_,pyo3::PyErr> = pyo3::Python::with_gil(|py| {
			let globals = Self::make_globals(py)?;
			/* Get dataset shape */
			Ok(with_h5py_dataset!(py,globals,locals,file,dataset,{
				py.eval(
					"data.shape",
					Some(globals),
					Some(locals),
				)?.extract()?
			}))
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
	fn make_globals<'py>(py: pyo3::Python<'py>) -> Result<&'py PyDict,pyo3::PyErr> {
		let globals: &'py PyDict = PyDict::new(py);
		globals.set_item("h5py", py.import("h5py")?)?;
		globals.set_item("np", py.import("numpy")?)?;
		globals.set_item("gc", py.import("gc")?)?;
		Ok(globals)
	}
	fn _static_get_row(i_row: usize, n_cols: usize, file: String, dataset: String) -> Array1<T> {
		let buffer = Array1::from_elem((n_cols,),T::zero());
		let _ = pyo3::Python::with_gil(|py| {
			let globals = Self::make_globals(py)?;
			Ok::<(),pyo3::PyErr>(with_h5py_dataset!(py,globals,locals,file.as_str(),dataset.as_str(),{
				/* An as-empty-as-possible reference to stick the PyArray's lifetime to */
				let array_container = pyo3::types::PyList::empty(py);
				let array_ref = unsafe{numpy::PyArray1::<T>::borrow_from_array(&buffer, array_container)};
				locals.set_item("i", i_row)?;
				locals.set_item("rows", array_ref)?;
				py.run("rows[:] = data[i]",Some(globals),Some(locals))?;
				locals.del_item("rows")?;
			}))
		}).unwrap();
		buffer
	}
	fn _static_get_rows(i_rows: Vec<usize>, n_cols: usize, file: String, dataset: String) -> Array2<T> {
		let buffer = Array2::from_elem((i_rows.len(), n_cols),T::zero());
		let _ = pyo3::Python::with_gil(|py| {
			let globals = Self::make_globals(py)?;
			Ok::<(),pyo3::PyErr>(with_h5py_dataset!(py,globals,locals,file.as_str(),dataset.as_str(),{
				/* An as-empty-as-possible reference to stick the PyArray's lifetime to */
				let array_container = pyo3::types::PyList::empty(py);
				let array_ref = unsafe{numpy::PyArray2::<T>::borrow_from_array(&buffer, array_container)};
				locals.set_item("idx", i_rows)?;
				locals.set_item("rows", array_ref)?;
				py.run("rows[:] = data[np.sort(idx)]",Some(globals),Some(locals))?;
				locals.del_item("rows")?;
			}))
		}).unwrap();
		buffer
	}
	fn _static_get_rows_slice(i_row_from: usize, i_row_to: usize, n_cols: usize, file: String, dataset: String) -> Array2<T> {
		let buffer = Array2::from_elem((i_row_to-i_row_from, n_cols),T::zero());
		let _ = pyo3::Python::with_gil(|py| {
			let globals = Self::make_globals(py)?;
			Ok::<(),pyo3::PyErr>(with_h5py_dataset!(py,globals,locals,file.as_str(),dataset.as_str(),{
				/* An as-empty-as-possible reference to stick the PyArray's lifetime to */
				let array_container = pyo3::types::PyList::empty(py);
				let array_ref = unsafe{numpy::PyArray2::<T>::borrow_from_array(&buffer, array_container)};
				locals.set_item("start", i_row_from)?;
				locals.set_item("end", i_row_to)?;
				locals.set_item("rows", array_ref)?;
				py.run("rows[:] = data[start:end]",Some(globals),Some(locals))?;
				locals.del_item("rows")?;
			}))
		}).unwrap();
		buffer
	}
	async fn get_row_async(&self, i_row: usize) -> Array1<T> {
		H5PyDataset::_static_get_row(i_row, self.n_cols(), self.file.clone(), self.dataset.clone())
	}
	async fn get_rows_async(&self, i_rows: Vec<usize>) -> Array2<T> {
		H5PyDataset::_static_get_rows(i_rows, self.n_cols(), self.file.clone(), self.dataset.clone())
	}
	async fn get_rows_slice_async(&self, i_row_from: usize, i_row_to: usize) -> Array2<T> {
		H5PyDataset::_static_get_rows_slice(i_row_from, i_row_to, self.n_cols(), self.file.clone(), self.dataset.clone())
	}
}
impl<T: NumpyEquivalent+num::Zero> MatrixDataSource<T> for H5PyDataset<T> {
	fn n_rows(&self) -> usize { self.n_rows }
	fn n_cols(&self) -> usize { self.n_cols }
	fn get_row(&self, i_row: usize) -> Array1<T> {
		H5PyDataset::_static_get_row(i_row, self.n_cols(), self.file.clone(), self.dataset.clone())
	}
	fn get_rows(&self, i_rows: &Vec<usize>) -> Array2<T> {
		H5PyDataset::_static_get_rows(i_rows.clone(), self.n_cols(), self.file.clone(), self.dataset.clone())
	}
	fn get_rows_slice(&self, i_row_from: usize, i_row_to: usize) -> Array2<T> {
		H5PyDataset::_static_get_rows_slice(i_row_from, i_row_to, self.n_cols(), self.file.clone(), self.dataset.clone())
	}
}

pub struct CachingH5PyReader<T: CachingNumpyEquivalent> {
	_phantom: PhantomData<T>,
	has_active_query: bool,
	query_is_range: bool,
	file_name: String,
	dataset_name: String,
	dataset: H5PyDataset<T>,
	cache_future: Option<Pin<Box<dyn Future<Output=Array2<T>>>>>,
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
			// self.cache_future = Some(Box::pin(Self::load_rows(self.file_name.clone(), self.dataset_name.clone(), idx)));
			// self.cache_future = Some(Box::pin(unsafe {std::mem::transmute::<&Self, &'static Self>(self)}.load_rows(idx)));
			let n_cols = self.n_cols();
			self.cache_future = Some(Box::pin((async |file:String,dataset:String,mut idx:Vec<usize>,n_cols:usize| -> Array2<T>{
				idx.sort();
				let idx = Array1::from_vec(idx);
				let buffer = Array2::from_elem((idx.len(), n_cols),T::zero());
				with_h5py_dataset!(closure py,locals,file,dataset,{
					/* An as-empty-as-possible reference to stick the PyArray lifetime */
					let array_container = pyo3::types::PyList::empty(py);
					let idx_ref = unsafe{numpy::PyArray1::<usize>::borrow_from_array(&idx, array_container)};
					let array_ref = unsafe{numpy::PyArray2::<T>::borrow_from_array(&buffer, array_container)};
					locals.set_item("idx", idx_ref)?;
					locals.set_item("rows", array_ref)?;
					py.run("data.read_direct(rows,np.s_[idx],np.s_[:])",None,Some(locals))?;
					// py.run("rows[:] = data[np.sort(idx)]",None,Some(locals))?;
					locals.del_item("rows")?;
					locals.del_item("idx")?;
					Ok(())
				});
				buffer
			})(self.file_name.clone(),self.dataset_name.clone(),idx,n_cols)));
			Ok(())
		}
	}
	fn prepare_rows_slice(&mut self, start: usize, end: usize) -> Result<(), ()> {
		if self.has_active_query {
			Err(())
		} else {
			self.query_is_range = true;
			self.has_active_query = true;
			// self.cache_future = Some(Box::pin(Self::load_rows_slice(self.file_name.clone(), self.dataset_name.clone(), start, end)));
			// self.cache_future = Some(Box::pin(unsafe {std::mem::transmute::<&Self, &'static Self>(self)}.load_rows_slice(start, end)));
			let n_cols = self.n_cols();
			self.cache_future = Some(Box::pin((async |file:String,dataset:String,start:usize,end:usize,n_cols:usize| -> Array2<T>{
				let batch_size = end - start;
				let buffer = Array2::from_elem((batch_size, n_cols),T::zero());
				with_h5py_dataset!(closure py,locals,file,dataset,{
					/* An as-empty-as-possible reference to stick the PyArray lifetime */
					let array_container = pyo3::types::PyList::empty(py);
					let array_ref = unsafe{numpy::PyArray2::<T>::borrow_from_array(&buffer, array_container)};
					locals.set_item("start", start)?;
					locals.set_item("end", end)?;
					locals.set_item("rows", array_ref)?;
					py.run("data.read_direct(rows,np.s_[start:end],np.s_[:])",None,Some(locals))?;
					// py.run("rows[:] = data[start:end]",None,Some(locals))?;
					locals.del_item("rows")?;
					Ok(())
				});
				buffer
			})(self.file_name.clone(),self.dataset_name.clone(),start,end,n_cols)));
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