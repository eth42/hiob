use pyo3::prelude::*;
use numpy::{PyArray1,PyArray2,PyReadonlyArray1,PyReadonlyArray2,ToPyArray};
use paste::paste;
use ndarray::{Array2,OwnedRepr};
#[cfg(feature="parallel")]
use {rayon::ThreadPoolBuilder, pyo3::exceptions::PyValueError};
#[cfg(not(feature="parallel"))]
use pyo3::exceptions::{PyValueError, PyWarning};
	
use crate::binarizer::{HIOB,StochasticHIOB};
use crate::data::{H5PyDataset};
use crate::eval::BinarizationEvaluator;
use crate::bit_vectors::BitVector;
use crate::index::THX;


macro_rules! hiob_struct_gen {
	(($($pts:ty),*), $bts:tt) => {
		$(hiob_struct_gen!($pts, $bts);)*
	};
	($prec_type: ty, ($($bts:ty),*)) => {
		$(hiob_struct_gen!($prec_type, $bts);)*
	};
	($prec_type: ty, $bin_type: ty) => {
		paste! {
			#[allow(non_camel_case_types)]
			#[pyclass]
			pub struct [<HIOB_ $prec_type _ $bin_type>] {
				hiob: HIOB<$prec_type,$bin_type>
			}
			#[pymethods]
			impl [<HIOB_ $prec_type _ $bin_type>] {
				#[new]
				pub fn new(
					data: PyReadonlyArray2<$prec_type>,
					n_bits: usize,
					scale: Option<$prec_type>,
					centers: Option<PyReadonlyArray2<$prec_type>>,
					init_greedy: Option<bool>,
					init_ransac: Option<bool>,
					ransac_pairs_per_bit: Option<usize>,
					ransac_sub_sample: Option<usize>
				) -> Self {
					Self{hiob: HIOB::new(
						data.as_array().into_owned(),
						n_bits,
						scale,
						if centers.is_some() {
							Some(centers.unwrap().as_array().into_owned())
						} else { None },
						init_greedy,
						init_ransac,
						ransac_pairs_per_bit,
						ransac_sub_sample
					)}
				}
				pub fn run(&mut self, n_iterations: usize) {
					self.hiob.run(n_iterations);
				}
				pub fn binarize<'py>(&self, py: Python<'py>, queries: PyReadonlyArray2<$prec_type>) -> &'py PyArray2<$bin_type> {
					self.hiob.binarize(&queries.as_array()).to_pyarray(py)
				}
				pub fn binarize_h5<'py>(&self, py: Python<'py>, file: String, dataset: String, batch_size: Option<usize>) -> PyResult<&'py PyArray2<$bin_type>> {
					let result = self.hiob.binarize_h5(file.as_str(), dataset.as_str(), batch_size.unwrap_or(1000));
					if result.is_ok() {
						Ok(result.unwrap().to_pyarray(py))
					} else {
						Err(PyValueError::new_err(result.unwrap_err().to_string()))
					}
				}
				#[getter]
				pub fn get_n_data(&self) -> PyResult<usize> {
					Ok(self.hiob.get_n_data())
				}
				#[getter]
				pub fn get_n_dims(&self) -> PyResult<usize> {
					Ok(self.hiob.get_n_dims())
				}
				#[getter]
				pub fn get_n_bits(&self) -> PyResult<usize> {
					Ok(self.hiob.get_n_bits())
				}
				#[getter]
				pub fn get_scale(&self) -> PyResult<$prec_type> {
					Ok(self.hiob.get_scale())
				}
				#[setter]
				pub fn set_scale(&mut self, scale: $prec_type) -> PyResult<()> {
					Ok(self.hiob.set_scale(scale))
				}
				#[getter]
				pub fn get_data<'py>(&self, py: Python<'py>) -> &'py PyArray2<$prec_type> {
					self.hiob.get_data().to_pyarray(py)
				}
				#[getter]
				pub fn get_centers<'py>(&self, py: Python<'py>) -> &'py PyArray2<$prec_type> {
					self.hiob.get_centers().to_pyarray(py)
				}
				// #[getter]
				// pub fn get_data_bins<'py>(&self, py: Python<'py>) -> &'py PyArray2<$bin_type> {
				// 	self.hiob.get_data_bins().to_pyarray(py)
				// }
				#[getter]
				pub fn get_overlap_mat<'py>(&self, py: Python<'py>) -> &'py PyArray2<usize> {
					self.hiob.get_overlap_mat().to_pyarray(py)
				}
				#[getter]
				pub fn get_sim_mat<'py>(&self, py: Python<'py>) -> &'py PyArray2<f64> {
					self.hiob.get_sim_mat().to_pyarray(py)
				}
				#[getter]
				pub fn get_sim_sums<'py>(&self, py: Python<'py>) -> &'py PyArray1<f64> {
					self.hiob.get_sim_sums().to_pyarray(py)
				}
				#[getter]
				pub fn get_update_parallel(&self) -> PyResult<bool> {
					Ok(self.hiob.get_update_parallel())
				}
				#[setter]
				pub fn set_update_parallel(&mut self, b: bool) -> PyResult<()> {
					Ok(self.hiob.set_update_parallel(b))
				}
				#[getter]
				pub fn get_displace_parallel(&self) -> PyResult<bool> {
					Ok(self.hiob.get_displace_parallel())
				}
				#[setter]
				pub fn set_displace_parallel(&mut self, b: bool) -> PyResult<()> {
					Ok(self.hiob.set_displace_parallel(b))
				}
			}
		}
	};
}
// hiob_struct_gen!((f32, f64), (bool, i8, i16, i32, i64, u8, u16, u32, u64));
hiob_struct_gen!((f32, f64), (bool, u8, u16, u32, u64));
macro_rules! hiob_python_export {
	($module: ident, ($($pts:ty),*), $bts:tt) => {
		$(hiob_python_export!($module, $pts, $bts);)*
	};
	($module: ident, $prec_type: ty, ($($bts:ty),*)) => {
		$(hiob_python_export!($module, $prec_type, $bts);)*
	};
	($module: ident, $prec_type: ty, $bin_type: ty) => {
		paste!{
			$module.add_class::<[<HIOB_ $prec_type _ $bin_type>]>()?;
		}
	};
}


macro_rules! stochastic_hiob_struct_gen {
	($datasource: ident, ($($pts:ty),*), $bts:tt) => {
		$(stochastic_hiob_struct_gen!($datasource, $pts, $bts);)*
	};
	($datasource: ident, $prec_type: ty, ($($bts:ty),*)) => {
		$(stochastic_hiob_struct_gen!($datasource, $prec_type, $bts);)*
	};
	(H5, $prec_type: ty, $bin_type: ty) => {
		paste! {
			#[allow(non_camel_case_types)]
			#[pyclass]
			pub struct [<StochasticHIOB_H5_ $prec_type _ $bin_type>] {
				shiob: StochasticHIOB<$prec_type,$bin_type,H5PyDataset<$prec_type>>
			}
			#[pymethods]
			impl [<StochasticHIOB_H5_ $prec_type _ $bin_type>] {
				#[new]
				pub fn new(
					file: String,
					dataset: String,
					sample_size: usize,
					its_per_sample: usize,
					n_bits: usize,
					perm_gen_rounds: Option<usize>,
					scale: Option<$prec_type>,
					centers: Option<PyReadonlyArray2<$prec_type>>,
					init_greedy: Option<bool>,
					init_ransac: Option<bool>,
					ransac_pairs_per_bit: Option<usize>,
					ransac_sub_sample: Option<usize>
				) -> PyResult<Self> {
					let data_source = H5PyDataset::<$prec_type>::new(file.as_str(), dataset.as_str());
					Ok(Self{shiob: StochasticHIOB::new(
						data_source,
						sample_size,
						its_per_sample,
						n_bits,
						perm_gen_rounds,
						scale,
						if centers.is_some() {
							Some(centers.unwrap().as_array().into_owned())
						} else { None },
						init_greedy,
						init_ransac,
						ransac_pairs_per_bit,
						ransac_sub_sample
					)})
				}
			}
			stochastic_hiob_struct_gen!(funs H5, $prec_type, $bin_type);
		}
	};
	(ND, $prec_type: ty, $bin_type: ty) => {
		paste! {
			#[allow(non_camel_case_types)]
			#[pyclass]
			pub struct [<StochasticHIOB_ND_ $prec_type _ $bin_type>] {
				shiob: StochasticHIOB<$prec_type,$bin_type,Array2<$prec_type>>
			}
			#[pymethods]
			impl [<StochasticHIOB_ND_ $prec_type _ $bin_type>] {
				#[new]
				pub fn new(
					data: PyReadonlyArray2<$prec_type>,
					sample_size: usize,
					its_per_sample: usize,
					n_bits: usize,
					perm_gen_rounds: Option<usize>,
					scale: Option<$prec_type>,
					centers: Option<PyReadonlyArray2<$prec_type>>,
					init_greedy: Option<bool>,
					init_ransac: Option<bool>,
					ransac_pairs_per_bit: Option<usize>,
					ransac_sub_sample: Option<usize>
				) -> PyResult<Self> {
					Ok(Self{shiob: StochasticHIOB::new(
						data.as_array().into_owned(),
						sample_size,
						its_per_sample,
						n_bits,
						perm_gen_rounds,
						scale,
						if centers.is_some() {
							Some(centers.unwrap().as_array().into_owned())
						} else { None },
						init_greedy,
						init_ransac,
						ransac_pairs_per_bit,
						ransac_sub_sample
					)})
				}
			}
			stochastic_hiob_struct_gen!(funs ND, $prec_type, $bin_type);
		}
	};
	(funs $datasource: ident, $prec_type: ty, $bin_type: ty) => {
		paste! {
			#[pymethods]
			impl [<StochasticHIOB_ $datasource _ $prec_type _ $bin_type>] {
				pub fn run(&mut self, n_iterations: usize) {
					self.shiob.run(n_iterations);
				}
				pub fn binarize<'py>(&self, py: Python<'py>, queries: PyReadonlyArray2<$prec_type>) -> &'py PyArray2<$bin_type> {
					self.shiob.binarize(&queries.as_array()).to_pyarray(py)
				}
				pub fn binarize_h5<'py>(&self, py: Python<'py>, file: String, dataset: String, batch_size: Option<usize>) -> PyResult<&'py PyArray2<$bin_type>> {
					let result = self.shiob.binarize_h5(file.as_str(), dataset.as_str(), batch_size.unwrap_or(1000));
					if result.is_ok() {
						Ok(result.unwrap().to_pyarray(py))
					} else {
						Err(PyValueError::new_err(result.unwrap_err().to_string()))
					}
				}
				#[getter]
				pub fn get_sample_size(&self) -> PyResult<usize> {
					Ok(self.shiob.get_sample_size())
				}
				#[setter]
				pub fn set_sample_size(&mut self, value: usize) -> PyResult<()> {
					self.shiob.set_sample_size(value); Ok(())
				}
				#[getter]
				pub fn get_its_per_sample(&self) -> PyResult<usize> {
					Ok(self.shiob.get_its_per_sample())
				}
				#[setter]
				pub fn set_its_per_sample(&mut self, value: usize) -> PyResult<()> {
					self.shiob.set_its_per_sample(value); Ok(())
				}
				#[getter]
				pub fn get_n_samples(&self) -> PyResult<usize> {
					Ok(self.shiob.get_n_samples())
				}
				#[getter]
				pub fn get_n_dims(&self) -> PyResult<usize> {
					Ok(self.shiob.get_n_dims())
				}
				#[getter]
				pub fn get_n_bits(&self) -> PyResult<usize> {
					Ok(self.shiob.get_n_bits())
				}
				#[getter]
				pub fn get_scale(&self) -> PyResult<$prec_type> {
					Ok(self.shiob.get_scale())
				}
				#[setter]
				pub fn set_scale(&mut self, scale: $prec_type) -> PyResult<()> {
					Ok(self.shiob.set_scale(scale))
				}
				#[getter]
				pub fn get_data<'py>(&self, py: Python<'py>) -> &'py PyArray2<$prec_type> {
					self.shiob.get_data().to_pyarray(py)
				}
				#[getter]
				pub fn get_centers<'py>(&self, py: Python<'py>) -> &'py PyArray2<$prec_type> {
					self.shiob.get_centers().to_pyarray(py)
				}
				// #[getter]
				// pub fn get_data_bins<'py>(&self, py: Python<'py>) -> &'py PyArray2<$bin_type> {
				// 	self.shiob.get_data_bins().to_pyarray(py)
				// }
				#[getter]
				pub fn get_overlap_mat<'py>(&self, py: Python<'py>) -> &'py PyArray2<usize> {
					self.shiob.get_overlap_mat().to_pyarray(py)
				}
				#[getter]
				pub fn get_sim_mat<'py>(&self, py: Python<'py>) -> &'py PyArray2<f64> {
					self.shiob.get_sim_mat().to_pyarray(py)
				}
				#[getter]
				pub fn get_sim_sums<'py>(&self, py: Python<'py>) -> &'py PyArray1<f64> {
					self.shiob.get_sim_sums().to_pyarray(py)
				}
				#[getter]
				pub fn get_update_parallel(&self) -> PyResult<bool> {
					Ok(self.shiob.get_update_parallel())
				}
				#[setter]
				pub fn set_update_parallel(&mut self, b: bool) -> PyResult<()> {
					Ok(self.shiob.set_update_parallel(b))
				}
				#[getter]
				pub fn get_displace_parallel(&self) -> PyResult<bool> {
					Ok(self.shiob.get_displace_parallel())
				}
				#[setter]
				pub fn set_displace_parallel(&mut self, b: bool) -> PyResult<()> {
					Ok(self.shiob.set_displace_parallel(b))
				}
			}
		}
	}
}
// stochastic_hiob_struct_gen!(H5, (f32, f64), (bool, i8, i16, i32, i64, u8, u16, u32, u64));
// stochastic_hiob_struct_gen!(ND, (f32, f64), (bool, i8, i16, i32, i64, u8, u16, u32, u64));
stochastic_hiob_struct_gen!(H5, (f32, f64), (bool, u8, u16, u32, u64));
stochastic_hiob_struct_gen!(ND, (f32, f64), (bool, u8, u16, u32, u64));
macro_rules! stochastic_hiob_python_export {
	($module: ident, ($($pts:ty),*), $bts:tt) => {
		$(stochastic_hiob_python_export!($module, $pts, $bts);)*
	};
	($module: ident, $prec_type: ty, ($($bts:ty),*)) => {
		$(stochastic_hiob_python_export!($module, $prec_type, $bts);)*
	};
	($module: ident, $prec_type: ty, $bin_type: ty) => {
		paste!{
			$module.add_class::<[<StochasticHIOB_ND_ $prec_type _ $bin_type>]>()?;
			$module.add_class::<[<StochasticHIOB_H5_ $prec_type _ $bin_type>]>()?;
		}
	};
}


macro_rules! eval_fun_gen {
	() => {
		#[pymethods]
		impl RawBinarizationEvaluator {
			#[new]
			pub fn new() -> Self { RawBinarizationEvaluator{bin_eval: BinarizationEvaluator::new()} }
			pub fn k_at_n_recall_prec_all(
				&self,
				dot_neighbors: PyReadonlyArray2<usize>,
				hamming_neighbors: PyReadonlyArray2<usize>
			) -> PyResult<f64> {
				Ok(
					self.bin_eval.k_at_n_recall_prec_all(
						&dot_neighbors.as_array(),
						&hamming_neighbors.as_array()
					)
				)
			}
		}
	};
}
macro_rules! eval_fun_gen_p {
	($prec_type:ty $(, $pts:ty)+) => {
		eval_fun_gen_p!($prec_type);
		$(eval_fun_gen_p!($pts);)*
	};
	($prec_type:ty) => {
		paste! {
			#[pymethods]
			impl RawBinarizationEvaluator {
				pub fn [<brute_force_k_largest_dot_ $prec_type>]<'py>(
					&self,
					py: Python<'py>,
					data: PyReadonlyArray2<$prec_type>,
					queries: PyReadonlyArray2<$prec_type>,
					k: usize
				) -> (&'py PyArray2<$prec_type>, &'py PyArray2<usize>) {
					let (dots, idxs) = self.bin_eval.brute_force_k_largest_dot(
						&data.as_array(),
						&queries.as_array(),
						k
					);
					(dots.to_pyarray(py), idxs.to_pyarray(py))
				}
				pub fn [<k_at_n_recall_prec_hamming_neighbors_ $prec_type>](
					&self,
					data_bin: PyReadonlyArray2<$prec_type>,
					queries_bin: PyReadonlyArray2<$prec_type>,
					hamming_neighbors: PyReadonlyArray2<usize>,
					k: usize
				) -> PyResult<f64> {
					Ok(
						self.bin_eval.k_at_n_recall_prec_hamming_neighbors(
							&data_bin.as_array(),
							&queries_bin.as_array(),
							&hamming_neighbors.as_array(),
							k
						)
					)
				}
			}
		}
	};
}
macro_rules! eval_fun_gen_b {
	($bin_type:ty $(, $bts:ty)+) => {
		eval_fun_gen_b!($bin_type);
		$(eval_fun_gen_b!($bts);)*
	};
	($bin_type:ty) => {
		paste! {
			#[pymethods]
			impl RawBinarizationEvaluator {
				pub fn [<brute_force_k_smallest_hamming_ $bin_type>]<'py>(
					&self,
					py: Python<'py>,
					data_bin: PyReadonlyArray2<$bin_type>,
					queries_bin: PyReadonlyArray2<$bin_type>,
					k: usize
				) -> (&'py PyArray2<usize>, &'py PyArray2<usize>) {
					let (dists, idxs) = self.bin_eval.brute_force_k_smallest_hamming(
						&data_bin.as_array(),
						&queries_bin.as_array(),
						k
					);
					(dists.to_pyarray(py), idxs.to_pyarray(py))
				}
				pub fn [<k_at_n_recall_prec_dot_neighbors_ $bin_type>](
					&self,
					data_bin: PyReadonlyArray2<$bin_type>,
					queries_bin: PyReadonlyArray2<$bin_type>,
					true_neighbors: PyReadonlyArray2<usize>,
					n: usize
				) -> PyResult<f64> {
					Ok(
						self.bin_eval.k_at_n_recall_prec_dot_neighbors(
							&data_bin.as_array(),
							&queries_bin.as_array(),
							&true_neighbors.as_array(),
							n
						)
					)
				}
			}
		}
	};
}
macro_rules! eval_fun_gen_pb {
	(($($pts:ty),*), $bts:tt) => {
		$(eval_fun_gen_pb!($pts, $bts);)*
	};
	($prec_type: ty, ($($bts:ty),*)) => {
		$(eval_fun_gen_pb!($prec_type, $bts);)*
	};
	($prec_type: ty, $bin_type: ty) => {
		paste! {
			#[pymethods]
			impl RawBinarizationEvaluator {
				pub fn [<k_at_n_recall_ $prec_type _ $bin_type>](
					&self,
					data: PyReadonlyArray2<$prec_type>,
					data_bin: PyReadonlyArray2<$bin_type>,
					queries: PyReadonlyArray2<$prec_type>,
					queries_bin: PyReadonlyArray2<$bin_type>,
					k: usize,
					n: usize
				) -> PyResult<f64> {
					Ok(
						self.bin_eval.k_at_n_recall(
							&data.as_array(),
							&data_bin.as_array(),
							&queries.as_array(),
							&queries_bin.as_array(),
							k,
							n
						)
					)
				}
			}
		}
	}
}
#[pyclass]
pub struct RawBinarizationEvaluator {
	bin_eval: BinarizationEvaluator
}
eval_fun_gen!();
eval_fun_gen_p!(f32, f64);
// eval_fun_gen_b!(bool, i8, i16, i32, i64, u8, u16, u32, u64);
eval_fun_gen_b!(bool, u8, u16, u32, u64);
// eval_fun_gen_pb!((f32, f64), (bool, i8, i16, i32, i64, u8, u16, u32, u64));
eval_fun_gen_pb!((f32, f64), (bool, u8, u16, u32, u64));


macro_rules! thx_struct_gen {
	(($($bts:ty),*), $fs:tt) => {
		$(thx_struct_gen!($bts, $fs);)*
	};
	($bits_type: ty, ($($fs:literal),*)) => {
		$(thx_struct_gen!($bits_type, $fs);)*
	};
	($bits_type: ty, $fanout: literal) => {
		paste! {
			#[allow(non_camel_case_types)]
			#[pyclass]
			pub struct [<THX_ $bits_type _ $fanout>] {
				thx: THX<$bits_type,OwnedRepr<$bits_type>,$fanout>
			}
			#[pymethods]
			impl [<THX_ $bits_type _ $fanout>] {
				#[new]
				pub fn new(data: PyReadonlyArray2<$bits_type>, n_bits_per_layer: usize) -> Self {
					Self{thx: THX::new(data.as_array().into_owned(), n_bits_per_layer)}
				}
				pub fn query_approx<'py>(&self, py: Python<'py>, queries: PyReadonlyArray2<$bits_type>, k_neighbors: usize) -> (&'py PyArray2<usize>, &'py PyArray2<usize>) {
					let (dists, idx) = self.thx.query_approx(&queries.as_array(), k_neighbors);
					(dists.to_pyarray(py), idx.to_pyarray(py))
				}
				pub fn query_approx_single<'py>(&self, py: Python<'py>, queries: PyReadonlyArray1<$bits_type>, k_neighbors: usize) -> (&'py PyArray1<usize>, &'py PyArray1<usize>) {
					let (dists, idx) = self.thx.query_approx_single(&queries.as_array(), k_neighbors);
					(dists.to_pyarray(py), idx.to_pyarray(py))
				}
				pub fn query_range_approx(&self, queries: PyReadonlyArray2<$bits_type>, max_dist: usize) -> (Vec<Vec<usize>>, Vec<Vec<usize>>) {
					self.thx.query_range_approx(&queries.as_array(), max_dist)
				}
				pub fn query_range_approx_single(&self, queries: PyReadonlyArray1<$bits_type>, max_dist: usize) -> (Vec<usize>, Vec<usize>) {
					self.thx.query_range_approx_single(&queries.as_array(), max_dist)
				}
				#[getter]
				pub fn get_n_nodes(&self) -> PyResult<usize> {
					Ok(self.thx.get_n_nodes())
				}
				#[getter]
				pub fn get_height(&self) -> PyResult<usize> {
					Ok(self.thx.get_height())
				}
				#[staticmethod]
				pub fn compute_n_nodes(data: PyReadonlyArray2<$bits_type>, n_bits_per_layer: usize) -> PyResult<usize> {
					let total_bits = data.as_array().row(0).size();
					Ok(THX::<$bits_type,OwnedRepr<$bits_type>,$fanout>::compute_n_nodes(total_bits, n_bits_per_layer))
				}
			}
		}
	}
}
// thx_struct_gen!((bool, i8, i16, i32, i64, u8, u16, u32, u64), (2,3,4,5,6,7,8,9,10));
thx_struct_gen!((bool, u8, u16, u32, u64), (2,3,4,5,6,7,8,9,10));
macro_rules! thx_python_export {
	($module: ident, ($($bts:ty),*), $fs:tt) => {
		$(thx_python_export!($module, $bts, $fs);)*
	};
	($module: ident, $bits_type: ty, ($($fs:literal),*)) => {
		$(thx_python_export!($module, $bits_type, $fs);)*
	};
	($module: ident, $bits_type: ty, $fanout: literal) => {
		paste!{
			$module.add_class::<[<THX_ $bits_type _ $fanout>]>()?;
		}
	};
}


#[pyfunction]
pub fn limit_threads(_num_threads: usize) -> Result<(), PyErr> {
	#[cfg(feature="parallel")]
	{
		let result = ThreadPoolBuilder::new().num_threads(_num_threads).build_global();
		if result.is_ok() {
			Ok(())
		} else {
			Err(PyErr::new::<PyValueError,_>(result.err().unwrap().to_string()))
		}
	}
	#[cfg(not(feature="parallel"))]
	Err(PyErr::new::<PyWarning,_>("Number of threads could not be set, because this package was built without multi-threading."))
}


/* Declaration of the python package generated by maturin. */
#[pymodule]
fn hiob(_py: Python, m: &PyModule) -> PyResult<()> {
	// hiob_python_export!(m, (f32, f64), (bool, i8, i16, i32, i64, u8, u16, u32, u64));
	// stochastic_hiob_python_export!(m, (f32, f64), (bool, i8, i16, i32, i64, u8, u16, u32, u64));
	// thx_python_export!(m, (bool, i8, i16, i32, i64, u8, u16, u32, u64), (2,3,4,5,6,7,8,9,10));
	hiob_python_export!(m, (f32, f64), (bool, u8, u16, u32, u64));
	stochastic_hiob_python_export!(m, (f32, f64), (bool, u8, u16, u32, u64));
	thx_python_export!(m, (bool, u8, u16, u32, u64), (2,3,4,5,6,7,8,9,10));
	m.add_class::<RawBinarizationEvaluator>()?;
	m.add_wrapped(wrap_pyfunction!(limit_threads))?;
	Ok(())
}
