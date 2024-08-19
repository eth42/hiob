#![allow(non_camel_case_types)]
use pyo3::prelude::*;
use numpy::{PyArray1,PyArray2,PyReadonlyArray1,PyReadonlyArray2,ToPyArray};
use num::NumCast;
use paste::paste;
use ndarray::{Array2,OwnedRepr};
use pyo3::exceptions::PyValueError;
use pyo3::types::PyType;
#[cfg(feature="half")]
use half::f16;

use crate::binarizer::{HIOB,StochasticHIOB,HIOBParams,StochasticHIOBParams};
use crate::pydata::H5PyDataset;
use crate::eval::BinarizationEvaluator;
use crate::bit_vectors::BitVector;
use crate::index::THX;
use crate::min_hash_search::{MinHashSearcher, ChunkyMinHashSearcher};


macro_rules! get_gen {
	($($obj: ident).+, $field: ident) => { paste! {
		Ok($($obj).+.[<get_ $field>]())
	}};
	($($obj: ident).+, $field: ident => $cast: ty) => { paste! {
		Ok(<$cast as NumCast>::from($($obj).+.[<get_ $field>]()).unwrap())
	}};
	($($obj: ident).+, $field: ident O=> $cast: ty) => { paste! {
		Ok($($obj).+.[<get_ $field>]().map(|v| <$cast as NumCast>::from(v).unwrap()))
	}};
	($py: ident $($obj: ident).+, $field: ident) => { paste! {
		$($obj).+.[<get_ $field>]().to_pyarray($py)
	}};
}
macro_rules! set_gen {
	($($obj: ident).+, $field: ident) => { paste! {
		Ok($($obj).+.[<set_ $field>]($field))
	}};
	($($obj: ident).+, $field: ident => $cast: ty) => { paste! {
		Ok($($obj).+.[<set_ $field>](<$cast as NumCast>::from($field).unwrap()))
	}};
	($($obj: ident).+, $field: ident O=> $cast: ty) => { paste! {
		Ok($($obj).+.[<set_ $field>]($field.map(|v| <$cast as NumCast>::from(v).unwrap())))
	}};
	// (py $($obj: ident).+, $field: ident) => { paste! {
	// 	$($obj).+.[<set_ $field>](i_center, &center.as_array());
	// }};
}
// macro_rules! set_gen {
// 	() => {};
// 	/* @params value: int, ... */
// 	(@$attribute: ident $field: ident: $type: ty $(, $($rest: tt)+)?) => {
// 		paste! {
// 			pub fn [<set_ $field>](&mut self, $field: $type) { self.$attribute.[<set_ $field>]($field) }
// 		}
// 		$(set_gen!($($rest)+);)?
// 	};
// 	/* params value: int, ... */
// 	($attribute: ident $field: ident: $type: ty $(, $($rest: tt)+)?) => {
// 		paste! {
// 			pub fn [<set_ $field>](&mut self, $field: $type) { self.$attribute.$field = $field }
// 		}
// 		$(set_gen!($($rest)+);)?
// 	};
// 	/* value: int, ... */
// 	($field: ident: $type: ty $(, $($rest: tt)+)?) => {
// 		paste! {
// 			pub fn [<set_ $field>](&mut self, $field: $type) { self.$field = $field }
// 		}
// 		$(set_gen!($($rest)+);)?
// 	};
// }
// macro_rules! get_set_gen {
// 	($($rest: tt)+) => {
// 		get_gen!($($rest)+);
// 		set_gen!($($rest)+);
// 	};
// }


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
					affine: bool,
					scale: Option<f64>,
					centers: Option<PyReadonlyArray2<$prec_type>>,
					center_biases: Option<PyReadonlyArray1<$prec_type>>,
					balance_regression_factor: Option<f64>,
					init_greedy: Option<bool>,
					init_ransac: Option<bool>,
					ransac_pairs_per_bit: Option<usize>,
					ransac_sub_sample: Option<usize>
				) -> Self {
					Self{hiob: HIOB::new(
						data.as_array().into_owned(),
						n_bits,
						HIOBParams::new()
						.with_affine(affine)
						.maybe_with_scale(scale.map(|v| <$prec_type as NumCast>::from(v).unwrap()))
						.with_centers(centers.map(|v| v.as_array().into_owned()))
						.with_center_biases(center_biases.map(|v| v.as_array().into_owned()))
						.maybe_with_balance_regression_factor(balance_regression_factor.map(|v| <$prec_type as NumCast>::from(v).unwrap()))
						.maybe_with_init_greedy(init_greedy)
						.maybe_with_init_ransac(init_ransac)
						.maybe_with_ransac_pairs_per_bit(ransac_pairs_per_bit)
						.maybe_with_ransac_sub_sample(ransac_sub_sample)
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
					get_gen!(self.hiob, n_data)
				}
				#[getter]
				pub fn get_n_dims(&self) -> PyResult<usize> {
					get_gen!(self.hiob, n_dims)
				}
				#[getter]
				pub fn get_n_bits(&self) -> PyResult<usize> {
					get_gen!(self.hiob, n_bits)
				}
				#[getter]
				pub fn get_scale(&self) -> PyResult<f64> {
					get_gen!(self.hiob, scale => f64)
				}
				#[setter]
				pub fn set_scale(&mut self, scale: f64) -> PyResult<()> {
					set_gen!(self.hiob, scale => $prec_type)
				}
				#[getter]
				pub fn get_balance_regression_factor(&self) -> PyResult<f64> {
					get_gen!(self.hiob, balance_regression_factor => f64)
				}
				#[setter]
				pub fn set_balance_regression_factor(&mut self, balance_regression_factor: f64) -> PyResult<()> {
					set_gen!(self.hiob, balance_regression_factor => $prec_type)
				}
				#[getter]
				pub fn get_data<'py>(&self, py: Python<'py>) -> &'py PyArray2<$prec_type> {
					get_gen!(py self.hiob, data)
				}
				#[getter]
				pub fn get_centers<'py>(&self, py: Python<'py>) -> &'py PyArray2<$prec_type> {
					get_gen!(py self.hiob, centers)
				}
				#[getter]
				pub fn get_is_affine(&self) -> PyResult<bool> {
					get_gen!(self.hiob, affine)
				}
				#[getter]
				pub fn get_center_biases<'py>(&self, py: Python<'py>) -> &'py PyArray1<$prec_type> {
					get_gen!(py self.hiob, center_biases)
				}
				pub fn set_center(&mut self, i_center: usize, center: PyReadonlyArray1<$prec_type>) {
					self.hiob.set_center(i_center, &center.as_array());
				}
				pub fn set_bias(&mut self, i_center: usize, bias: f64) {
					self.hiob.set_bias(i_center, <$prec_type as NumCast>::from(bias).unwrap());
				}
				pub fn set_center_bias(&mut self, i_center: usize, center: PyReadonlyArray1<$prec_type>, bias: f64) {
					self.hiob.set_center_bias(i_center, &center.as_array(), <$prec_type as NumCast>::from(bias).unwrap());
				}
				// #[getter]
				// pub fn get_data_bins<'py>(&self, py: Python<'py>) -> &'py PyArray2<$bin_type> {
				// 	self.hiob.get_data_bins().to_pyarray(py)
				// }
				#[getter]
				pub fn get_overlap_mat<'py>(&self, py: Python<'py>) -> &'py PyArray2<usize> {
					get_gen!(py self.hiob, overlap_mat)
				}
				#[getter]
				pub fn get_sim_mat<'py>(&self, py: Python<'py>) -> &'py PyArray2<f64> {
					get_gen!(py self.hiob, sim_mat)
				}
				#[getter]
				pub fn get_sim_sums<'py>(&self, py: Python<'py>) -> &'py PyArray1<f64> {
					get_gen!(py self.hiob, sim_sums)
				}
				#[getter]
				pub fn get_update_parallel(&self) -> PyResult<bool> {
					get_gen!(self.hiob, update_parallel)
				}
				#[setter]
				pub fn set_update_parallel(&mut self, update_parallel: bool) -> PyResult<()> {
					set_gen!(self.hiob, update_parallel)
				}
				#[getter]
				pub fn get_displace_parallel(&self) -> PyResult<bool> {
					get_gen!(self.hiob, displace_parallel)
				}
				#[setter]
				pub fn set_displace_parallel(&mut self, displace_parallel: bool) -> PyResult<()> {
					set_gen!(self.hiob, displace_parallel)
				}
			}
		}
	};
}
// hiob_struct_gen!((f32, f64), (bool, i8, i16, i32, i64, u8, u16, u32, u64));
hiob_struct_gen!((f32, f64), (bool, u8, u16, u32, u64));
#[cfg(feature="half")]
hiob_struct_gen!(f16, (bool, u8, u16, u32, u64));
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
					n_bits: usize,
					sample_size: Option<usize>,
					its_per_sample: Option<usize>,
					affine: Option<bool>,
					inversive: Option<bool>,
					n_inverter_init_samples: Option<usize>,
					kernelized: Option<bool>,
					kernel_reshape: Option<Vec<usize>>,
					kernel_width: Option<usize>,
					perm_gen_rounds: Option<usize>,
					scale: Option<f64>,
					centers: Option<PyReadonlyArray2<$prec_type>>,
					center_biases: Option<PyReadonlyArray1<$prec_type>>,
					balance_regression_factor: Option<f64>,
					init_greedy: Option<bool>,
					init_ransac: Option<bool>,
					init_itq: Option<bool>,
					ransac_pairs_per_bit: Option<usize>,
					ransac_sub_sample: Option<usize>,
					itq_central: Option<bool>,
					itq_iterations: Option<usize>,
					noise_std: Option<f64>,
					pre_noise: Option<bool>,
					update_parallel: Option<bool>,
					displace_parallel: Option<bool>,
				) -> PyResult<Self> {
					let data_source = H5PyDataset::<$prec_type>::new(file.as_str(), dataset.as_str());
					Ok(Self{shiob: StochasticHIOB::new(
						data_source,
						n_bits,
						StochasticHIOBParams::new()
						.maybe_with_sample_size(sample_size)
						.maybe_with_its_per_sample(its_per_sample)
						.maybe_with_inversive(inversive)
						.maybe_with_n_inverter_init_samples(n_inverter_init_samples)
						.maybe_with_kernelized(kernelized)
						.maybe_with_kernel_reshape(kernel_reshape)
						.maybe_with_kernel_width(kernel_width)
						.maybe_with_perm_gen_rounds(perm_gen_rounds)
						.with_noise_std(noise_std.map(|v| <$prec_type as NumCast>::from(v).unwrap()))
						.maybe_with_pre_noise(pre_noise),
						HIOBParams::new()
						.maybe_with_affine(affine)
						.maybe_with_scale(scale.map(|v| <$prec_type as NumCast>::from(v).unwrap()))
						.with_centers(centers.map(|v| v.as_array().into_owned()))
						.with_center_biases(center_biases.map(|v| v.as_array().into_owned()))
						.maybe_with_balance_regression_factor(balance_regression_factor.map(|v| <$prec_type as NumCast>::from(v).unwrap()))
						.maybe_with_init_greedy(init_greedy)
						.maybe_with_init_ransac(init_ransac)
						.maybe_with_init_itq(init_itq)
						.maybe_with_ransac_pairs_per_bit(ransac_pairs_per_bit)
						.maybe_with_ransac_sub_sample(ransac_sub_sample)
						.maybe_with_itq_central(itq_central)
						.maybe_with_itq_iterations(itq_iterations)
						.maybe_with_update_parallel(update_parallel)
						.maybe_with_displace_parallel(displace_parallel),
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
					n_bits: usize,
					sample_size: Option<usize>,
					its_per_sample: Option<usize>,
					affine: Option<bool>,
					inversive: Option<bool>,
					n_inverter_init_samples: Option<usize>,
					kernelized: Option<bool>,
					kernel_reshape: Option<Vec<usize>>,
					kernel_width: Option<usize>,
					perm_gen_rounds: Option<usize>,
					scale: Option<f64>,
					centers: Option<PyReadonlyArray2<$prec_type>>,
					center_biases: Option<PyReadonlyArray1<$prec_type>>,
					balance_regression_factor: Option<f64>,
					init_greedy: Option<bool>,
					init_ransac: Option<bool>,
					init_itq: Option<bool>,
					ransac_pairs_per_bit: Option<usize>,
					ransac_sub_sample: Option<usize>,
					itq_central: Option<bool>,
					itq_iterations: Option<usize>,
					noise_std: Option<f64>,
					pre_noise: Option<bool>,
					update_parallel: Option<bool>,
					displace_parallel: Option<bool>,
				) -> PyResult<Self> {
					Ok(Self{shiob: StochasticHIOB::new(
						data.as_array().into_owned(),
						n_bits,
						StochasticHIOBParams::new()
						.maybe_with_sample_size(sample_size)
						.maybe_with_its_per_sample(its_per_sample)
						.maybe_with_inversive(inversive)
						.maybe_with_n_inverter_init_samples(n_inverter_init_samples)
						.maybe_with_kernelized(kernelized)
						.maybe_with_kernel_reshape(kernel_reshape)
						.maybe_with_kernel_width(kernel_width)
						.maybe_with_perm_gen_rounds(perm_gen_rounds)
						.with_noise_std(noise_std.map(|v| <$prec_type as NumCast>::from(v).unwrap()))
						.maybe_with_pre_noise(pre_noise),
						HIOBParams::new()
						.maybe_with_affine(affine)
						.maybe_with_scale(scale.map(|v| <$prec_type as NumCast>::from(v).unwrap()))
						.with_centers(centers.map(|v| v.as_array().into_owned()))
						.with_center_biases(center_biases.map(|v| v.as_array().into_owned()))
						.maybe_with_balance_regression_factor(balance_regression_factor.map(|v| <$prec_type as NumCast>::from(v).unwrap()))
						.maybe_with_init_greedy(init_greedy)
						.maybe_with_init_ransac(init_ransac)
						.maybe_with_init_itq(init_itq)
						.maybe_with_ransac_pairs_per_bit(ransac_pairs_per_bit)
						.maybe_with_ransac_sub_sample(ransac_sub_sample)
						.maybe_with_itq_central(itq_central)
						.maybe_with_itq_iterations(itq_iterations)
						.maybe_with_update_parallel(update_parallel)
						.maybe_with_displace_parallel(displace_parallel),
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
				pub fn get_inversive_balls<'py>(&self, py: Python<'py>) -> Option<(&'py PyArray2<$prec_type>, &'py PyArray1<$prec_type>)> {
					self.shiob.get_inversive_balls().map(|(a,b)| (a.to_pyarray(py), b.to_pyarray(py)))
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
					get_gen!(self.shiob, sample_size)
				}
				#[setter]
				pub fn set_sample_size(&mut self, sample_size: usize) -> PyResult<()> {
					set_gen!(self.shiob, sample_size)
				}
				#[getter]
				pub fn get_its_per_sample(&self) -> PyResult<usize> {
					get_gen!(self.shiob, its_per_sample)
				}
				#[setter]
				pub fn set_its_per_sample(&mut self, its_per_sample: usize) -> PyResult<()> {
					set_gen!(self.shiob, its_per_sample)
				}
				#[getter]
				pub fn get_n_data(&self) -> PyResult<usize> {
					get_gen!(self.shiob, n_data)
				}
				#[getter]
				pub fn get_n_dims(&self) -> PyResult<usize> {
					get_gen!(self.shiob, n_dims)
				}
				#[getter]
				pub fn get_n_bits(&self) -> PyResult<usize> {
					get_gen!(self.shiob, n_bits)
				}
				#[getter]
				pub fn get_scale(&self) -> PyResult<f64> {
					get_gen!(self.shiob, scale => f64)
				}
				#[setter]
				pub fn set_scale(&mut self, scale: f64) -> PyResult<()> {
					set_gen!(self.shiob, scale => $prec_type)
				}
				#[getter]
				pub fn get_balance_regression_factor(&self) -> PyResult<f64> {
					get_gen!(self.shiob, balance_regression_factor => f64)
				}
				#[setter]
				pub fn set_balance_regression_factor(&mut self, balance_regression_factor: f64) -> PyResult<()> {
					set_gen!(self.shiob, balance_regression_factor => $prec_type)
				}
				#[getter]
				pub fn get_data<'py>(&self, py: Python<'py>) -> &'py PyArray2<$prec_type> {
					get_gen!(py self.shiob, data)
				}
				#[getter]
				pub fn get_centers<'py>(&self, py: Python<'py>) -> &'py PyArray2<$prec_type> {
					get_gen!(py self.shiob, centers)
				}
				#[getter]
				pub fn get_affine(&self) -> PyResult<bool> {
					get_gen!(self.shiob, affine)
				}
				#[getter]
				pub fn get_inversive(&self) -> PyResult<bool> {
					get_gen!(self.shiob, inversive)
				}
				#[getter]
				pub fn get_kernelized(&self) -> PyResult<bool> {
					get_gen!(self.shiob, kernelized)
				}
				#[getter]
				pub fn get_kernel_width(&self) -> PyResult<usize> {
					get_gen!(self.shiob, kernel_width)
				}
				#[getter]
				pub fn get_kernel_reshape(&self) -> PyResult<Vec<usize>> {
					get_gen!(self.shiob, kernel_reshape)
				}
				#[getter]
				pub fn get_inverter_scale(&self) -> PyResult<Option<$prec_type>> {
					get_gen!(self.shiob, inverter_scale)
				}
				#[getter]
				pub fn get_inverter_shift<'py>(&self, py: Python<'py>) -> PyResult<Option<&'py PyArray1<$prec_type>>> {
					Ok(self.shiob.get_inverter_shift().map(|v| v.to_pyarray(py)))
				}
				#[getter]
				pub fn get_center_biases<'py>(&self, py: Python<'py>) -> &'py PyArray1<$prec_type> {
					get_gen!(py self.shiob, center_biases)
				}
				pub fn set_center(&mut self, i_center: usize, center: PyReadonlyArray1<$prec_type>) {
					self.shiob.set_center(i_center, &center.as_array());
				}
				pub fn set_bias(&mut self, i_center: usize, bias: f64) {
					self.shiob.set_bias(i_center, <$prec_type as NumCast>::from(bias).unwrap());
				}
				pub fn set_center_bias(&mut self, i_center: usize, center: PyReadonlyArray1<$prec_type>, bias: f64) {
					self.shiob.set_center_bias(i_center, &center.as_array(), <$prec_type as NumCast>::from(bias).unwrap());
				}
				// #[getter]
				// pub fn get_data_bins<'py>(&self, py: Python<'py>) -> &'py PyArray2<$bin_type> {
				// 	self.shiob.get_data_bins().to_pyarray(py)
				// }
				#[getter]
				pub fn get_overlap_mat<'py>(&self, py: Python<'py>) -> &'py PyArray2<usize> {
					get_gen!(py self.shiob, overlap_mat)
				}
				#[getter]
				pub fn get_sim_mat<'py>(&self, py: Python<'py>) -> &'py PyArray2<f64> {
					get_gen!(py self.shiob, sim_mat)
				}
				#[getter]
				pub fn get_sim_sums<'py>(&self, py: Python<'py>) -> &'py PyArray1<f64> {
					get_gen!(py self.shiob, sim_sums)
				}
				#[getter]
				pub fn get_update_parallel(&self) -> PyResult<bool> {
					get_gen!(self.shiob, update_parallel)
				}
				#[setter]
				pub fn set_update_parallel(&mut self, update_parallel: bool) -> PyResult<()> {
					set_gen!(self.shiob, update_parallel)
				}
				#[getter]
				pub fn get_displace_parallel(&self) -> PyResult<bool> {
					get_gen!(self.shiob, displace_parallel)
				}
				#[setter]
				pub fn set_displace_parallel(&mut self, displace_parallel: bool) -> PyResult<()> {
					set_gen!(self.shiob, displace_parallel)
				}
				#[getter]
				pub fn get_noise_std(&self) -> PyResult<Option<f64>> {
					get_gen!(self.shiob, noise_std O=> f64)
				}
				#[setter]
				pub fn set_noise_std(&mut self, noise_std: Option<f64>) -> PyResult<()> {
					set_gen!(self.shiob, noise_std O=> $prec_type)
				}
			}
		}
	}
}
// stochastic_hiob_struct_gen!(H5, (f32, f64), (bool, i8, i16, i32, i64, u8, u16, u32, u64));
// stochastic_hiob_struct_gen!(ND, (f32, f64), (bool, i8, i16, i32, i64, u8, u16, u32, u64));
stochastic_hiob_struct_gen!(H5, (f32, f64), (bool, u8, u16, u32, u64));
stochastic_hiob_struct_gen!(ND, (f32, f64), (bool, u8, u16, u32, u64));
#[cfg(feature="half")]
stochastic_hiob_struct_gen!(H5, f16, (bool, u8, u16, u32, u64));
#[cfg(feature="half")]
stochastic_hiob_struct_gen!(ND, f16, (bool, u8, u16, u32, u64));
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
					data: PyReadonlyArray2<$prec_type>,
					queries: PyReadonlyArray2<$prec_type>,
					hamming_neighbors: PyReadonlyArray2<usize>,
					k: usize
				) -> PyResult<f64> {
					Ok(
						self.bin_eval.k_at_n_recall_prec_hamming_neighbors(
							&data.as_array(),
							&queries.as_array(),
							&hamming_neighbors.as_array(),
							k
						)
					)
				}
				pub fn [<refine_ $prec_type>]<'py>(
					&self,
					py: Python<'py>,
					data: PyReadonlyArray2<$prec_type>,
					queries: PyReadonlyArray2<$prec_type>,
					hamming_neighbors: PyReadonlyArray2<usize>,
					k: usize,
					chunk_size: Option<usize>,
				) -> (&'py PyArray2<$prec_type>, &'py PyArray2<usize>) {
					let (dots, idxs) = self.bin_eval.refine(
						&data.as_array(),
						&queries.as_array(),
						&hamming_neighbors.as_array(),
						k,
						chunk_size,
					);
					(dots.to_pyarray(py), idxs.to_pyarray(py))
				}
				pub fn [<refine_h5_ $prec_type>]<'py>(
					&self,
					py: Python<'py>,
					data_file: String,
					data_dataset: String,
					queries: PyReadonlyArray2<$prec_type>,
					hamming_neighbors: PyReadonlyArray2<usize>,
					k: usize,
					chunk_size: Option<usize>,
					io_chunk_size: Option<usize>,
				) -> (&'py PyArray2<$prec_type>, &'py PyArray2<usize>) {
					let (dots, idxs) = self.bin_eval.refine_h5(
						data_file.as_str(),
						data_dataset.as_str(),
						&queries.as_array(),
						&hamming_neighbors.as_array(),
						k,
						chunk_size,
						io_chunk_size,
					);
					(dots.to_pyarray(py), idxs.to_pyarray(py))
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
					k: usize,
					chunk_size: Option<usize>
				) -> (&'py PyArray2<usize>, &'py PyArray2<usize>) {
					let (dists, idxs) = self.bin_eval.brute_force_k_smallest_hamming(
						&data_bin.as_array(),
						&queries_bin.as_array(),
						k,
						chunk_size
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
				pub fn [<refine_with_other_bin_ $bin_type>]<'py>(
					&self,
					py: Python<'py>,
					data_bin: PyReadonlyArray2<$bin_type>,
					queries_bin: PyReadonlyArray2<$bin_type>,
					hamming_ids: PyReadonlyArray2<usize>,
					k: usize,
					chunk_size: Option<usize>
				) -> (&'py PyArray2<usize>, &'py PyArray2<usize>) {
					let (dists, idxs) = self.bin_eval.refine_with_other_bin(
						&data_bin.as_array(),
						&queries_bin.as_array(),
						&hamming_ids.as_array(),
						k,
						chunk_size
					);
					(dists.to_pyarray(py), idxs.to_pyarray(py))
				}
				pub fn [<cascading_k_smallest_hamming_ $bin_type>]<'py>(
					&self,
					py: Python<'py>,
					data_bins: Vec<PyReadonlyArray2<$bin_type>>,
					queries_bins: Vec<PyReadonlyArray2<$bin_type>>,
					ks: Vec<usize>,
					chunk_size: Option<usize>
				) -> (&'py PyArray2<usize>, &'py PyArray2<usize>) {
					let data_bins: Vec<_> = (0..data_bins.len()).map(|i| data_bins[i].as_array()).collect();
					let queries_bins: Vec<_> = (0..queries_bins.len()).map(|i| queries_bins[i].as_array()).collect();
					let (dists, idxs) = self.bin_eval.cascading_k_smallest_hamming(
						&data_bins,
						&queries_bins,
						&ks,
						chunk_size
					);
					(dists.to_pyarray(py), idxs.to_pyarray(py))
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
				pub fn [<query_ $prec_type _ $bin_type>]<'py>(
					&self,
					py: Python<'py>,
					data: PyReadonlyArray2<$prec_type>,
					data_bin: PyReadonlyArray2<$bin_type>,
					queries: PyReadonlyArray2<$prec_type>,
					queries_bin: PyReadonlyArray2<$bin_type>,
					k: usize,
					n: usize,
					chunk_size: Option<usize>,
				) -> (&'py PyArray2<$prec_type>, &'py PyArray2<usize>) {
					let (dots, idx) = self.bin_eval.query(
						&data.as_array(),
						&data_bin.as_array(),
						&queries.as_array(),
						&queries_bin.as_array(),
						k,
						n,
						chunk_size,
					);
					(dots.to_pyarray(py), idx.to_pyarray(py))
				}
				pub fn [<query_cascade_ $prec_type _ $bin_type>]<'py>(
					&self,
					py: Python<'py>,
					data: PyReadonlyArray2<$prec_type>,
					data_bins: Vec<PyReadonlyArray2<$bin_type>>,
					queries: PyReadonlyArray2<$prec_type>,
					queries_bins: Vec<PyReadonlyArray2<$bin_type>>,
					k: usize,
					ns: Vec<usize>,
					chunk_size: Option<usize>,
				) -> (&'py PyArray2<$prec_type>, &'py PyArray2<usize>) {
					let data_bins: Vec<_> = (0..data_bins.len()).map(|i| data_bins[i].as_array()).collect();
					let queries_bins: Vec<_> = (0..queries_bins.len()).map(|i| queries_bins[i].as_array()).collect();
					let (dots, idx) = self.bin_eval.query_cascade(
						&data.as_array(),
						&data_bins,
						&queries.as_array(),
						&queries_bins,
						k,
						&ns,
						chunk_size,
					);
					(dots.to_pyarray(py), idx.to_pyarray(py))
				}
				pub fn [<query_h5_ $prec_type _ $bin_type>]<'py>(
					&self,
					py: Python<'py>,
					file: String,
					dataset: String,
					data_bin: PyReadonlyArray2<$bin_type>,
					queries: PyReadonlyArray2<$prec_type>,
					queries_bin: PyReadonlyArray2<$bin_type>,
					k: usize,
					n: usize,
					chunk_size: Option<usize>,
				) -> (&'py PyArray2<$prec_type>, &'py PyArray2<usize>) {
					let (dots, idx) = self.bin_eval.query_h5(
						file.as_str(),
						dataset.as_str(),
						&data_bin.as_array(),
						&queries.as_array(),
						&queries_bin.as_array(),
						k,
						n,
						chunk_size,
					);
					(dots.to_pyarray(py), idx.to_pyarray(py))
				}
				pub fn [<query_cascade_h5_ $prec_type _ $bin_type>]<'py>(
					&self,
					py: Python<'py>,
					file: String,
					dataset: String,
					data_bins: Vec<PyReadonlyArray2<$bin_type>>,
					queries: PyReadonlyArray2<$prec_type>,
					queries_bins: Vec<PyReadonlyArray2<$bin_type>>,
					k: usize,
					ns: Vec<usize>,
					chunk_size: Option<usize>,
				) -> (&'py PyArray2<$prec_type>, &'py PyArray2<usize>) {
					let data_bins: Vec<_> = (0..data_bins.len()).map(|i| data_bins[i].as_array()).collect();
					let queries_bins: Vec<_> = (0..queries_bins.len()).map(|i| queries_bins[i].as_array()).collect();
					let (dots, idx) = self.bin_eval.query_cascade_h5(
						file.as_str(),
						dataset.as_str(),
						&data_bins,
						&queries.as_array(),
						&queries_bins,
						k,
						&ns,
						chunk_size,
					);
					(dots.to_pyarray(py), idx.to_pyarray(py))
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
#[cfg(feature="half")]
eval_fun_gen_p!(f16);
// eval_fun_gen_b!(bool, i8, i16, i32, i64, u8, u16, u32, u64);
eval_fun_gen_b!(bool, u8, u16, u32, u64);
// eval_fun_gen_pb!((f32, f64), (bool, i8, i16, i32, i64, u8, u16, u32, u64));
eval_fun_gen_pb!((f32, f64), (bool, u8, u16, u32, u64));
#[cfg(feature="half")]
eval_fun_gen_pb!(f16, (bool, u8, u16, u32, u64));


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



macro_rules! searcher_struct_gen {
	(($($bts:ty),*), $rts:tt) => {
		$(searcher_struct_gen!($bts, $rts);)*
	};
	($bin_type: ty, ($($rts:ty),*)) => {
		$(searcher_struct_gen!($bin_type, $rts);)*
	};
	($bin_type: ty, $ref_type: ty) => {
		paste! {
			#[allow(non_camel_case_types)]
			#[pyclass]
			pub struct [<RawMinHashSearcher_ $bin_type _ $ref_type>] {
				searcher: MinHashSearcher<'static,$bin_type,$ref_type>,
				bin_data: Box<Array2<$bin_type>>,
			}
			#[pymethods]
			impl [<RawMinHashSearcher_ $bin_type _ $ref_type>] {
				#[new]
				pub fn new(data: PyReadonlyArray2<$bin_type>, n_hashes: usize, n_positions: usize) -> Self {
					let bin_data = Box::new(data.as_array().into_owned());
					let data_ptr = &*bin_data as *const Array2<$bin_type>;
					let data_ref = unsafe{&*data_ptr};
					let searcher: MinHashSearcher<'static,$bin_type,$ref_type> = MinHashSearcher::new(data_ref, n_hashes, n_positions);
					Self{searcher, bin_data}
				}
				pub fn query<'py>(&self, py: Python<'py>, query: PyReadonlyArray2<'py, $bin_type>, n_neighbors: usize, chunk_size: Option<usize>) -> (&'py PyArray2<usize>, &'py PyArray2<usize>) {
					let (a,b) = self.searcher.query(&query.as_array().into_owned(), n_neighbors, chunk_size);
					(a.to_pyarray(py), b.to_pyarray(py))
				}
				#[classmethod]
				pub fn expected_size(_pytype: &PyType, n_data: usize, n_hashes: usize, n_pos: usize) -> usize {
					MinHashSearcher::<'_,$bin_type,$ref_type>::expected_size(n_data, n_hashes, n_pos)
				}
			}
			#[allow(non_camel_case_types)]
			#[pyclass]
			pub struct [<RawChunkyMinHashSearcher_ $bin_type _ $ref_type>] {
				searcher: ChunkyMinHashSearcher<'static,$bin_type,$ref_type>,
				bin_data: Box<Array2<$bin_type>>,
			}
			#[pymethods]
			impl [<RawChunkyMinHashSearcher_ $bin_type _ $ref_type>] {
				#[new]
				pub fn new(data: PyReadonlyArray2<$bin_type>, n_hashes: usize, n_positions: usize) -> Self {
					let bin_data = Box::new(data.as_array().into_owned());
					let data_ptr = &*bin_data as *const Array2<$bin_type>;
					let data_ref = unsafe{&*data_ptr};
					let searcher: ChunkyMinHashSearcher<'static,$bin_type,$ref_type> = ChunkyMinHashSearcher::new(data_ref, n_hashes, n_positions);
					Self{searcher, bin_data}
				}
				pub fn query<'py>(&self, py: Python<'py>, query: PyReadonlyArray2<'py, $bin_type>, n_neighbors: usize, chunk_size: Option<usize>) -> (&'py PyArray2<usize>, &'py PyArray2<usize>) {
					let (a,b) = self.searcher.query(&query.as_array().into_owned(), n_neighbors, chunk_size);
					(a.to_pyarray(py), b.to_pyarray(py))
				}
				#[classmethod]
				pub fn expected_size(_pytype: &PyType, n_data: usize, n_hashes: usize, n_pos: usize) -> usize {
					ChunkyMinHashSearcher::<'_,$bin_type,$ref_type>::expected_size(n_data, n_hashes, n_pos)
				}
			}
		}
	};
}
searcher_struct_gen!((bool, u8, u16, u32, u64), (u8, u16, u32, u64));
macro_rules! searcher_python_export {
	($module: ident, ($($bts:ty),*), $rts:tt) => {
		$(searcher_python_export!($module, $bts, $rts);)*
	};
	($module: ident, $bin_type: ty, ($($rts:ty),*)) => {
		$(searcher_python_export!($module, $bin_type, $rts);)*
	};
	($module: ident, $bin_type: ty, $ref_type: ty) => {
		paste!{
			$module.add_class::<[<RawMinHashSearcher_ $bin_type _ $ref_type>]>()?;
			$module.add_class::<[<RawChunkyMinHashSearcher_ $bin_type _ $ref_type>]>()?;
		}
	};
}



#[pyfunction]
pub fn limit_threads(_num_threads: usize) -> Result<(), PyErr> {
	let result = crate::limit_threads(_num_threads);
	if result.is_ok() {
		Ok(())
	} else {
		Err(PyErr::new::<PyValueError,_>(result.err().unwrap().to_string()))
	}
}

#[pyfunction]
pub fn num_threads() -> PyResult<usize> {
	Ok(crate::num_threads())
}

#[pyfunction]
pub fn supports_f16() -> PyResult<bool> {
	Ok(crate::supports_f16())
}


/* Declaration of the python package generated by maturin. */
#[pymodule]
fn hiob(_py: Python, m: &PyModule) -> PyResult<()> {
	// hiob_python_export!(m, (f32, f64), (bool, i8, i16, i32, i64, u8, u16, u32, u64));
	// stochastic_hiob_python_export!(m, (f32, f64), (bool, i8, i16, i32, i64, u8, u16, u32, u64));
	// thx_python_export!(m, (bool, i8, i16, i32, i64, u8, u16, u32, u64), (2,3,4,5,6,7,8,9,10));
	hiob_python_export!(m, (f32, f64), (bool, u8, u16, u32, u64));
	#[cfg(feature="half")]
	hiob_python_export!(m, f16, (bool, u8, u16, u32, u64));
	stochastic_hiob_python_export!(m, (f32, f64), (bool, u8, u16, u32, u64));
	#[cfg(feature="half")]
	stochastic_hiob_python_export!(m, f16, (bool, u8, u16, u32, u64));
	thx_python_export!(m, (bool, u8, u16, u32, u64), (2,3,4,5,6,7,8,9,10));
	searcher_python_export!(m, (bool, u8, u16, u32, u64), (u8, u16, u32, u64));
	m.add_class::<RawBinarizationEvaluator>()?;
	m.add_wrapped(wrap_pyfunction!(limit_threads))?;
	m.add_wrapped(wrap_pyfunction!(num_threads))?;
	m.add_wrapped(wrap_pyfunction!(supports_f16))?;
	Ok(())
}
