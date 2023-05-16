use ndarray::{Array1, Array2, s, Slice, Data, ArrayBase, Ix2, Axis};
use num::Zero;
#[cfg(feature="parallel")]
use rayon::iter::ParallelIterator;

use crate::progress::{par_iter, MaybeSend};

pub trait MatrixDataSource<T> {
	fn n_rows(&self) -> usize;
	fn n_cols(&self) -> usize;
	fn get_row(&self, i_row: usize) -> Array1<T>;
	fn get_rows(&self, i_rows: Vec<usize>) -> Array2<T>;
	fn get_rows_slice(&self, i_row_from: usize, i_row_to: usize) -> Array2<T>;
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
	fn get_rows(&self, i_rows: Vec<usize>) -> Array2<T> {
		Array2::from_shape_fn(
			(i_rows.len(), self.n_cols()),
			|(i,j)| self[[i_rows[i], j]]
		)
	}
	fn get_rows_slice(&self, i_row_from: usize, i_row_to: usize) -> Array2<T> {
		self.slice_axis(Axis(0), Slice::from(i_row_from..i_row_to.min(self.n_rows()))).to_owned()
	}
}

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


pub fn read_h5_dataset(file: &str, dataset: &str) -> Result<hdf5::Dataset, hdf5::Error> {
	let file = hdf5::File::open(file)?;
	file.dataset(dataset)
}
#[allow(unused)]
pub fn store_h5_dataset<T: hdf5::H5Type>(file: &str, dataset: &str, data: &Array2<T>) -> Result<(), hdf5::Error>{
	let file = hdf5::File::create(file)?;
	let dataset_builder = file.new_dataset_builder();
	dataset_builder.with_data(data).create(dataset)?;
	Ok(())
}
