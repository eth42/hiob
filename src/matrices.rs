/* This file contains matrix wrappers to efficiently compute
 * the maximum and minimum values of a symmetric matrix using
 * Hamerly's heuristic for k-means. */
use ndarray::{Array1, Array2, ArrayView2};
use num::Float;



pub struct SymArgmaxMatrix<F: Float> {
	pub matrix: Array2<F>,
	row_max: Array1<F>,
	row_max_idx: Array1<usize>,
}
impl<F: Float> SymArgmaxMatrix<F> {
	pub fn new(matrix: Array2<F>) -> Self {
		let d = matrix.dim().0;
		let mut ret = SymArgmaxMatrix {
			matrix,
			row_max: Array1::from_elem(d, -F::infinity()),
			row_max_idx: Array1::zeros(d),
		};
		for i in 0..d {
			ret.update_row_max(i);
		}
		ret
	}
	pub fn get_row_max(&self, i: usize) -> F {
		self.row_max[i]
	}
	pub fn get_max(&self) -> F {
		self.row_max.iter().cloned().fold(-F::infinity(), F::max)
	}
	pub fn get_row_argmax(&self, i: usize) -> usize {
		self.row_max_idx[i]
	}
	pub fn get_argmax(&self) -> (usize, usize) {
		let mut max = -F::infinity();
		let mut max_idx = (0, 0);
		for (i, &v) in self.row_max.iter().enumerate() {
			if v > max {
				max = v;
				max_idx = (i, self.row_max_idx[i]);
			}
		}
		if max_idx.0 > max_idx.1 {
			(max_idx.1, max_idx.0)
		} else {
			max_idx
		}
	}
	pub fn update_row(&mut self, i: usize, new_row: Array1<F>) {
		/* Update row i */
		self.matrix.row_mut(i).assign(&new_row);
		let mut max = -F::infinity();
		let mut max_idx = 0;
		for (j, &v) in new_row.iter().enumerate() {
			if v > max {
				max = v;
				max_idx = j;
			}
		}
		/* Update column i */
		self.row_max[i] = max;
		self.row_max_idx[i] = max_idx;
		for (j, &v) in new_row.iter().enumerate() {
			if j != i {
				self.update_value(j, i, v);
			}
		}
	}
	pub fn update_value_sym(&mut self, i: usize, j: usize, val: F) {
		self.update_value(i, j, val);
		self.update_value(j, i, val);
	}
	fn update_value(&mut self, i: usize, j: usize, val: F) {
		self.matrix[[i, j]] = val;
		if val > self.row_max[i] {
			self.row_max[i] = val;
			self.row_max_idx[i] = j;
		} else if j == self.row_max_idx[i] {
			self.update_row_max(i);
		}
		if val > self.row_max[j] {
			self.row_max[j] = val;
			self.row_max_idx[j] = i;
		} else if i == self.row_max_idx[j] {
			self.update_row_max(j);
		}
	}
	fn update_row_max(&mut self, i: usize) {
		let mut max = -F::infinity();
		let mut max_idx = 0;
		for (j, &v) in self.matrix.row(i).iter().enumerate() {
			if v > max {
				max = v;
				max_idx = j;
			}
		}
		self.row_max[i] = max;
		self.row_max_idx[i] = max_idx;
	}
	pub fn view<'a>(&'a self) -> ArrayView2<'a, F> {
		self.matrix.view()
	}
}


pub struct SymArgminMatrix<F: Float> {
	pub matrix: Array2<F>,
	row_min: Array1<F>,
	row_min_idx: Array1<usize>,
}
impl<F: Float> SymArgminMatrix<F> {
	pub fn new(matrix: Array2<F>) -> Self {
		let d = matrix.dim().0;
		let mut ret = SymArgminMatrix {
			matrix,
			row_min: Array1::from_elem(d, F::infinity()),
			row_min_idx: Array1::zeros(d),
		};
		for i in 0..d {
			ret.update_row_min(i);
		}
		ret
	}
	pub fn get_row_min(&self, i: usize) -> F {
		self.row_min[i]
	}
	pub fn get_min(&self) -> F {
		self.row_min.iter().cloned().fold(F::infinity(), F::min)
	}
	pub fn get_row_argmin(&self, i: usize) -> usize {
		self.row_min_idx[i]
	}
	pub fn get_argmin(&self) -> (usize, usize) {
		let mut min = F::infinity();
		let mut min_idx = (0, 0);
		for (i, &v) in self.row_min.iter().enumerate() {
			if v < min {
				min = v;
				min_idx = (i, self.row_min_idx[i]);
			}
		}
		if min_idx.0 > min_idx.1 {
			(min_idx.1, min_idx.0)
		} else {
			min_idx
		}
	}
	pub fn update_row(&mut self, i: usize, new_row: Array1<F>) {
		/* Update row i */
		self.matrix.row_mut(i).assign(&new_row);
		let mut min = F::infinity();
		let mut min_idx = 0;
		for (j, &v) in new_row.iter().enumerate() {
			if v < min {
				min = v;
				min_idx = j;
			}
		}
		/* Update column i */
		self.row_min[i] = min;
		self.row_min_idx[i] = min_idx;
		for (j, &v) in new_row.iter().enumerate() {
			if j != i {
				self.update_value(j, i, v);
			}
		}
	}
	pub fn update_value_sym(&mut self, i: usize, j: usize, val: F) {
		self.update_value(i, j, val);
		self.update_value(j, i, val);
	}
	fn update_value(&mut self, i: usize, j: usize, val: F) {
		self.matrix[[i, j]] = val;
		if val < self.row_min[i] {
			self.row_min[i] = val;
			self.row_min_idx[i] = j;
		} else if j == self.row_min_idx[i] {
			self.update_row_min(i);
		}
		if val < self.row_min[j] {
			self.row_min[j] = val;
			self.row_min_idx[j] = i;
		} else if i == self.row_min_idx[j] {
			self.update_row_min(j);
		}
	}
	fn update_row_min(&mut self, i: usize) {
		let mut min = F::infinity();
		let mut min_idx = 0;
		for (j, &v) in self.matrix.row(i).iter().enumerate() {
			if v < min {
				min = v;
				min_idx = j;
			}
		}
		self.row_min[i] = min;
		self.row_min_idx[i] = min_idx;
	}
	pub fn view<'a>(&'a self) -> ArrayView2<'a, F> {
		self.matrix.view()
	}
}

#[test]
fn test_sym_argmax_matrix() {
	use ndarray_rand::RandomExt;
	use ndarray_rand::rand_distr::StandardNormal;
	let d = 20;
	let n_changes = 100;
	/* Initialize random matrix */
	let mat: Array2<f32> = Array2::random((d, d), StandardNormal);
	let mut m = SymArgmaxMatrix::new(mat.clone() + mat.t());
	for _ in 0..n_changes {
		/* Update a row */
		let i = rand::random::<usize>() % d;
		let new_row: Array1<f32> = Array1::random(d, StandardNormal);
		m.update_row(i, new_row);
		/* Check that the matrix is still symmetric */
		for i in 0..d {
			for j in 0..i+1 {
				assert_eq!(m.matrix[[i, j]], m.matrix[[j, i]]);
			}
		}
		/* Check that the maximum is correct */
		let true_max = m.matrix.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
		let max = m.get_max();
		assert_eq!(true_max, max);
		let (i, j) = m.get_argmax();
		assert_eq!(max, m.matrix[[i, j]]);
	}
}
#[test]
fn test_sym_argmin_matrix() {
	use ndarray_rand::RandomExt;
	use ndarray_rand::rand_distr::StandardNormal;
	let d = 20;
	let n_changes = 100;
	/* Initialize random matrix */
	let mat: Array2<f32> = Array2::random((d, d), StandardNormal);
	let mut m = SymArgminMatrix::new(mat.clone() + mat.t());
	for _ in 0..n_changes {
		/* Update a row */
		let i = rand::random::<usize>() % d;
		let new_row: Array1<f32> = Array1::random(d, StandardNormal);
		m.update_row(i, new_row);
		/* Check that the matrix is still symmetric */
		for i in 0..d {
			for j in 0..i+1 {
				assert_eq!(m.matrix[[i, j]], m.matrix[[j, i]]);
			}
		}
		/* Check that the minimum is correct */
		let true_min = m.matrix.iter().cloned().fold(f32::INFINITY, f32::min);
		let min = m.get_min();
		assert_eq!(true_min, min);
		let (i, j) = m.get_argmin();
		assert_eq!(min, m.matrix[[i, j]]);
	}
}

