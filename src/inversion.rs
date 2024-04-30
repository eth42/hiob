use ndarray::{Array1, Array2, Axis, Slice, Data, ArrayBase, Ix1, Ix2};
use crate::types::HIOBFloat;
use crate::float_vectors::{DotProduct, InnerProduct};
use crate::vec_math::{vec_norm, vec_norms, vec_sq_norms};
use paste::paste;

/* Create logarithmically spaced grid */
fn _log_grid<F: HIOBFloat>(start: F, end: F, n: usize) -> Vec<F> {
	let log_start = num::Float::ln(start);
	let log_end = num::Float::ln(end);
	let log_delta = log_end - log_start;
	(0..n).map(|i| {
		let f = F::from(i).unwrap() / F::from(n-1).unwrap();
		num::Float::exp(log_start + log_delta * f)
	}).collect()
}

/* RABID local intrinsic dimensionality estimator for POI-centered vectors */
fn _rabid<F: HIOBFloat>(vecs: &Array2<F>, is_normalized: bool) -> F {
	let k = vecs.dim().0;
	/* Precompute vector norms if vectors are not normalized */
	let norms: Vec<F> = if !is_normalized { unsafe{vec_norms(vecs)} } else { vec![F::one(); k] };
	/* Compute sum of squared cosines over lower triangular of cosine matrix */
	let sq_cosine_sum = vecs.axis_iter(Axis(0)).enumerate()
	.map(|(i, vec_i)| {
			vecs.slice_axis(Axis(0), Slice::from(..i))
			.axis_iter(Axis(0)).enumerate()
			.map(|(j, vec_j)| {
				let cos = DotProduct::prod_arrs(&vec_i, &vec_j) / (norms[i] * norms[j]);
				cos * cos
			}).sum::<F>()
	}).sum::<F>() * F::from(2).unwrap();
	/* Evaluate RABID (reciprocal of mean squared cosine) */
	F::from(k*k-k).unwrap() / sq_cosine_sum
}

/* Estimate an appropriate scale value for the spherical inversion */
pub fn initial_inversion_scale_guess<F: HIOBFloat>(vecs: &Array2<F>) -> F {
	unsafe{vec_norms(vecs)}.into_iter().sum::<F>() / F::from(vecs.dim().0).unwrap()
}
pub fn logarithmic_grid_inversion_scale_guess<F: HIOBFloat>(vecs: &Array2<F>, grid_width: F, grid_size: usize) -> F {
	let initial_s = initial_inversion_scale_guess(vecs);
	let grid = _log_grid(initial_s / grid_width, initial_s * grid_width, grid_size);
	let rabids = grid.iter().map(|&s|
		_rabid(&spherical_inversion(vecs, s), true)
	).collect::<Vec<F>>();
	rabids.into_iter().zip(grid.into_iter()).reduce(|(rabid_i, s_i), (rabid_j, s_j)|
		if rabid_j > rabid_i { (rabid_j, s_j) } else { (rabid_i, s_i) }
	).unwrap().1
}

/* Perform spherical inversion */
pub fn spherical_inversion<F: HIOBFloat, D: Data<Elem=F>>(vecs: &ArrayBase<D, Ix2>, scale: F) -> Array2<F> {
	let (k,d) = vecs.dim();
	let two = F::from(2).unwrap();
	let mut inversed = Array2::zeros((k, d+1));
	unsafe{vec_sq_norms(vecs)}.into_iter().zip(
		vecs.axis_iter(Axis(0))
		.zip(inversed.axis_iter_mut(Axis(0)))
	)
	.map(|(sq_norm, (vec, mut inv))| {
		let factor = sq_norm + scale*scale;
		inv.slice_axis_mut(Axis(0), Slice::from(..d)).assign(&(
			vec.iter().map(|&v| two*scale*v/factor))
			.collect::<Array1<F>>()
		);
		inv[d] = two*scale*scale/factor - F::one();
	}).count();
	inversed
}
pub fn spherical_inversion_single<F: HIOBFloat, D: Data<Elem=F>>(vecs: &ArrayBase<D, Ix1>, scale: F) -> Array1<F> {
	spherical_inversion(&vecs.to_shape((1,vecs.dim())).unwrap(), scale).into_shape(vecs.dim()+1).unwrap()
}
pub fn spherical_uninversion<F: HIOBFloat, D: Data<Elem=F>>(vecs: &ArrayBase<D, Ix2>, scale: F) -> Array2<F> {
	let (n_vecs, n_dims) = vecs.dim();
	let mut result = Array2::zeros((n_vecs, n_dims-1));
	vecs.slice_axis(Axis(1), Slice::from(..n_dims-1)).axis_iter(Axis(0))
	.zip(vecs.column(n_dims-1).iter())
	.map(|(vec, &bias)| vec.mapv(|v| v*scale/(F::one()+bias)))
	.enumerate()
	.for_each(|(i, row)| result.row_mut(i).assign(&row));
	result
}
pub fn spherical_uninversion_single<F: HIOBFloat, D: Data<Elem=F>>(vecs: &ArrayBase<D, Ix1>, scale: F) -> Array1<F> {
	spherical_uninversion(&vecs.to_shape((1,vecs.dim())).unwrap(), scale).into_shape(vecs.dim()-1).unwrap()
}

/* Translation between caps and balls */
pub fn cap_to_ball<F: HIOBFloat, D: Data<Elem=F>>(normal: &ArrayBase<D, Ix1>, bias: F, scale: F) -> (Array1<F>, F) {
	let d = normal.dim()-1; /* Dimension of the non-spherical space */
	let (normal, bias) = if normal[d] < F::zero() { (normal.mapv(|v| -v), -bias) } else { (normal.to_owned(), bias) };
	let two = F::from(2).unwrap();
	let alpha = (F::one()-bias*bias)/(two*(bias+normal[d]));
	let sq_radius = scale*scale*two*alpha/(bias+normal[d]);
	let mut center = normal.to_owned();
	center[d] = center[d] - alpha;
	let norm = unsafe{vec_norm(&center)};
	let center = center.mapv_into(|v| v/norm);
	let center = spherical_uninversion_single(&center, scale);
	(center, sq_radius)
}


pub struct SphericalInverter<F: HIOBFloat> {
	pub scale: F,
	pub shift: Option<Array1<F>>,
}
crate::types::param_struct!(SphericalInverterParams<F: HIOBFloat> {
	mean_center: bool = true,
	init_grid: bool = false,
	grid_width: F = F::from(5.).unwrap(),
	grid_size: usize = 10,
});
impl<F: HIOBFloat> SphericalInverter<F> {
	pub fn new(init_sample: &Array2<F>, params: SphericalInverterParams<F>) -> Self {
		let k = init_sample.dim().0;
		let (shift, local_init_sample) = if params.mean_center {
			let mean = init_sample.sum_axis(Axis(0)).mapv_into(|v| v/F::from(k).unwrap());
			let shifted_sample = init_sample - &mean;
			(Some(mean), shifted_sample)
		} else {
			(None, init_sample.clone())
		};
		let scale = if params.init_grid {
			logarithmic_grid_inversion_scale_guess(&local_init_sample, params.grid_width, params.grid_size)
		} else {
			initial_inversion_scale_guess(&local_init_sample)
		};
		SphericalInverter {
			scale,
			shift,
		}
	}
	pub fn invert(&self, vecs: &Array2<F>) -> Array2<F> {
		if self.shift.is_some() {
			let shifted_vecs = vecs - self.shift.as_ref().unwrap();
			spherical_inversion(&shifted_vecs, self.scale)
		} else {
			spherical_inversion(vecs, self.scale)
		}
	}
}


#[test]
fn rabid_test() {
	use ndarray_rand::RandomExt;
	use ndarray_rand::rand_distr::StandardNormal;
	let vecs: Array2<f32> = Array2::random((500, 10), StandardNormal{});
	let rabid = _rabid(&vecs, false);
	/* 500 points should always be enough to estimate 10 dimensions
	 * with an accuracy of +- 0.2. */
	assert!((rabid-10.).abs() < 0.2, "rabid: {}", rabid);
}

#[test]
fn inversion_scale_guess_test() {
	use ndarray_rand::RandomExt;
	use ndarray_rand::rand_distr::StandardNormal;
	let vecs: Array2<f32> = Array2::random((500, 10), StandardNormal{});
	/* Ensure that the RABID estimate of the logarithmic grid search is better than the initial guess */
	let scale = initial_inversion_scale_guess(&vecs);
	let scale_rabid = _rabid(&spherical_inversion(&vecs, scale), true);
	assert!(scale_rabid <= 11., "scale: {}, rabid: {}", scale, scale_rabid);
	assert!(scale_rabid > 5., "scale: {}, rabid: {}", scale, scale_rabid);
	let scale_log = logarithmic_grid_inversion_scale_guess(&vecs, 5., 10);
	let scale_log_rabid = _rabid(&spherical_inversion(&vecs, scale_log), true);
	assert!(scale_log_rabid > scale_rabid, "scale_log: {}, rabid: {}; scale: {}, rabid: {}", scale_log, scale_log_rabid, scale, scale_rabid);
	// println!("scale_log: {}, rabid: {}; scale: {}, rabid: {}", scale_log, scale_log_rabid, scale, scale_rabid);
}

#[test]
fn spherical_inversion_test() {
	use ndarray_rand::RandomExt;
	use ndarray_rand::rand_distr::StandardNormal;
	let vecs: Array2<f32> = Array2::random((500, 10), StandardNormal{});
	let scale = initial_inversion_scale_guess(&vecs);
	/* Ensure that all vectors are approximately unit length */
	let inversed = spherical_inversion(&vecs, scale);
	let norms = unsafe {vec_norms(&inversed)};
	for norm in norms.iter() {
		assert!((norm-1.).abs() < 0.01, "norm: {}", norm);
	}
	/* Ensure that inversion forward and backward yields the same vectors */
	let uninversed = spherical_uninversion(&inversed, scale);
	vecs.axis_iter(Axis(0)).zip(uninversed.axis_iter(Axis(0)))
	.for_each(|(vec, uninv)| {
		vec.iter().zip(uninv.iter())
		.for_each(|(v, u)| {
			assert!((v-u).abs() < 1e-6, "v: {}, u: {}", v, u);
		});
	});
}

#[test]
fn spherical_inverter_test() {
	use ndarray_rand::RandomExt;
	use ndarray_rand::rand_distr::StandardNormal;
	let vecs: Array2<f32> = Array2::random((500, 10), StandardNormal{});
	let inverter = SphericalInverter::new(&vecs, SphericalInverterParams::new());
	let inversed = inverter.invert(&vecs);
	assert_eq!(vecs.shape()[0], inversed.shape()[0]);
	assert_eq!(vecs.shape()[1], inversed.shape()[1]-1);
}
