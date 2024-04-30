use ndarray::{ArrayBase, Data, Ix1, Ix2, Axis};
use crate::types::HIOBFloat;


#[inline(always)]
pub unsafe fn vec_norm<F: HIOBFloat, D: Data<Elem=F>>(vec: &ArrayBase<D, Ix1>) -> F {
	num::Float::sqrt(vec.iter().map(|&v| v*v).reduce(|a,b| a+b).unwrap_unchecked())
}
#[inline(always)]
pub unsafe fn vec_norms<F: HIOBFloat, D: Data<Elem=F>>(vecs: &ArrayBase<D, Ix2>) -> Vec<F> {
	vecs.axis_iter(Axis(0)).map(|row| vec_norm(&row)).collect()
}
#[inline(always)]
pub unsafe fn vec_sq_norm<F: HIOBFloat, D: Data<Elem=F>>(vec: &ArrayBase<D, Ix1>) -> F {
	vec.iter().map(|&v| v*v).reduce(|a,b| a+b).unwrap_unchecked()
}
#[inline(always)]
pub unsafe fn vec_sq_norms<F: HIOBFloat, D: Data<Elem=F>>(vecs: &ArrayBase<D, Ix2>) -> Vec<F> {
	vecs.axis_iter(Axis(0)).map(|row| vec_sq_norm(&row)).collect()
}

