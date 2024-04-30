use ndarray_linalg::{Lapack, Scalar};
use num::{Float, FromPrimitive};
use std::{iter::Sum, ops::AddAssign};

use crate::bits::Bits;
#[cfg(feature="python")]
use crate::pydata::NumpyEquivalent;
#[cfg(feature="rust-hdf5")]
use {hdf5::H5Type, num::Zero};



macro_rules! param_struct {
	/* Matching e.g. SomeParams[Debug, Clone]<F: Float> {a: F = F::one()} */
	(
		$name:ident /* Name of the parameter struct */
		$([$($derived_type:ty),*])? /* Derived types */
		$(<$($generic_names:ident : $generic_types:path)*>)? /* Generics */
		{$($field_name:ident: $field_type:ty = $field_value:expr),*$(,)?} /* Fields */
	) => { paste! {
		#[derive($($($derived_type,)*)?)]
		pub struct $name$(<$($generic_names: $generic_types),*>)? {
			$(pub $field_name: $field_type),*
		}
		impl$(<$($generic_names: $generic_types),*>)? $name$(<$($generic_names),*>)? {
			pub fn new() -> Self {
				Self {
					$($field_name: $field_value),*
				}
			}
			pub fn new_full($($field_name: Option<$field_type>,)*) -> Self {
				let mut ret = Self::new();
				$(
					if $field_name.is_some() { ret.$field_name = $field_name.unwrap(); }
				)*
				ret
			}
			$(
				pub fn [<with_ $field_name>](mut self, $field_name: $field_type) -> Self {
					self.$field_name = $field_name;
					self
				}
			)*
			$(
				pub fn [<maybe_with_ $field_name>](mut self, $field_name: Option<$field_type>) -> Self {
					if $field_name.is_some() {
						self = self.[<with_ $field_name>]($field_name.unwrap());
					}
					self
				}
			)*
		}
	}};
}
pub(crate) use param_struct;



macro_rules! trait_combiner {
	($combination_name: ident) => {
		pub trait $combination_name {}
		impl<T> $combination_name for T {}
	};
	($combination_name: ident: $t: tt $(+ $ts: tt)*) => {
		pub trait $combination_name: $t $(+ $ts)* {}
		impl<T: $t $(+ $ts)*> $combination_name for T {}
	};
}
#[cfg(feature="rust-hdf5")]
#[cfg(feature="python")]
trait_combiner!(HIOBFloat: Scalar+Lapack+FromPrimitive+CachingNumpyEquivalent+H5Type+Float+Sum+AddAssign+MaybeSend+MaybeSync);
#[cfg(feature="rust-hdf5")]
#[cfg(not(feature="python"))]
trait_combiner!(HIOBFloat: Scalar+Lapack+FromPrimitive+H5Type+Float+Sum+AddAssign+MaybeSend+MaybeSync);
#[cfg(not(feature="rust-hdf5"))]
#[cfg(feature="python")]
trait_combiner!(HIOBFloat: Scalar+Lapack+FromPrimitive+CachingNumpyEquivalent+Float+Sum+AddAssign+MaybeSend+MaybeSync);
#[cfg(not(feature="rust-hdf5"))]
#[cfg(not(feature="python"))]
trait_combiner!(HIOBFloat: Scalar+Lapack+FromPrimitive+Float+Sum+AddAssign+MaybeSend+MaybeSync);
trait_combiner!(HIOBBits: Bits+Clone+MaybeSend+MaybeSync);


#[cfg(feature="parallel")]
trait_combiner!(MaybeSync: Sync);
#[cfg(not(feature="parallel"))]
trait_combiner!(MaybeSync);
#[cfg(feature="parallel")]
trait_combiner!(MaybeSend: Send);
#[cfg(not(feature="parallel"))]
trait_combiner!(MaybeSend);


#[cfg(feature="rust-hdf5")]
trait_combiner!(CachingH5Type: H5Type+Zero+Copy+Clone+Send+'static);
#[cfg(feature="python")]
trait_combiner!(CachingNumpyEquivalent: NumpyEquivalent+'static);

