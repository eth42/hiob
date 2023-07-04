// #![allow(dead_code)]

mod bit_vectors;
mod float_vectors;
mod binarizer;
mod bits;
mod progress;
mod eval;
mod index;
mod heaps;
mod random;
mod data;

#[cfg(feature="python")]
mod pybridge;
#[cfg(feature="python")]
mod pydata;


#[allow(unreachable_code)]
pub fn limit_threads(_num_threads: usize) -> Result<(), Box<dyn std::error::Error>> {
	#[cfg(feature="parallel")]
	rayon::ThreadPoolBuilder::new().num_threads(_num_threads).build_global()?;
	#[cfg(not(feature="parallel"))]
	panic!("Number of threads could not be set, because this package was built without multi-threading.");
	Ok(())
}

pub fn num_threads() -> usize {
	#[cfg(feature="parallel")]
	let result = rayon::current_num_threads();
	#[cfg(not(feature="parallel"))]
	let result = 1;
	result
}

pub fn supports_f16() -> bool {
	#[cfg(feature="half")]
	let result = true;
	#[cfg(not(feature="half"))]
	let result = false;
	result
}