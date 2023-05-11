#[cfg(feature="progressbars")]
use indicatif::{ProgressBar, ProgressState, ProgressStyle};
#[cfg(feature="progressbars")]
use std::{fmt::Write};
#[cfg(feature="parallel")]
use rayon::prelude::*;
use std::iter::{ExactSizeIterator, Iterator};


macro_rules! trait_combiner {
	($combination_name: ident) => {
		pub trait $combination_name {}
		impl<T> $combination_name for T {}
	};
	($combination_name: ident: $t: ident $(+ $ts: ident)*) => {
		pub trait $combination_name: $t $(+ $ts)* {}
		impl<T: $t $(+ $ts)*> $combination_name for T {}
	};
}
#[cfg(feature="parallel")]
trait_combiner!(MaybeSync: Sync);
#[cfg(not(feature="parallel"))]
trait_combiner!(MaybeSync);
#[cfg(feature="parallel")]
trait_combiner!(MaybeSend: Send);
#[cfg(not(feature="parallel"))]
trait_combiner!(MaybeSend);


#[cfg(feature="progressbars")]
pub struct ProgressWrapper<I: ExactSizeIterator> {
	iter: I,
	bar: ProgressBar
}
#[cfg(feature="progressbars")]
impl<I: ExactSizeIterator> ProgressWrapper<I> {
	pub fn new(iter: I, msg: &str) -> Self {
		let bar = ProgressBar::new(iter.len() as u64);
    bar.set_style(
			ProgressStyle::with_template("{msg}: [{wide_bar:.cyan/blue}] {pos}/{len} ({elapsed_precise}/{duration_precise})")
			.unwrap()
			.with_key("eta", |state: &ProgressState, w: &mut dyn Write| write!(w, "{:.1}s", state.eta().as_secs_f64()).unwrap())
			// .progress_chars("@")
		);
		bar.set_message(format!("{}",msg));
		ProgressWrapper{
			iter: iter,
			bar: bar
		}
	}
}
#[cfg(feature="progressbars")]
impl<I: ExactSizeIterator> Iterator for ProgressWrapper<I> {
	type Item=I::Item;
	fn next(&mut self) -> Option<Self::Item> {
		let item = self.iter.next();
		if item.is_some() {
			self.bar.inc(1);
		} else {
			self.bar.finish();
		}
		item
	}
}
#[cfg(feature="progressbars")]
impl<I: ExactSizeIterator> ExactSizeIterator for ProgressWrapper<I> {
	fn len(&self) -> usize { self.iter.len() }
}


pub fn named_range(max: usize, msg: &str) -> impl Iterator<Item=usize> {
	named_iter(0..max, msg)
}
#[cfg(feature="parallel")]
#[allow(dead_code)]
pub fn named_par_range(max: usize, msg: &str) -> impl ParallelIterator<Item=usize> {
	named_range(max, msg).par_bridge()
}
#[cfg(not(feature="parallel"))]
#[allow(dead_code)]
pub fn named_par_range(max: usize, msg: &str) -> impl Iterator<Item=usize> {
	named_range(max, msg)
}
pub fn named_iter<I: ExactSizeIterator>(iter: I, _msg: &str) -> impl ExactSizeIterator<Item=I::Item> {
	let ret = iter;
	#[cfg(feature="progressbars")]
	let ret = ProgressWrapper::new(ret, _msg);
	ret
}
#[cfg(feature="parallel")]
pub fn named_par_iter<I: ExactSizeIterator+Send>(iter: I, msg: &str) -> impl ParallelIterator<Item=I::Item> where I::Item: Send {
	par_iter(named_iter(iter, msg))
}
#[cfg(not(feature="parallel"))]
pub fn named_par_iter<I: ExactSizeIterator>(iter: I, msg: &str) -> impl ExactSizeIterator<Item=I::Item> {
	par_iter(named_iter(iter, msg))
}
#[cfg(feature="parallel")]
pub fn par_iter<I: Iterator+Send>(iter: I) -> impl ParallelIterator<Item=I::Item> where I::Item: Send {
	iter.par_bridge()
}
#[cfg(not(feature="parallel"))]
pub fn par_iter<I: Iterator>(iter: I) -> I {
	iter
}

