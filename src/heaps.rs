use std::{
	iter::{IntoIterator, Iterator},
	cmp::{Ordering, Eq, PartialEq, Ord, PartialOrd},
	collections::BinaryHeap
};

pub trait HeapKey: PartialOrd+Copy {}
impl<T: PartialOrd+Copy> HeapKey for T {}
pub trait HeapValue: Copy {}
impl<T: Copy> HeapValue for T {}

pub trait GenericHeap: IntoIterator {
	type Key: HeapKey;
	type Value: HeapValue;
	type Pair: GenericHeapPair<Key=Self::Key, Value=Self::Value>;
	fn wrapped_heap(&self) -> &BinaryHeap<Self::Pair>;
	fn wrapped_heap_mut(&mut self) -> &mut BinaryHeap<Self::Pair>;
	fn push(&mut self, key: Self::Key, value: Self::Value) {
		self.wrapped_heap_mut().push(Self::Pair::new(key, value))
	}
	fn pop(&mut self) -> Option<(Self::Key, Self::Value)> {
		let element = self.wrapped_heap_mut().pop();
		if element.is_some() {
			let element = element.unwrap();
			Some((*element.key_ref(), *element.value_ref()))
		} else { None }
	}
	fn peek(&self) -> Option<(Self::Key, Self::Value)> {
		let heap = self.wrapped_heap();
		let element = heap.peek();
		if element.is_some() {
			let element = element.unwrap();
			Some((*element.key_ref(), *element.value_ref()))
		} else { None }
	}
	fn size(&self) -> usize {
		self.wrapped_heap().len()
	}
	fn reserve(&mut self, size: usize) {
		self.wrapped_heap_mut().reserve(size);
	}
}


pub struct HeapIter<H: GenericHeap> {
	heap: H
}
impl<H: GenericHeap> HeapIter<H> {
	fn new(heap: H) -> Self { Self {heap: heap} }
}
impl<H: GenericHeap> Iterator for HeapIter<H> {
	type Item = (H::Key,H::Value);
	fn next(&mut self) -> Option<Self::Item> {
		self.heap.pop()
	}
}

pub struct MaxHeap<T: HeapKey, V: HeapValue> {
	heap: BinaryHeap<MaxHeapPair<T, V>>
}
impl<T: HeapKey, V: HeapValue> MaxHeap<T,V> {
	pub fn new() -> Self {
		MaxHeap{heap: BinaryHeap::new()}
	}
}
impl<T: HeapKey, V: HeapValue> GenericHeap for MaxHeap<T,V> {
	type Key = T;
	type Value = V;
	type Pair = MaxHeapPair<T,V>;
	fn wrapped_heap(&self) -> &BinaryHeap<MaxHeapPair<T,V>> { &self.heap }
	fn wrapped_heap_mut(&mut self) -> &mut BinaryHeap<MaxHeapPair<T,V>> { &mut self.heap }
}
impl<T: HeapKey, V: HeapValue> IntoIterator for MaxHeap<T,V> {
	type Item = (T,V);
	type IntoIter = HeapIter<Self>;
	fn into_iter(self) -> Self::IntoIter {
		HeapIter::new(self)
	}
}


pub struct MinHeap<T: HeapKey, V: HeapValue> {
	heap: BinaryHeap<MinHeapPair<T, V>>
}
impl<T: HeapKey, V: HeapValue> MinHeap<T,V> {
	pub fn new() -> Self {
		MinHeap{heap: BinaryHeap::new()}
	}
}
impl<T: HeapKey, V: HeapValue> GenericHeap for MinHeap<T,V> {
	type Key = T;
	type Value = V;
	type Pair = MinHeapPair<T,V>;
	fn wrapped_heap(&self) -> &BinaryHeap<MinHeapPair<T,V>> { &self.heap }
	fn wrapped_heap_mut(&mut self) -> &mut BinaryHeap<MinHeapPair<T,V>> { &mut self.heap }
}
impl<T: HeapKey, V: HeapValue> IntoIterator for MinHeap<T,V> {
	type Item = (T,V);
	type IntoIter = HeapIter<Self>;
	fn into_iter(self) -> Self::IntoIter {
		HeapIter::new(self)
	}
}





pub trait GenericHeapPair: Ord {
	type Key: HeapKey;
	type Value: HeapValue;
	fn new(key: Self::Key, value: Self::Value) -> Self;
	fn key_ref<'a>(&'a self) -> &'a Self::Key;
	fn value_ref<'a>(&'a self) -> &'a Self::Value;
}

pub struct MinHeapPair<T: HeapKey, V: HeapValue> {
	key: T,
	value: V
}
impl<T: HeapKey, V: HeapValue> GenericHeapPair for MinHeapPair<T, V> {
	type Key = T;
	type Value = V;
	fn new(key: T, value: V) -> Self {
		MinHeapPair{key:key, value:value}
	}
	fn key_ref<'a>(&'a self) -> &'a T {&self.key}
	fn value_ref<'a>(&'a self) -> &'a V {&self.value}
}
impl<T: HeapKey, V: HeapValue> PartialEq for MinHeapPair<T, V> {
	fn eq(&self, other: &Self) -> bool {
		self.key == other.key
	}
}
impl<T: HeapKey, V: HeapValue> PartialOrd for MinHeapPair<T, V> {
	fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
		let pcmp = self.key.partial_cmp(&other.key);
		if pcmp.is_none() { None } else { Some(pcmp.unwrap().reverse()) }
	}
}
impl<T: HeapKey, V: HeapValue> Eq for MinHeapPair<T, V> {}
impl<T: HeapKey, V: HeapValue> Ord for MinHeapPair<T, V> {
	fn cmp(&self, other: &Self) -> Ordering {
		self.key.partial_cmp(&other.key).unwrap()
	}
}

pub struct MaxHeapPair<T: HeapKey, V: HeapValue> {
	key: T,
	value: V
}
impl<T: HeapKey, V: HeapValue> GenericHeapPair for MaxHeapPair<T, V> {
	type Key = T;
	type Value = V;
	fn new(key: T, value: V) -> Self {
		MaxHeapPair{key:key, value:value}
	}
	fn key_ref<'a>(&'a self) -> &'a T {&self.key}
	fn value_ref<'a>(&'a self) -> &'a V {&self.value}
}
impl<T: HeapKey, V: HeapValue> PartialEq for MaxHeapPair<T, V> {
	fn eq(&self, other: &Self) -> bool {
		self.key == other.key
	}
}
impl<T: HeapKey, V: HeapValue> PartialOrd for MaxHeapPair<T, V> {
	fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
		self.key.partial_cmp(&other.key)
	}
}
impl<T: HeapKey, V: HeapValue> Eq for MaxHeapPair<T, V> {}
impl<T: HeapKey, V: HeapValue> Ord for MaxHeapPair<T, V> {
	fn cmp(&self, other: &Self) -> Ordering {
		self.key.partial_cmp(&other.key).unwrap()
	}
}