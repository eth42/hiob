use rand::random;


pub struct RandomPermutationGenerator {
	current_value: usize,
	max_value: usize,
	scaler: usize,
	bit_mask: usize,
	random_masks: Vec<usize>
}

impl RandomPermutationGenerator {
	pub fn new(max_value: usize, num_rounds: usize) -> Self {
		let next_power_of_two = max_value.next_power_of_two();
		let bit_mask = next_power_of_two-1;
		let helper = (next_power_of_two-1) / 3;
		let scaler = if (helper % 2) == 0 {
			(next_power_of_two*2-1) / 3
		} else {
			helper
		};
		let mut ret = Self{
			current_value: 0,
			max_value: max_value,
			scaler: scaler,
			bit_mask: bit_mask,
			random_masks: vec![0;num_rounds]
		};
		ret.reset_rand_state();
		ret
	}

	pub fn reset_rand_state(&mut self) {
		self.current_value = 0;
		self.random_masks.iter_mut()
		.for_each(|v| *v = random());
	}
	fn apply_round(&self, value: usize, i_round: usize) -> usize {
		((value * self.scaler) ^ self.random_masks[i_round]) & self.bit_mask
	}
	fn apply_rounds(&self, mut value: usize) -> usize {
		loop {
			(0..self.random_masks.len())
			.for_each(|i_round| value = self.apply_round(value, i_round));
			if value < self.max_value { break; }
		}
		value
	}
	#[allow(unused)]
	pub fn next_usize(&mut self) -> usize {
		let val = self.apply_rounds(self.current_value);
		self.current_value += 1;
		if self.current_value >= self.max_value {
			self.reset_rand_state();
		}
		val
	}
	pub fn next_usizes(&mut self, mut count: usize) -> Vec<usize> {
		let mut ret = vec![0; count];
		let mut offset = 0;
		while self.current_value + count >= self.max_value {
			(self.current_value..self.max_value)
			.enumerate()
			.map(|(i_vec,val)| (i_vec+offset, val))
			.for_each(|(i_vec, val)| {
				ret[i_vec] = self.apply_rounds(val);
			});
			offset += self.max_value-self.current_value;
			count -= self.max_value-self.current_value;
			self.reset_rand_state();
		}
		(self.current_value..self.current_value+count)
		.enumerate()
		.map(|(i_vec,val)| (i_vec+offset, val))
		.for_each(|(i_vec, val)| {
			ret[i_vec] = self.apply_rounds(val);
		});
		self.current_value += count;
		ret
	}
}


#[test]
fn test_random_permuter() {
	let max_val = 100_000_000;
	let n_rounds = 4;
	let mut permuter = RandomPermutationGenerator::new(max_val, n_rounds);
	println!("{:032b}", permuter.max_value);
	println!("{:032b}", permuter.bit_mask);
	println!("{:032b}", permuter.scaler);
	(0..20).for_each(|_| {
		println!("### Start ###");
		permuter.next_usizes(25).into_iter()
		.for_each(|val|{ println!("{:032b}", val); });
	println!("### End ###");
	})
}
