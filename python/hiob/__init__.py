from .hiob import *
from .hiob import RawBinarizationEvaluator
import numpy as np

def _float_type_name(float_type):
		if float_type == np.float32: return "f32"
		elif float_type == np.float64: return "f64"
		elif float_type == np.float16 and supports_f16(): return "f16"
		else: raise ValueError("Unsupported type '{:}'.".format(float_type))
def _bits_type_name(bits_type):
		if bits_type == bool: return "bool"
		elif bits_type == np.uint8: return "u8"
		elif bits_type == np.uint16: return "u16"
		elif bits_type == np.uint32: return "u32"
		elif bits_type == np.uint64: return "u64"
		elif bits_type == np.int8: return "i8"
		elif bits_type == np.int16: return "i16"
		elif bits_type == np.int32: return "i32"
		elif bits_type == np.int64: return "i64"
		else: raise ValueError("Unsupported type '{:}'.".format(bits_type))

class HIOB:
	def __init__(
		self,
		X: np.ndarray,
		n_bits: int,
		affine: bool = False,
		output_type: type = np.uint64,
		scale: float = None,
		centers: np.ndarray = None,
		center_biases: np.ndarray = None,
		init_greedy: bool = None,
		init_ransac: bool = None,
		ransac_pairs_per_bit: int = None,
		ransac_sub_sample: int = None,
		update_parallel: bool = None,
		displace_parallel: bool = None,
	):
		if n_bits < 1:
			raise ValueError("The number of bits should be at least 1.")
		self._input_type = X.dtype
		self._output_type = output_type
		# Match input type to string
		input_type_name = _float_type_name(self._input_type)
		output_type_name = _bits_type_name(self._output_type)
		# Create specific instance
		specific_type = "HIOB_{:}_{:}".format(input_type_name, output_type_name)
		self._rust_hiob = globals()[specific_type](
			X,
			n_bits,
			affine,
			scale,
			None if centers is None else centers.astype(self._input_type),
			None if center_biases is None else center_biases.astype(self._input_type),
			init_greedy,
			init_ransac,
			ransac_pairs_per_bit,
			ransac_sub_sample,
		)
		if not update_parallel is None: self._rust_hiob.update_parallel = update_parallel
		if not displace_parallel is None: self._rust_hiob.displace_parallel = displace_parallel
		attributes = {}
		for name in dir(self._rust_hiob):
			if name.startswith("__"): continue
			att = getattr(self._rust_hiob, name)
			if type(att).__name__ == "builtin_function_or_method":
				# Forward functions but automatically cast inputs to enforce
				# the usage of numpy arrays and the correct input types.
				def wrapper_fun_gen(fun):
					def wrapper_fun(*args, **kwargs):
						def auto_cast_argument(arg):
							# Automatically turn every list or tuple into numpy arrays
							if type(arg) in [tuple, list]: arg = np.array(arg)
							# Automatically cast every (float) array input to the correct input type
							if type(arg) == np.ndarray:# and arg.dtype.name.startswith("float"):
								return arg.astype(self._input_type)
							return arg
						return fun(
							*[auto_cast_argument(arg) for arg in args],
							**{kw: auto_cast_argument(arg) for kw,arg in kwargs}
						)
					return wrapper_fun
				setattr(self, name, wrapper_fun_gen(att))
			else:
				# Forward attributes with implicit getter
				def make_getter(specific_name):
					return lambda s: getattr(s._rust_hiob, specific_name)
				def make_setter(specific_name):
					return lambda s, v: s._rust_hiob.__setattr__(specific_name, v)
				getter = make_getter(name)
				setter = make_setter(name)
				# Test if the setter can be called, otherwise remove it
				try: setter(self, getter(self))
				except: setter = None
				attributes[name] = property(fget=getter,fset=setter)
		# Hacky class extension to add properties. Source:
		# https://stackoverflow.com/questions/48448074/adding-a-property-to-an-existing-object-instance
		wrapper_class_name = specific_type+"_Wrapper"
		wrapper_class = type(wrapper_class_name, (self.__class__,), {**self.__dict__, **attributes})
		self.__class__ = wrapper_class

class StochasticHIOB:
	def __init__(self, this_should_only_be_called_from_class_methods=True):
		if this_should_only_be_called_from_class_methods:
			print("Please do not call this constructor manually! Use the from_... functions.")
	def from_h5_file(
		file: str,
		dataset: str,
		sample_size: int,
		its_per_sample: int,
		n_bits: int,
		affine: bool = False,
		input_type: type = np.float32,
		output_type: type = np.uint64,
		perm_gen_rounds: int = None,
		scale: float = None,
		centers: np.ndarray = None,
		center_biases: np.ndarray = None,
		init_greedy: bool = None,
		init_ransac: bool = None,
		ransac_pairs_per_bit: int = None,
		ransac_sub_sample: int = None,
		update_parallel: bool = None,
		displace_parallel: bool = None,
		noise_std: float = None,
	):
		if n_bits < 1:
			raise ValueError("The number of bits should be at least 1.")
		self = StochasticHIOB(False)
		self._input_type = input_type
		self._output_type = output_type
		# Match input type to string
		input_type_name = _float_type_name(self._input_type)
		output_type_name = _bits_type_name(self._output_type)
		# Create specific instance
		specific_type = "StochasticHIOB_H5_{:}_{:}".format(input_type_name, output_type_name)
		self._rust_shiob = globals()[specific_type](
			file,
			dataset,
			sample_size,
			its_per_sample,
			n_bits,
			affine,
			perm_gen_rounds,
			scale,
			None if centers is None else centers.astype(self._input_type),
			None if center_biases is None else center_biases.astype(self._input_type),
			init_greedy,
			init_ransac,
			ransac_pairs_per_bit,
			ransac_sub_sample,
			noise_std,
		)
		self._post_constructor_init(specific_type, update_parallel, displace_parallel)
		return self
	def from_ndarray(
		X: np.ndarray,
		sample_size: int,
		its_per_sample: int,
		n_bits: int,
		affine: bool = False,
		output_type: type = np.uint64,
		perm_gen_rounds: int = None,
		scale: float = None,
		centers: np.ndarray = None,
		center_biases: np.ndarray = None,
		init_greedy: bool = None,
		init_ransac: bool = None,
		ransac_pairs_per_bit: int = None,
		ransac_sub_sample: int = None,
		update_parallel: bool = None,
		displace_parallel: bool = None,
		noise_std: float = None,
	):
		if n_bits < 1:
			raise ValueError("The number of bits should be at least 1.")
		self = StochasticHIOB(False)
		self._input_type = X.dtype
		self._output_type = output_type
		# Match input type to string
		input_type_name = _float_type_name(self._input_type)
		output_type_name = _bits_type_name(self._output_type)
		# Create specific instance
		specific_type = "StochasticHIOB_ND_{:}_{:}".format(input_type_name, output_type_name)
		self._rust_shiob = globals()[specific_type](
			X,
			sample_size,
			its_per_sample,
			n_bits,
			affine,
			perm_gen_rounds,
			scale,
			None if centers is None else centers.astype(self._input_type),
			None if center_biases is None else center_biases.astype(self._input_type),
			init_greedy,
			init_ransac,
			ransac_pairs_per_bit,
			ransac_sub_sample,
			noise_std,
		)
		self._post_constructor_init(specific_type, update_parallel, displace_parallel)
		return self
	def _post_constructor_init(self, specific_type, update_parallel=None, displace_parallel=None):
		if not update_parallel is None: self._rust_shiob.update_parallel = update_parallel
		if not displace_parallel is None: self._rust_shiob.displace_parallel = displace_parallel
		attributes = {}
		for name in dir(self._rust_shiob):
			if name.startswith("__"): continue
			att = getattr(self._rust_shiob, name)
			if type(att).__name__ == "builtin_function_or_method":
				# Forward functions but automatically cast inputs to enforce
				# the usage of numpy arrays and the correct input types.
				def wrapper_fun_gen(fun):
					def wrapper_fun(*args, **kwargs):
						def auto_cast_argument(arg):
							# Automatically turn every list or tuple into numpy arrays
							if type(arg) in [tuple, list]: arg = np.array(arg)
							# Automatically cast every (float) array input to the correct input type
							if type(arg) == np.ndarray:# and arg.dtype.name.startswith("float"):
								return arg.astype(self._input_type)
							return arg
						return fun(
							*[auto_cast_argument(arg) for arg in args],
							**{kw: auto_cast_argument(arg) for kw,arg in kwargs}
						)
					return wrapper_fun
				setattr(self, name, wrapper_fun_gen(att))
			else:
				# Forward attributes with implicit getter
				def make_getter(specific_name):
					return lambda s: getattr(s._rust_shiob, specific_name)
				def make_setter(specific_name):
					return lambda s, v: s._rust_shiob.__setattr__(specific_name, v)
				getter = make_getter(name)
				setter = make_setter(name)
				# Test if the setter can be called, otherwise remove it
				try: setter(self, getter(self))
				except: setter = None
				attributes[name] = property(fget=getter,fset=setter)
		# Hacky class extension to add properties. Source:
		# https://stackoverflow.com/questions/48448074/adding-a-property-to-an-existing-object-instance
		wrapper_class_name = specific_type+"_Wrapper"
		wrapper_class = type(wrapper_class_name, (self.__class__,), {**self.__dict__, **attributes})
		self.__class__ = wrapper_class

class BinarizationEvaluator:
	def __init__(self):
		self._raw = RawBinarizationEvaluator()
		self._usize = RawBinarizationEvaluator().brute_force_k_largest_dot_f32(np.ones((1,1),dtype=np.float32),np.ones((1,1),dtype=np.float32),1)[1].dtype
	def _clean_float_input(self, data, queries):
		data = np.array(data)
		ftype = data.dtype
		ftype_name = _float_type_name(ftype)
		queries = np.array(queries).astype(data.dtype)
		return data, queries, ftype_name
	def _clean_bin_input(self, data_bin, queries_bin):
		data_bin = np.array(data_bin)
		btype = data_bin.dtype
		btype_name = _bits_type_name(btype)
		queries_bin = np.array(queries_bin)
		if data_bin.dtype != queries_bin.dtype: raise ValueError("Data and query binarizations must have the same data type (received '{:}' and '{:}').".format(data_bin.dtype, queries_bin.dtype))
		return data_bin, queries_bin, btype_name
	def brute_force_k_largest_dot(self, data: np.ndarray, queries: np.ndarray, k: int):
		data, queries, ftype_name = self._clean_float_input(data, queries)
		fun = getattr(self._raw, "brute_force_k_largest_dot_{:}".format(ftype_name))
		return fun(data, queries, k)
	def brute_force_k_smallest_hamming(self, data_bin: np.ndarray, queries_bin: np.ndarray, k: int, chunk_size: bool=None):
		data_bin, queries_bin, btype_name = self._clean_bin_input(data_bin, queries_bin)
		fun = getattr(self._raw, "brute_force_k_smallest_hamming_{:}".format(btype_name))
		return fun(data_bin, queries_bin, k, chunk_size)
	def k_at_n_recall_prec_dot_neighbors(self, data_bin: np.ndarray, queries_bin: np.ndarray, true_neighbors: np.ndarray, n: int):
		data_bin, queries_bin, btype_name = self._clean_bin_input(data_bin, queries_bin)
		true_neighbors = np.array(true_neighbors).astype(self._usize)
		fun = getattr(self._raw, "k_at_n_recall_prec_dot_neighbors_{:}".format(btype_name))
		return fun(data_bin, queries_bin, true_neighbors, n)
	def k_at_n_recall_prec_hamming_neighbors(self, data: np.ndarray, queries: np.ndarray, pred_neighbors: np.ndarray, k: int):
		data, queries, ftype_name = self._clean_float_input(data, queries)
		pred_neighbors = np.array(pred_neighbors).astype(self._usize)
		fun = getattr(self._raw, "k_at_n_recall_prec_hamming_neighbors_{:}".format(ftype_name))
		return fun(data, queries, pred_neighbors, k)
	def k_at_n_recall_prec_all(self, true_neighbors: np.ndarray, pred_neighbors: np.ndarray):
		true_neighbors = np.array(true_neighbors).astype(self._usize)
		pred_neighbors = np.array(pred_neighbors).astype(self._usize)
		fun = getattr(self._raw, "k_at_n_recall_prec_all")
		return fun(true_neighbors, pred_neighbors)
	def k_at_n_recall(self, data: np.ndarray, data_bin: np.ndarray, queries: np.ndarray, queries_bin: np.ndarray, k: int, n: int):
		data, queries, ftype_name = self._clean_float_input(data, queries)
		data_bin, queries_bin, btype_name = self._clean_bin_input(data_bin, queries_bin)
		fun = getattr(self._raw, "k_at_n_recall_{:}_{:}".format(ftype_name,btype_name))
		return fun(data, data_bin, queries, queries_bin, k, n)
	def refine(self, data: np.ndarray, queries: np.ndarray, hamming_ids: np.ndarray, k: int, chunk_size: int=None):
		data, queries, ftype_name = self._clean_float_input(data, queries)
		fun = getattr(self._raw, "refine_{:}".format(ftype_name))
		return fun(data, queries, hamming_ids, k, chunk_size)
	def refine_h5(self, data_file: str, data_dataset: str, queries: np.ndarray, hamming_ids: np.ndarray, k: int, chunk_size: int=None):
		ftype_name = _float_type_name(queries.dtype)
		fun = getattr(self._raw, "refine_h5_{:}".format(ftype_name))
		return fun(data_file, data_dataset, queries, hamming_ids, k, chunk_size)
	def refine_with_other_bin(self, data_bin: np.ndarray, queries_bin: np.ndarray, hamming_ids: np.ndarray, k: int, chunk_size: int=None):
		data_bin, queries_bin, btype_name = self._clean_bin_input(data_bin, queries_bin)
		fun = getattr(self._raw, "refine_with_other_bin_{:}".format(btype_name))
		return fun(data_bin, queries_bin, hamming_ids, k, chunk_size)
	def query(self, data, data_bin, queries, queries_bin, k, n, chunk_size=None):
		assert k <= n
		data, queries, ftype_name = self._clean_float_input(data, queries)
		data_bin, queries_bin, btype_name = self._clean_bin_input(data_bin, queries_bin)
		fun = getattr(self._raw, "query_{:}_{:}".format(ftype_name, btype_name))
		return fun(data, data_bin, queries, queries_bin, k, n, chunk_size)
	def query_h5(self, data_file, data_dataset, data_bin, queries, queries_bin, k, n, chunk_size=None):
		assert k <= n
		ftype_name = _float_type_name(queries.dtype)
		data_bin, queries_bin, btype_name = self._clean_bin_input(data_bin, queries_bin)
		fun = getattr(self._raw, "query_h5_{:}_{:}".format(ftype_name, btype_name))
		return fun(data_file, data_dataset, data_bin, queries, queries_bin, k, n, chunk_size)
	def query_cascade(self, data, data_bins, queries, queries_bins, k, ns, chunk_size=None):
		assert len(data_bins) > 0
		assert len(data_bins) == len(queries_bins)
		assert len(data_bins) == len(ns)
		data, queries, ftype_name = self._clean_float_input(data, queries)
		btype_name = None
		cleaned_data_bins = []
		cleaned_queries_bins = []
		for data_bin, queries_bin in zip(data_bins, queries_bins):
			data_bin, queries_bin, local_btype_name = self._clean_bin_input(data_bin, queries_bin)
			if btype_name is None: btype_name = local_btype_name
			elif btype_name != local_btype_name: raise ValueError("All binarizations must have the same data type (received '{:}' and '{:}').".format(btype_name, local_btype_name))
			cleaned_data_bins.append(data_bin)
			cleaned_queries_bins.append(queries_bin)
		fun = getattr(self._raw, "query_cascade_{:}_{:}".format(ftype_name, btype_name))
		return fun(data, cleaned_data_bins, queries, cleaned_queries_bins, k, ns, chunk_size)
	def query_cascade_h5(self, data_file: str, data_dataset: str, data_bins, queries, queries_bins, k, ns, chunk_size=None):
		assert len(data_bins) > 0
		assert len(data_bins) == len(queries_bins)
		assert len(data_bins) == len(ns)
		ftype_name = _float_type_name(queries.dtype)
		btype_name = None
		cleaned_data_bins = []
		cleaned_queries_bins = []
		for data_bin, queries_bin in zip(data_bins, queries_bins):
			data_bin, queries_bin, local_btype_name = self._clean_bin_input(data_bin, queries_bin)
			if btype_name is None: btype_name = local_btype_name
			elif btype_name != local_btype_name: raise ValueError("All binarizations must have the same data type (received '{:}' and '{:}').".format(btype_name, local_btype_name))
			cleaned_data_bins.append(data_bin)
			cleaned_queries_bins.append(queries_bin)
		fun = getattr(self._raw, "query_cascade_h5_{:}_{:}".format(ftype_name, btype_name))
		return fun(data_file, data_dataset, cleaned_data_bins, queries, cleaned_queries_bins, k, ns, chunk_size)

class THX:
	def __init__(
		self,
		X: np.ndarray,
		n_bits_per_layer: int,
		fanout: int,
	):
		self._type = X.dtype
		# Match input type to string
		type_name = _bits_type_name(self._type)
		# Create specific instance
		specific_type = "THX_{:}_{:}".format(type_name, fanout)
		if not specific_type in globals().keys():
			raise ValueError("Unsupported fanout; can't find type {:}".format(specific_type))
		self._rust_thx = globals()[specific_type](X, n_bits_per_layer)
		attributes = {}
		for name in dir(self._rust_thx):
			if name.startswith("__"): continue
			att = getattr(self._rust_thx, name)
			if type(att).__name__ == "builtin_function_or_method":
				# Forward functions
				setattr(self, name, att)
			else:
				# Forward attributes with implicit getter
				def make_getter(specific_name):
					return lambda s: getattr(s._rust_thx, specific_name)
				def make_setter(specific_name):
					return lambda s, v: s._rust_thx.__setattr__(specific_name, v)
				getter = make_getter(name)
				setter = make_setter(name)
				# Test if the setter can be called, otherwise remove it
				try: setter(self, getter(self))
				except: setter = None
				attributes[name] = property(fget=getter,fset=setter)
		# Hacky class extension to add properties. Source:
		# https://stackoverflow.com/questions/48448074/adding-a-property-to-an-existing-object-instance
		wrapper_class_name = specific_type+"_Wrapper"
		wrapper_class = type(wrapper_class_name, (self.__class__,), {**self.__dict__, **attributes})
		self.__class__ = wrapper_class
	
	def compute_n_nodes(X: np.ndarray, n_bits_per_layer: int, fanout: int):
		# Match input type to string
		type_name = _bits_type_name(X.dtype)
		# Create specific instance
		specific_type = "THX_{:}_{:}".format(type_name, fanout)
		if not specific_type in globals().keys():
			raise ValueError("Unsupported fanout; can't find type {:}".format(specific_type))
		return globals()[specific_type].compute_n_nodes(X, n_bits_per_layer)


