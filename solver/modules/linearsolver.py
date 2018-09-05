import numpy as np
import copy
import re
from fractions import Fraction as Q

def prmatr(m):
	"""Виводить матрицю у звичайному вигляді, без технічних символів та слів."""

	for i in m:
		for j in i:
			print(j, end=" ")
		print()

class InputParser:
	"""Клас для оброблення вхідної інформації з файлу або об'єкту.
	
	Повертає оброблену інформацію через метод get_data."""
	op_list = ["<=", ">=", "<", ">", "=", "arbitrary"]
	
	def __init__(self, input_info):
		inner_text = ""
		if input_info["data_type"] == "file":
			with open(input_info["data"]) as f:
				inner_text = f.read()
		elif input_info["data_type"] == "string":
			inner_text = input_info["data"]

		elif input_info["data_type"] == "object":
			cont = input_info["data"]
			self.first_line_vect = list(map(Q, cont["obj_func"]))
			self.task_type = cont["task_type"]
			self.last_conditions = cont["last_cond"]
			for i in range(len(self.last_conditions)):
				self.last_conditions[i][1] = Q(self.last_conditions[i][1])
			for i in range(len(cont["matrix"])):
				cont["matrix"][i] = list(map(Q, cont["matrix"][i]))
			matr_len = 0
			for i in cont["matrix"]:
				if len(i) > matr_len:
					matr_len = len(i)
			for i in range(len(cont["matrix"])):
				if len(cont["matrix"][i]) < matr_len:
					cont["matrix"][i] = cont["matrix"][i] + ([Q(0)] * (matr_len - len(cont["matrix"][i])))
			if len(self.first_line_vect) < matr_len:
				self.first_line_vect = self.first_line_vect + [Q(0)] * (matr_len - len(self.first_line_vect))
			self.first_line_vect = np.array(self.first_line_vect)
			self.main_matrix = np.array(cont["matrix"])
			self.inequalities = cont["ineq"]
			self.constants_vector = list(map(Q, cont["constants"]))
			self.expected_error = ""
			self.result = ""
			self.result_list = ""
			return

		else:
			print("Unknown format of input data")

		inner_text = inner_text.replace('\t', '').replace(' ', '').split("\n")

		#Обробка першого рядка з цільовою функцією
		counter = 0
		first_line = inner_text[counter]
		while(first_line == '' or first_line[0] == '#'):
			counter += 1
			first_line = inner_text[counter]
		first_line = InputParser._format_to_math_form(first_line)
		self.task_type, self.first_line_vect = self._parse_first_line(first_line)

		last_cond = ''
		raw_matrix = []
		raw_constants = []
		self.inequalities = []
		for line in inner_text[counter + 1:]:
			if line == '' or line[0] == "#":
				continue
			elif line[:3] == ">>>":
				last_cond = ""
				break
			elif line[0] != '|':
				last_cond = line
				break

			#Обробка умов та заповнення відповідної їм матриці
			line = InputParser._format_to_math_form(line[1:])
			for i in InputParser.op_list:
				if i in line:
					self.inequalities.append(i)
					break
			curr_sym = self.inequalities[len(self.inequalities)-1]
			line = line[0] + line[1:line.find(curr_sym)].replace("-", "+-") + line[line.find(curr_sym):]

			parts_arr,  constant = line[:line.find(curr_sym)].split("+"), line[line.find(curr_sym)+len(curr_sym):]
			raw_constants.append(Q(constant))
			raw_dict = {}
			for i in parts_arr:
				num, ind = i[:-1].split("x[")
				raw_dict[int(ind)] = Q(num)
			raw_list = [0] * max(raw_dict, key=int)
			for k, v in raw_dict.items():
				raw_list[k - 1] = v
			raw_matrix.append(raw_list)

		self.var_quantity = 0
		for row in raw_matrix:
			if len(row) > self.var_quantity:
				self.var_quantity = len(row)
		for k, row in enumerate(raw_matrix):
			if len(row) < self.var_quantity:
				for i in range(len(row), self.var_quantity):
					raw_matrix[k].append(Q(0, 1))

		self.main_matrix = np.array(raw_matrix)
		self.constants_vector = np.array(raw_constants)

		#Обробка рядка з обмеженнями змінних
		self.last_conditions = self._parse_last_cond(last_cond)
		#Обробка рядка з бажаним результатом розв'язку (використовується лише в тестуванні)
		self.result_list = []
		self.result = ""
		self.expected_error = ""
		counter = inner_text.index(last_cond) + 1
		last_line = ""
		if counter < len(inner_text):
			last_line = inner_text[counter]
		while(counter < len(inner_text) - 1 and last_line[:3] != '>>>'):
			counter += 1
			last_line = inner_text[counter]
		if counter >= len(inner_text) - 1 and last_line[:3] != '>>>':
			return
		raw_list, result, expected_error = self._parse_results(last_line)
		if raw_list != "":
			for i in raw_list.split(','):
				self.result_list.append(Q(i))
		self.result = result
		self.expected_error = expected_error


	@staticmethod
	def _format_to_math_form(line):
		"""Видаляє з рядка всі пробіли та додає одиничні множники де потрібно."""

		if line[0] == "x":
			line = "1" + line
		return line.replace(' ', '').replace('-x', '-1x').replace('+x', '+1x')

	def _parse_first_line(self, line):
		"""Отримує строку та обробляє її текст як інформацію про цільову функцію.

		Форма виводу: |numpy array of Qs| [ { factor's fraction }, ... ].
		Індекс кожного Q відповідає декрементованому індексу відповідної змінної.
		Не підтримує некоректну вхідну інформацію та константи в цільовій функції."""

		raw_array = {} #Результуючий масив, але невпорядкований

		#Розділення строки, використовуючи "+" як розділювач, з подальшим записом інформації в модель цільової функції в змінній first_line_vect
		#Змінна task_type містить строку ("max" або "min"), в залежності від вхідних даних
		line, task_type = line[:line.find("=>")], line[line.find("=>")+2:]
		line = line[0] + line[1:].replace('-', '+-')
		op_arr = line.split('+')
		for i in op_arr:
			num, index = i[:-1].split("x[")
			raw_array[int(index)] = Q(num)

		first_line_vect = [Q(0,1)] * max(raw_array, key=int)
		for k, v in raw_array.items():
			first_line_vect[k - 1] = v
		return task_type, np.array(first_line_vect)

	def _parse_last_cond(self, line):
		"""Отримує строку та обробляє її як таку, що містить інформацію про загальні умови.

		Форма виводу: |list of tuples| [ ( { index of inequality sign }, { condition's fraction } ), ... ].
		Індекс кожної пари відповідає декрементованому індексу відповідної змінної.
		Змінні не мають бути написані зі знаком "-"."""
		
		if line == "":
			return [["arbitrary", Q(0)]] * self.var_quantity	
		cond_list = line.split(",")
		raw_dict = {}
		for expr in cond_list:
			op_index = 0
			for i in InputParser.op_list:
				if i in expr:
					op_sym = i
					break
			f_pair = [op_sym, Q(expr[expr.find(op_sym)+len(op_sym):])]
			raw_dict[int(expr[2:expr.find(op_sym)-1])] = f_pair
		last_conditions = [[InputParser.op_list[5], Q(0)]] * max(raw_dict, key=int)
		for k, v in raw_dict.items():
			last_conditions[k - 1] = v
		complete_list = [["arbitrary", Q(0)]] * self.var_quantity
		complete_list[:len(last_conditions)] = last_conditions
		
		return complete_list

	def _parse_results(self, line):
		"""Отримує строку так обробляє її як таку, що містить інформацію про бажаний результат.

		Інформація, отримана з цього методу використовується у тестуванні.
		Форма виводу: |tuple| ( { масив значень відповідних змінних }, { значення цільової функції } )."""

		if not "(" in line:
			return "", "", line[3:]
		return line[line.find("(") + 1:line.find(")")], line[line.find("|") + 1:], ""

	def get_data(self):
		"""Повертає об'єкт з усією обробленою інформацією, що була отримана."""

		return {
			"objective_function": self.first_line_vect,
			"task_type": self.task_type,
			"last_conditions": self.last_conditions,
			"matrix": self.main_matrix,
			"inequalities": self.inequalities,
			"constants": self.constants_vector,
			"expected_vect": self.result_list,
			"expected_result": self.result,
			"error": self.expected_error
		}

	def print_first_line(self):
		"""Виводить вектор цільової функції."""

		print("First line: {}\n".format(self.first_line_vect))

	def print_task_type(self):
		"""Виводить тип задачі."""

		print("Task type: {}\n".format(self.task_type))

	def print_last_cond(self):
		"""Виводить вектор обмежень змінних."""

		print("Last line: {}\n".format(self.last_conditions))

	def print_main_matrix(self):
		"""Виводить основну матрицю."""

		print("Matrix: {}\n".format(self.main_matrix))

	def print_constants(self):
		"""Виводить вектор вільних змінних."""

		print("Constants' vector: {}\n".format(self.constants_vector))

	def print_inequalities(self):
		"""Виводить вектор знаків рівності\нерівності з системи початкових умов."""

		print("Inequalities' vector: {}\n".format(self.inequalities))


# ------ Solver class section ------


class Solver:
	"""Основний клас, що містить спільні для всіх способів розв'язання методи.

	Є базовим для класів, які відповідають різним способам розв'язання."""

	def __init__(self, input_data):
		reader_data = ""
		reader_data = InputParser(input_data).get_data()
		self.objective_function = reader_data["objective_function"]
		self.task_type = reader_data["task_type"]
		self.last_conditions = reader_data["last_conditions"]
		self.matrix = reader_data["matrix"]
		self.inequalities = reader_data["inequalities"]
		self.constants = reader_data["constants"]
		if input_data['data_type'] != "object":
			self.expected_vect = np.array(reader_data["expected_vect"])
			self.expected_result = Q(reader_data["expected_result"]) if reader_data["expected_result"] != "" else ""
			self.expected_error = reader_data["error"]
		self.result_error = ""
		self.mute = input_data["mute"]
		self.col_num = 0
		self.row_num = 0
		self.basis = []
		self.basis_koef = np.array([])
		self.obj_shift = Q(0)



		self.writer = Logger(self.mute)
		self.writer.initiate("initial_info")
		
		was_max = False
		if self.task_type == "max":
			was_max = True

		self.writer.log(info=reader_data, is_max=was_max)

		if was_max:
			self.objective_function *= Q(-1)

	def _make_basis_column(self):
		"""Зводить задану в атрибутах колонку до одиничного вектора з одиницею на місці обраного в атрибутах рядка."""

		self.writer.initiate("basis_col")
		prev_table = copy.deepcopy(self._get_all_table_data())
		operations_list = [1] * len(self.matrix)

		if self.matrix[self.row_num][self.col_num] == 0:
			raise SolvingError("В якості ведучого елемента вибрано нуль, подальші розрахунки неможливі")
			return
		elif self.matrix[self.row_num][self.col_num] != 1:
			operations_list[self.row_num] = self.matrix[self.row_num][self.col_num]
			self.constants[self.row_num] /= self.matrix[self.row_num][self.col_num]
			self.matrix[self.row_num] /= self.matrix[self.row_num][self.col_num]
		
		chosen_row = self.matrix[self.row_num]
		for i in [x for x in range(len(self.matrix)) if x != self.row_num]:
			operations_list[i] = self.matrix[i][self.col_num]
			self.constants[i] -= self.constants[self.row_num] * self.matrix[i][self.col_num]
			self.matrix[i] -= chosen_row * self.matrix[i][self.col_num]
			was_changed = True

		self.writer.log(
			p_table=prev_table,
			table=self._get_all_table_data(),
			op=operations_list,
			row=self.row_num,
			col=self.col_num
		)

	def _make_conditions_equalities(self):
		"""Зводить всі нерівності умов до рівностей.
		
		Не підтримуються строгі нерівності."""

		was_changed = False
		for i in range(len(self.inequalities)):
			if self.inequalities[i] == "<" or self.inequalities[i] == ">":
				raise SolvingError("Строгі нерівності не підтримуються")
			elif self.inequalities[i] == ">=":
				self.matrix[i] *= Q(-1)
				self.constants[i] *= Q(-1)
				self.inequalities[i] = "<="
				was_changed = True
			if self.inequalities[i] == "<=":
				temp_matrix = []
				for j in range(len(self.matrix)):
					temp_matrix.append([Q(0)] * (len(self.matrix[0]) + 1))
				temp_matrix[i][-1] = Q(1)
				temp_matrix = np.array(temp_matrix)
				temp_matrix[:,:-1] = self.matrix
				self.matrix = temp_matrix
				self.inequalities[i] = "="
				self.objective_function = np.append(self.objective_function, Q(0))
				self.last_conditions.append([">=", Q(0)])
				was_changed = True
		if was_changed:
			self.writer.initiate("inequalities")
			self.writer.log(
				objective_function = self.objective_function, 
				constant = self.obj_shift, 
				matrix = self.matrix, 
				constants = self.constants, 
				inequalities = self.inequalities,
				task_type = "min",
				last_cond = self.last_conditions
			)

	def _get_basis_vectors_nums(self):
		"""Повертає список змінних, чиї вектори входять до одиничної підматриці матриці."""

		self.writer.initiate("show_basis")
		temp_matrix = self.matrix.T
		result = [-1] * len(temp_matrix[0])
		for i in range(len(temp_matrix)):
			num = -1
			for j in range(len(temp_matrix[i])):
				if temp_matrix[i][j] != 0 and temp_matrix[i][j] != 1:
					num = -1
					break
				if temp_matrix[i][j] == 1:
					if num == -1:
						num = j
					else:
						num = -1
						break
			if num > -1:
				result[num] = i
		if -1 in result:
			self.writer.log(basis=None)
		else:
			self.writer.log(basis=result)
		return result

	def _set_basis_koef(self):
		"""Записує порядкові номери та коефіцієнти базисних змінних в цільовій функції в окремі вектори."""

		self.basis[self.row_num] = self.col_num
		self.basis_koef[self.row_num] = self.objective_function[self.col_num]

	def _expand_objective_function_if_needed(self):
		"""Додає в цільову функцію штучні змінні з нульовим коефіцієнтом."""

		diff = len(self.matrix[0]) - len(self.objective_function)
		if diff > 0:
			num = len(self.objective_function)
			temp_array = [Q(0)] * (num + diff)
			temp_array[:num] = self.objective_function
			self.objective_function = np.array(temp_array)

	def get_result(self):
		"""Повертає результат обчислень"""

		return self.writer.get_logs()

class SimplexSolver(Solver):
	"""Виконує розв'язання задачі лінійного програмування симплекс методом."""

	def __init__(self, input_data):
		super(SimplexSolver, self).__init__(input_data)
		self.deltas = np.array([])
		self.thetas = np.array([])
		self.artificial_variables = []

	def print_all(self):
		"""Виводить в консоль всю доступну на даний момент інформацію про розвиток розв'язку задачі."""

		print(">------------------------------------------------------------<")
		print("Objective func: {}".format(self.objective_function))
		print("Basis constants: {}".format(self.basis_koef))
		print("Basis variables: {}".format(self.basis))
		print("Main matrix:\n-------------------------------")
		prmatr(self.matrix)
		print("-------------------------------\nConstants: {}".format(self.constants))
		print("Thetas: {}".format(self.thetas))
		print("Deltas: {}".format(self.deltas))
		print(">------------------------------------------------------------<\n")

	def _calculate_deltas(self):
		"""Розраховує вектор з дельтами."""

		self.writer.initiate("deltas")
		temp_matrix = self.matrix.T
		temp_array = []
		for i in range(len(temp_matrix)):
			temp_array.append(self.objective_function[i] - temp_matrix[i].dot(self.basis_koef))
			self.writer.log(index=i, const=self.objective_function[i], mult1=temp_matrix[i], mult2=self.basis_koef, res=temp_array[-1])
		self.deltas = np.array(temp_array)
		self.writer.log(table=self._get_all_table_data())

	def _calculate_thetas(self):
		"""Розраховує вектор-стовпчик з відношеннями "тета"."""

		self.thetas = [Q(0)] * len(self.constants)
		self.writer.initiate("thetas")

		for i in range(len(self.matrix)):
			if self.matrix[i][self.col_num] == 0:
				self.thetas[i] = -1
				self.writer.log(div1=self.constants[i], div2=self.matrix[i][self.col_num], error="zerodiv", ind=self.basis[i])
			elif self.matrix[i][self.col_num] < 0 and self.constants[i] == 0:
				self.thetas[i] = -1
				self.writer.log(div1=self.constants[i], div2=self.matrix[i][self.col_num], error="negative", ind=self.basis[i])
			else:
				self.thetas[i] = self.constants[i] / self.matrix[i][self.col_num]
				self.writer.log(div1=self.constants[i], div2=self.matrix[i][self.col_num], res=self.thetas[i], ind=self.basis[i])
		self.writer.log(table=self._get_all_table_data())

	def _find_ind_of_min_theta(self):
		"""Знаходить індекс ведучого рядка.

		Повертає -1 якщо такого немає."""

		self.writer.initiate("min_theta")
		temp_min = 0
		min_set = False
		found_ind = -1
		for i in range(len(self.thetas)):
			if self.thetas[i] >= 0:
				temp_min = self.thetas[i]
				found_ind = i
				min_set = True
				break
		if min_set:
			for i in range(len(self.thetas)):
				if self.thetas[i] < 0:
					continue
				if self.thetas[i] < temp_min:
					temp_min = self.thetas[i]
					found_ind = i

		self.writer.log(ind=self.basis[found_ind] if found_ind != -1 else -1)
		return found_ind

	def _reset_deltas_n_thetas(self):
		"""Скидає значення векторів "тета" та "дельта"."""

		self.deltas = ["-"] * len(self.matrix[0])
		self.thetas = ["-"] * len(self.matrix)

	def _make_constants_positive_if_needed(self):
		"""Якщо всі вільні члени від'ємні, то переходить до іншого базису."""

		self._reset_deltas_n_thetas()
		self.writer.initiate("initial_table")
		self.writer.log(table=self._get_all_table_data())
		for i in self.constants:
			if i >= 0:
				return
		unset = True
		for i in range(len(self.constants)):
			for j in range(len(self.matrix[i])):
				if self.matrix[i][j] < 0:
					self.col_num = j
					self.row_num = i
					unset = False
					break
			if not unset:
				break
		if not unset:
			self._make_basis_column()
		self.basis = self._get_basis_vectors_nums()
		for i in range(len(self.basis)):
			self.basis_koef[i] = self.objective_function[self.basis[i]]

	def _add_artificial_basis(self):
		"""Створює одиничну підматрицю за допомогою штучних змінних М-методом."""

		self.writer.initiate("artificial_basis")
		M = np.amax(np.array(np.append(np.append(self.matrix, self.constants), self.objective_function))) + 1
		for i in range(len(self.basis)):
			if self.basis[i] == -1:
				temp_matrix = []
				for j in range(len(self.matrix)):
					temp_matrix.append([Q(0)] * (len(self.matrix[0]) + 1))
				temp_matrix[i][-1] = Q(1)
				temp_matrix = np.array(temp_matrix)
				temp_matrix[:,:-1] = self.matrix
				self.matrix = temp_matrix
				self.objective_function = np.append(self.objective_function, M)
				self.artificial_variables.append(len(self.objective_function) - 1)
				self.last_conditions.append([">=", Q(0)])
				self.basis[i] = len(self.objective_function) - 1
		
		self.writer.log(
			m = M,
			matrix = self.matrix,
			objective_function = self.objective_function,
			constant = self.obj_shift,
			constants = self.constants,
			last_cond = self.last_conditions,
			task_type = "min",
			inequalities = self.inequalities
		)

	def _normalize_conditions(self):
		"""Зводить задачу до аналогічної, у якій всі змінні невід'ємні."""

		self.writer.initiate("normalizing")
		self.substitution_queue = []
		self.arbitrary_pairs = []
		for i in range(len(self.last_conditions)):
			if len(self.last_conditions[i][0]) == 1:
				return False

			elif self.last_conditions[i][0] == "<=":
				for j in range(len(self.matrix)):
					self.matrix[j][i] *= -1
				self.substitution_queue.insert(0, (i, "*=-1"))
				self.objective_function[i] *= -1
				self.last_conditions[i] = [">=", self.last_conditions[i][1] * -1]
				self.writer.log(index=i, op="a")
			if self.last_conditions[i][0] == ">=":
				if self.last_conditions[i][1] != 0:
					for j in range(len(self.matrix)):
						self.constants[j] -= self.matrix[j][i] * self.last_conditions[i][1]
					self.obj_shift += self.objective_function[i] * self.last_conditions[i][1]
					self.substitution_queue.insert(0, (i, "+={}".format(self.last_conditions[i][1])))
					self.writer.log(index=i, op="b", substitution=self.last_conditions[i][1])
					self.last_conditions[i][1] = Q(0)


			if self.last_conditions[i][0] == "arbitrary":
				new_pair = i, len(self.matrix[0])
				self.writer.log(index=i, op="c")
				new_matrix = []
				for j in range(len(self.matrix)):
					new_matrix.append([Q(0)] * (len(self.matrix[0]) + 1))
				for j in range(len(self.matrix)):
					new_matrix[j][-1] = -self.matrix[j][i]

				
				new_matrix = np.array(new_matrix)
				new_matrix[:,:-1] = self.matrix
				self.matrix = new_matrix
				
				self.objective_function = np.append(self.objective_function, -self.objective_function[i])
				self.last_conditions[i] = [">=", Q(0)]
				self.last_conditions.append([">=", Q(0)])
				self.arbitrary_pairs.append(new_pair)

		if len(self.arbitrary_pairs) > 0 or len(self.substitution_queue) > 0:
			self.writer.log(
				matrix = self.matrix,
				inequalities = self.inequalities,
				constants = self.constants,
				last_conditions = self.last_conditions,
				objective_function = self.objective_function,
				constant = self.obj_shift,
				task_type = "min"
			)
		return True

	def _cancel_subtitution(self):
		"""Повертає початкові значення змінним, якщо відбулася заміна."""

		self.writer.initiate("substitution")

		self.final_result = [Q(0)] * len(self.matrix[0])
		for i in range(len(self.basis)):
			self.writer.log(ind=self.basis[i], val=self.constants[i])
			self.final_result[self.basis[i]] = self.constants[i]

		if self.task_type == "max":
			self.writer.log(max=True)
			self.objective_function *= -1

		self.writer.log(sub_queue=self.substitution_queue)

		for i in self.substitution_queue:
			exec("self.final_result[i[0]]" + i[1])
			if "*" in i[1]:
				self.objective_function[i[0]] *= Q(i[1][2:])

		for i in self.arbitrary_pairs:
			self.writer.log(arb1=i[0], arb2=i[1])
			self.final_result[i[0]] -= self.final_result[i[1]]

	def _get_col_num(self, indices_list):
		"""Повертає індекс ведучого стовпчика, засновуючись на векторі з дельтами."""

		self.writer.initiate("get_col")
		if len(indices_list) == 1:
			self.writer.log(num=indices_list[0])
			return indices_list[0]
		for i in range(len(indices_list)):
			temp_thetas = []
			for j in range(len(self.matrix)):
				if self.matrix[j][indices_list[i]] == 0 or (self.constants[j] == 0 and self.matrix[j][indices_list[i]] < 0):
					temp_thetas.append(-1)
				else:
					temp_thetas.append(self.constants[j] / self.matrix[j][indices_list[i]])
			for j in temp_thetas:
				if j >= 0:
					break
			else:
				indices_list[i] = -1
		for i in indices_list:
			if i >= 0:
				self.writer.log(num=i)
				return i
		self.writer.log(no_col=True)
		return -1

	def _check_for_ambiguous_result(self):
		"""Перевіряє чи відповідає небазисній змінній нульова дельта.

		Якщо штучна змінна базисна, її пара теж вважається базисною."""

		basis = set(self.basis)
		for i in self.arbitrary_pairs:
			if i[0] in basis:
				basis.add(i[1])
			elif i[1] in basis:
				basis.add(i[0])
		non_basis_set = set(range(len(self.objective_function))) - basis
		for i in non_basis_set:
			if self.deltas[i] == 0:
				self.result_error = "infinite|{}".format(self.result)
				raise SolvingError("Базисній змінній відповідає нульова дельта:\nІснує нескінченна кількість розв'язків\nОптимальне значення цільової функції: {}".format(self.result))

	def _check_for_empty_allowable_area(self):
		"""Перевіряє чи є у кінцевому векторі з множниками змінних штучна змінна з відмнінним від нуля множником."""

		for i in self.artificial_variables:
			if self.final_result[i] != 0:
				self.result_error = "empty"
				raise SolvingError("В оптимальному розв'язку присутня штучна змінна:\nДопустима область порожня")

	def _get_all_table_data(self):
		"""Повертає всю необхідну для виведення симплекс таблиці інформацію."""

		return {
			"matrix": self.matrix,
			"objective_function": self.objective_function,
			"basis": self.basis,
			"basis_koef": self.basis_koef,
			"constants": self.constants,
			"deltas": self.deltas,
			"thetas": self.thetas
		}

	def _check_criterion_of_optimality(self):
		"""Перевіряє критерій оптимальності.

		Якщо критерій виконується, але наявні від'ємні вільні члени, виконується перехід до нового базису."""

		self.writer.initiate("optimality")
		for i in range(len(self.constants)):
			if self.constants[i] < 0:
				another_iteration = False
				for j in range(len(self.matrix[i])):
					if self.matrix[i][j] < 0:
						self.row_num = i
						self.col_num = j
						self.writer.log(ind_row=self.basis[i], ind_col=j, another_iteration=True)
						self._make_basis_column()
						self._set_basis_koef()
						another_iteration  = True
						break
				if another_iteration:
					break
				else:
					self.result_error = "empty"
					raise SolvingError("Неможливо отримати опорний розв'язок, всі дельта не менші нуля, але не всі значення базисних змінних більші рівні нулю:\nДопустима область порожня")
		else:
			self.writer.log(another_iteration=False)
			return True
		return False

	def _final_preparations(self):
		"""Скасовує всі заміни та записує результат у відповідні атрибути."""

		self.writer.initiate("final")
		self.result_vect = self.final_result[:self.initial_variables_quantity]
		obj_func_val = self.objective_function[:self.initial_variables_quantity].dot(np.array(self.result_vect))
		self.result = obj_func_val
		self._check_for_ambiguous_result()
		self._check_for_empty_allowable_area()

		self.writer.log(
			big_vect = self.final_result,
			vect = self.result_vect,
			obj_val = self.result
		)

	def solve(self):
		"""Розв'язує задачу симплекс методом."""

		self.initial_variables_quantity = len(self.matrix[0])
		if not self._normalize_conditions():
			raise SolvingError("В заданих умовах обмеження змінних містять строгі знаки нерівностей або знак рівності - дані вхідні дані некоректні для симплекс методу")
		self._make_conditions_equalities()
		self.basis = self._get_basis_vectors_nums()
		for i in self.basis:
			if i == -1:
				self._add_artificial_basis()
				break
		self.basis_koef = np.array([0] * len(self.basis))
		for i in range(len(self.basis)):
			self.basis_koef[i] = self.objective_function[self.basis[i]]
		self._make_constants_positive_if_needed()
		safety_counter = 0
		while True:
			safety_counter += 1
			if safety_counter > 100:
				raise SolvingError("Кількість ітерацій завелика, цикл зупинено")

			self._reset_deltas_n_thetas()
			self._calculate_deltas()
			min_delta = min(self.deltas)
			if min_delta < 0:
				self.col_num = self._get_col_num(np.where(self.deltas == min_delta)[0].tolist())
				if self.col_num == -1:
					self.result_error = "unlimited"
					raise SolvingError("Неможливо обрати ведучий стовпчик, всі стовпчики з від'ємними дельта утворюють від'ємні тета:\nЦільова функція необмежена на допустимій області")
				self._calculate_thetas()
				self.row_num = self._find_ind_of_min_theta()
				if self.row_num == -1:
					self.result_error = "unlimited"
					raise SolvingError("Всі тета від'ємні:\nЦільова функція необмежена на допустимій області")
				self._make_basis_column()
				self._set_basis_koef()
			else:
				if self._check_criterion_of_optimality():
					break

		self._cancel_subtitution()
		self._final_preparations()


# ------ Logger class section ------


class Logger:
	"""Загортає інформацію з класу Solver в текстову обгортку для подальшого виведення на екран."""

	def __init__(self, mute):
		self.pointer = ""
		self.inner_log = ""
		self.var_names = []
		self.counters = {
			"a": 0,
			"b": 0,
			"c": 0,
			"x": 0
		}
		self.mute = mute

	def initiate(self, func_name):
		"""Отримує та зберігає назву методу, який має обробити вхідну інформацію та утворити з неї текстову версію."""

		exec("self.pointer = self._{}".format(func_name))
		self.pointer()

	def log(self, **args):
		"""Передає аргументи функції, на яку вказує атрибут pointer."""

		if not self.mute:
			self.pointer(args)

	def draw_table(self, table_info, emphasize_list=[], op=[], row=0):
		"""Утворює html представлення симплекс таблиці та повертає його."""

		table_info = copy.deepcopy(table_info)
		to_emphasize = []
		for i in emphasize_list:
			if i["coords"] == -1:
				for j in range(len(table_info[i["name"]])):
					to_emphasize.append({"name": i["name"], "coords": j})

			else:
				to_emphasize.append({"name": i["name"], "coords": i["coords"]})

		op_strings = ["<td></td>"] * len(table_info["matrix"])
		const = ""
		for i in [x for x in range(len(op)) if x != row]:
			const = str(eval("-Q({})/Q({})".format(op[i], op[row])))
			if const == "0":
				op_strings[i] = "<td>#</td>"
				continue
			if const[0] == '-':
				const = "- " + const[1:]
			else:
				const = "+ " + const
			op_strings[i] = "<td>{} * {}</td>".format(const, self._wrap_variable(table_info["basis"][row])).replace("1 * ", "")
		if len(op) > 0:
			if op[row] < -1:
				op[row] = "({})".format(op[row])
			# &#247; = ÷
			op_strings[row] = "<td>#</td>" if op[row] == 1 else "<td>&#247; {}</td>".format(op[row])
		op_strings = ["<td></td>", "<td></td>"] + op_strings
		for i in range(len(table_info["basis"])):
			table_info["basis"][i] = self._wrap_variable(table_info["basis"][i])

		objective_constants = []
		for i in range(len(table_info["objective_function"])):
			objective_constants.append(table_info["objective_function"][i])
			table_info["objective_function"][i] = self._wrap_variable(i)

		for i in to_emphasize:
			table_info[i["name"]][i["coords"]] = self._emphasize(table_info[i["name"]][i["coords"]])

		first_row = "<td></td><td></td>"
		for i in objective_constants:
			first_row += "<td>{}</td>".format(i)
		first_row += "<td></td><td></td>" + op_strings[0]
		head_row = "<th>Z</th><th>Б</th>"
		for i in range(len(table_info["objective_function"])):
			head_row += "<th>{}</th>".format(table_info["objective_function"][i])
		head_row += "<th>&beta;</th><th>&theta;</th>" + op_strings[1]
		thead = "<tr>{}</tr><tr>{}</tr>".format(first_row, head_row)
		tbody = ""
		for i in range(len(table_info["matrix"])):
			row = "<td>{}</td>".format(table_info["basis_koef"][i])
			row += "<td>{}</td>".format(table_info["basis"][i])
			for j in range(len(table_info["matrix"][i])):
				row += "<td>{}</td>".format(table_info["matrix"][i][j])
			row += "<td>{}</td>".format(table_info["constants"][i])
			row += "<td>{}</td>".format(table_info["thetas"][i])
			row += op_strings[i + 2]
			tbody += "<tr>{}</tr>".format(row)
		last_row = "<td></td><td>&Delta;</td>"
		for i in table_info["deltas"]:
			last_row += "<td>{}</td>".format(i)
		last_row += "<td></td><td></td>"
		tbody += "<tr>{}</tr>".format(last_row)
		table = """
		<table>
			<thead>{}</thead>
			<tbody>{}</tbody>
		</table>""".format(thead, tbody).replace("\n", "")
		return table

	def get_logs(self):
		"""Повертає усі накопичені методами класу записи."""
		reps = {
			"\t": "",
			"\n": "<br>",
			"=>": "&#129046;",
			"<=": "&le;",
			">=": "&ge;",
			"*": "&middot;"
		}
		final_text = self.inner_log
		for k, v in reps.items():
			final_text = final_text.replace(k, v)

		final_text = re.sub(r'\s[\/]\s', ' &#247; ', final_text)
		
		for match in re.finditer(r'[\d]+[\/]', final_text):
			match = match.group()
			final_text = final_text.replace(match, '<div class="frac"><span>' + match[:-1] + '</span><span class="symbol">/</span>@', 1)

		for match in re.finditer(r'[\@][\d]+', final_text):
			match = match.group()
			final_text = final_text.replace(match, '<span class="bottom">' + match[1:] + '</span></div>', 1)
		return final_text


	# --------------- Helpers ---------------


	def _vector_to_math(self, expression, operation, right_part, constant = 0):
		"""Утворює текстове представлення рядка-нерівності.

		Може утворювати представлення цільової функції."""

		text_part = ""
		for i in range(1, len(expression)):
			sign = " + " if expression[i] >= 0 else " - "
			to_add = "{}{}*{}".format(sign, abs(expression[i]), self._wrap_variable(i))
			text_part = text_part + to_add
		sign = " + " if constant >= 0 else " - "
		text_part += "" if constant == 0 else "{}{}".format(sign, abs(constant))
		text_part = "{}*{}{} {} {}".format(expression[0], self._wrap_variable(0), text_part, operation, right_part)
		return text_part

	def _wrap_conditions(self, matrix, ineq, constants, last_cond = ""):
		"""Утворює текстове представлення вихідної системи нерівностей."""

		text_part = ""
		for i in range(len(matrix)):
			text_part += "| " + self._vector_to_math(matrix[i], ineq[i], constants[i]) + "<br>"
		if last_cond != "":
			text_part += "<br>"
			for i in range(len(last_cond)):
				sign, val = (last_cond[i][0], last_cond[i][1]) if last_cond[i][0] != "arbitrary" else ("-", "довільна")
				text_part += "{} {} {}, ".format(self._wrap_variable(i), sign, val)
			return text_part[:-2]
		return text_part

	def _bold(self, string):
		"""Загортає дану в параметрах строку в тег "<b>" та повертає її."""

		return "<b>{}</b>".format(string)

	def _add_entry(self, string):
		"""Додає до основного тексту виконання алгоритму новий запис."""

		self.inner_log += "<div>{}</div>".format(string)

	def _emphasize(self, string):
		"""Загортає дану в параметрах строку в спеціальний тег для подальшого особливого виділення в таблиці."""

		return "<b>{}</b>".format(string)

	def _wrap_variable(self, ind):
		"""Повертає текстове представлення змінної за її індексом в цільовій функції."""

		return "{}<sub>{}</sub>".format(self.var_names[ind][0], self.var_names[ind][1] + 1)

	def _wrap_multiplication(self, m1, m2):
		"""Повертає текстове представлення скалярного добутку двох векторів"""
		text_part = ""
		for i, j in zip(m1, m2):
			if i < 0:
				i = "({})".format(i)
			if j < 0:
				j = "({})".format(j)
			text_part += "{} * {} + ".format(i, j)
		return text_part[:-3]

	def _wrap_vector(self, vect):
		"""Повертає текстове представлення вектора змінних."""

		text_part = ""
		for i in vect:
			text_part += "{}, ".format(i)
		return "({})".format(text_part[:-2])


	# --------------- Pointer functions ---------------


	def _initial_info(self, input_data = ""):
		"""Виведення вхідних даних."""

		if input_data == "":
			text_part = "Отримано наступні вхідні дані для подальшого розв'язку:"
			self._add_entry(self._bold(text_part))
			return
		if input_data["is_max"]:
			self._add_entry("Задача задана на максимум, тому цільова функція буде домножена на -1 для отримання задачі на мінімум")
		input_data = input_data["info"]
		for i in range(len(input_data["objective_function"])):
			self.var_names.append(["x", i])
			self.counters["x"] += 1
		text_part = """Цільова функція: {0} 

		Обмеження:

		{1}
		""".format(
			self._vector_to_math(input_data["objective_function"], "=>", input_data["task_type"]),
			self._wrap_conditions(input_data["matrix"], input_data["inequalities"], input_data["constants"], input_data["last_conditions"])
		)
		self._add_entry(text_part)

	def _normalizing(self, input_data = ""):
		"""Виведення операцій зведення початкових умов до таких, що дозволяють виконання симплекс методу."""

		if input_data == "":
			return
		if "matrix" in input_data:
			text_part = "Заміни виконано. Задача має наступний вигляд:"
			self._add_entry(self._bold(text_part))
			self._add_entry(self._vector_to_math(input_data["objective_function"], "=>", input_data["task_type"], input_data["constant"]))
			self._add_entry(self._wrap_conditions(input_data["matrix"], input_data["inequalities"], input_data["constants"], input_data["last_conditions"]))
			return
		text_part = self._wrap_variable(input_data["index"])
		if input_data["op"] == "a":
			text_part = "a<sub>{}</sub> = -{}".format(
				self.counters["a"] + 1,
				text_part
			)
			self.var_names[input_data["index"]] = ["a", self.counters["a"]]
		elif input_data["op"] == "b":
			constant = input_data["substitution"] * -1
			text_part = "b<sub>{}</sub> = {} {} {}".format(
				self.counters["b"] + 1,
				text_part,
				"+" if constant >= 0 else "-",
				abs(constant)
			)
			self.var_names[input_data["index"]] = ["b", self.counters["b"]]
		elif input_data["op"] == "c":
			text_part = "{} = c<sub>{}</sub> - c<sub>{}</sub>".format(
				text_part,
				self.counters["c"] + 1,
				self.counters["c"] + 2
			)
			self.var_names[input_data["index"]] = ["c", self.counters["c"]]
			self.var_names.append(["c", self.counters["c"] + 1])
			self.counters["c"] += 1

		self.counters[input_data["op"]] += 1
		self._add_entry("Виконаємо заміну: " + text_part)
	
	def _inequalities(self, input_data = ""):
		"""Виведення операцій зведення вихідної системи нерівностей до системи рівностей."""

		if input_data == "":
			return

		for i in range(len(input_data["matrix"][0]) - len(self.var_names)):
			self.var_names.append(["x", self.counters["x"]])
			self.counters["x"] += 1

		text_part = self._bold("Перетворюємо всі нерівності у рівності:")
		self._add_entry(text_part)
		self._add_entry(self._vector_to_math(input_data["objective_function"], "=>", input_data["task_type"], input_data["constant"]))
		self._add_entry(self._wrap_conditions(input_data["matrix"], input_data["inequalities"], input_data["constants"], input_data["last_cond"]))

	def _artificial_basis(self, input_data = ""):
		"""Виведення операцій утворення штучного базису."""

		if input_data == "":
			return

		for i in range(len(input_data["matrix"][0]) - len(self.var_names)):
			self.var_names.append(["x", self.counters["x"]])
			self.counters["x"] += 1

		text_part = self._bold("Матриця задачі не міститиме одиничної підматриці, тому вводимо штучний базис:<br>")
		self._add_entry(text_part)
		text_part = """Для введення штучного базису використовуємо М-метод. Знаходимо найбільший множник серед усіх змінних та додаємо до нього одиницю.
		Отримуємо М = {}

		Цільова функція набуває вигляду: {}

		Загальні умови:

		{}""".format(
			input_data["m"],
			self._vector_to_math(input_data["objective_function"], "=>", input_data["task_type"], input_data["constant"]),
			self._wrap_conditions(input_data["matrix"], input_data["inequalities"], input_data["constants"], input_data["last_cond"])
		)
		self._add_entry(text_part)

	def _show_basis(self, input_data = ""):
		"""Виведення обраних для входження в базис змінних."""

		if input_data == "":
			return
		text_part = self._bold("Шукаємо набір змінних для базису:")
		self._add_entry(text_part)
		if input_data["basis"] == None:
			text_part = "Матриця не містить одиничної матриці, базис вибрати не можна"
		else:
			text_part = "В якості базису обираємо наступні змінні: "
			for i in input_data["basis"]:
				text_part += "{}, ".format(self._wrap_variable(i))
			text_part = text_part[:-2]
		self._add_entry(text_part)

	def _initial_table(self, input_data = ""):
		"""Виведення початкової симплекс таблиці."""

		if input_data == "":
			return
		text_part = self._bold("Проаналізувавши початкові умови, можемо записати симплекс таблицю:")
		self._add_entry(text_part)
		self._add_entry(self.draw_table(input_data["table"]))

	def _deltas(self, input_data = ""):
		"""Виведення розрахунку відносних оцінок "дельта"."""

		if input_data == "":
			text_part = self._bold("Проводимо розрахунок відносних оцінок змінних - \"дельта\":")
			self._add_entry(text_part)
			return
		elif "const" in input_data:
			self._add_entry("{}: {} - ({}) = {}".format(
				self._wrap_variable(input_data["index"]),
				input_data["const"],
				self._wrap_multiplication(input_data["mult1"], input_data["mult2"]),
				input_data["res"]
			))
			return
		self._add_entry("Отримали наступну таблицю:")
		self._add_entry(self.draw_table(input_data["table"], [{"name": "deltas", "coords": -1}]))

	def _get_col(self, input_data = ""):
		"""Виведення операцій вибору ведучого стовпчика."""

		if input_data == "":
			return
		text_part = self._bold("Обираємо ведучий стовпчик:")
		self._add_entry(text_part)
		text_part = ""
		if "num" in input_data:
			text_part = "Обираємо стовпчик з мінімальною від'ємною оцінкою \"дельта\", тому ведучим може бути стовпчик, що відповідає змінній {}, обираємо його".format(self._wrap_variable(input_data["num"]))
		elif "no_col" in input_data and input_data["no_col"] == True:
			text_part = "Можливий ведучий стовпчик відсутній"

		self._add_entry(text_part)

	def _thetas(self, input_data = ""):
		"""Виведення розрахунку відношень "тета"."""

		if input_data == "":
			text_part = self._bold("Проводимо розрахунок відношень між елементами векторів вільних членів та ведучого стовпчика - \"тета\":")
			self._add_entry(text_part)
			return
		if "table" in input_data:
			text_part = "Отримали таку матрицю:"
			self._add_entry(text_part)
			self._add_entry(self.draw_table(input_data["table"], [{"name": "thetas", "coords": -1}]))
			return
		parsed_div2 = input_data["div2"] if input_data["div2"] >= 0 else "({})".format(input_data["div2"])
		text_part = "{}: {} / {} ".format(self._wrap_variable(input_data["ind"]), input_data["div1"], parsed_div2)
		if "error" in input_data:
			if input_data["error"] == "zerodiv":
				text_part += "- Відношення містить ділення на нуль, не розраховуємо. Встановимо значення відношення рівним -1"
			elif input_data["error"] == "negative":
				text_part += "- Даний рядок не може бути ведучим через те, що відповідний йому елемент стовпчика менший або рівний нулю. Встановимо значення відношення рівним -1"
			self._add_entry(text_part)
			return
		text_part += "= " + str(input_data["res"])
		self._add_entry(text_part)

	def _basis_col(self, input_data = ""):
		"""Виведення операцій переходу до іншого базису."""

		if input_data == "":
			text_part = self._bold("Переходимо до іншого базису:")
			self._add_entry(text_part)
			return
		text_part = "В базисі замінюємо змінну {} на {}, а також утворюємо одиничний вектор у ведучому стовпчику відносно ведучого елемента:".format(
			self._wrap_variable(input_data["p_table"]["basis"][input_data["row"]]),
			self._wrap_variable(input_data["col"])
		)
		self._add_entry(text_part) 
		if input_data["op"][input_data["row"]] == 1:
		
			for i in [x for x in range(len(input_data["op"])) if x != input_data["row"]]:
				if input_data["op"][i] != 0:
					break
			else:
				text_part = "Ведучий стовпчик вже містить коректний одиничний вектор, перетворення не потрібні"
				self._add_entry(text_part)
				return
		self._add_entry(self.draw_table(
			input_data["p_table"],
			[
				{"name": "matrix", "coords": (input_data["row"], input_data["col"])},
				{"name": "basis", "coords": input_data["row"]},
				{"name": "objective_function", "coords": input_data["col"]}
			],
			input_data["op"],
			input_data["row"]
		))

	def _min_theta(self, input_data = ""):
		"""Виведення операцій вибору ведучого рядка."""

		if input_data == "":
			text_part = self._bold("Виконуємо пошук ведучого рядка:")
			self._add_entry(text_part)
			return
		if input_data["ind"] == -1:
			text_part = "Ведучий рядок обрати неможливо"
		else:
			text_part = "Шукаємо рядок з мінімальною оцінкою \"тета\", тому обираємо рядок, що відповідає змінній {}".format(self._wrap_variable(input_data["ind"]))
		self._add_entry(text_part)

	def _optimality(self, input_data = ""):
		"""Виведення перевірки коректності виконання критерію оптимальності."""

		if input_data == "":
			text_part = self._bold("Критерій оптимальності досягнуто:")
			self._add_entry(text_part)
			return
		if input_data["another_iteration"]:
			text_part = "Серед вільних членів наявні від'ємні. Перевіряємо рядки з від'ємними членами на наявність від'ємних елементів:"
			self._add_entry(text_part)
			text_part = "Рядок, що відповідає змінній {}, містить від'ємний елемент, тому обираємо відповідний цьому елементу стовпчик змінної {} ведучим".format(
				self._wrap_variable(input_data["ind_row"]),
				self._wrap_variable(input_data["ind_col"])
			)
		else:
			text_part = "Всі вільні члени задовольняють умовам задачі, тому виконання алгоритму завершено."
		self._add_entry(text_part)

	def _substitution(self, input_data = ""):
		"""Виведення результатів скасування замін змінних."""
		
		if input_data == "":
			text_part = self._bold("Скасовуємо заміни та переходимо до початкових змінних:")
			self._add_entry(text_part)
			text_part = "Змінні, що знаходяться в базисі, мають значення відповідних їм вільних членів:"
			self._add_entry(text_part)
			return

		if "ind" in input_data and "val" in input_data:
			text_part = "{} = {}".format(self._wrap_variable(input_data["ind"]), input_data["val"])
			self._add_entry(text_part)
			return

		if "max" in input_data and input_data["max"] == True:
			text_part = "Початкові умови містили задачу на максимум, тому значення цільової функції домножимо на -1."
			self._add_entry(text_part)
			return

		if "sub_queue" in input_data:
			text_part = ""
			for i in range(len(input_data["sub_queue"])):
				r_part = ""
				var = self._wrap_variable(input_data["sub_queue"][i][0])
				if input_data["sub_queue"][i][1][0] == "+":
					op = input_data["sub_queue"][i][1][2:] if input_data["sub_queue"][i][1][2] == "-" else "+{}".format(input_data["sub_queue"][i][1][2:])
					sign, num = op[0], op[1:]
					if i + 1 < len(input_data["sub_queue"]) and input_data["sub_queue"][i][0] == input_data["sub_queue"][i + 1][0]:
						sign = "-" if sign == "+" else "+"
						var = "-" + var
						i += 1
					r_part = "{} {} {}".format(var, sign, num)
				elif i == 0 or input_data["sub_queue"][i][0] == input_data["sub_queue"][i - 1][0]:
					continue
				else:
					var = "-" + var
					r_part = "{}".format(var)
				text_part = "x<sub>{}</sub> = {}".format(input_data["sub_queue"][i][0] + 1, r_part)
				self._add_entry(text_part)
			return
		elif "arb1" in input_data:
			text_part = "x<sub>{}</sub> = {} - {}".format(
				input_data["arb1"] + 1, 
				self._wrap_variable(input_data["arb1"]),
				self._wrap_variable(input_data["arb2"])
			)
			self._add_entry(text_part)

	def _final(self, input_data = ""):
		"""Виведення результатів виконання алгоритму."""
		
		if input_data == "":
			text_part = self._bold("Виконання алгоритму завершено. Виводимо результат та перевіряємо його на коректність:")			
			self._add_entry(text_part)
			return

		if "big_vect" in input_data:
			text_part = "Результат пройшов перевірку на коректність."
			self._add_entry(text_part)
			text_part = """
			Нарешті, виводимо кінцевий результат:

			Вектор з усіма змінними: {}
			Вектор з шуканими змінними: {}
			Значення цільової функції: {}""".format(
				self._wrap_vector(input_data["big_vect"]),
				self._wrap_vector(input_data["vect"]),
				input_data["obj_val"]
			)
			self._add_entry(text_part)


# ------ Custom exception section ------


class SolvingError(Exception):
	"""Клас винятків, пов'язаних з помилками виконання алгоритму."""

	def __init__(self, message):
		super().__init__(message)


# ------ Test section ------


test_input_string = """
# Not suitable for calculations 

x[2]+x[3]-2x[1]+223x[4] =>max

|2x[2]+x[1]-3x[3]-x[4]>4
|-2x[3]+x[1]+3x[2]<=0
|-x[2]+10x[3]-4x[1]<-7
|x[1]+10x[3]-4x[2]>=7/2

x[1]>0, x[2]>=0, x[3]<3/2, x[4]<=2
"""

import unittest
class TestParserMethods(unittest.TestCase):
	"""Tests for parsing class"""
	def test_math_formatting(self):
		"""Tests for valid expression formatting into a math form"""
		self.assertEqual(InputParser._format_to_math_form("- 9x[4] + 23x[1] -6x[2]+x[3] - x[5]=>max"), '-9x[4]+23x[1]-6x[2]+1x[3]-1x[5]=>max')

	def test_input(self):
		"""Tests for valid parsing of the input file"""
		dummy = InputParser({'data_type': 'string','data':test_input_string})
		test_dict = {
			"objective_function": np.array([Q(-2, 1), Q(1, 1), Q(1, 1), Q(223, 1)]),
			"task_type": "max",
			"last_conditions": [(">", Q(0, 1)), (">=", Q(0, 1)), ("<", Q(3, 2)), ("<=", Q(2, 1))],
			"matrix": np.array([
				[1, 2, -3, -1],
				[1, 3, -2, 0], 
				[-4, -1, 10, 0], 
				[1, -4, 10, 0]
			]),
			"inequalities": [">", "<=", "<", ">="],
			"constants": np.array([Q(4, 1), Q(0, 1), Q(-7, 1), Q(7, 2)])
		}
		for k, v in test_dict.items():
			self.assertTrue(np.array_equal(v, dummy.get_data()[k]))

class TestCommonLinearMethods(unittest.TestCase):
	"""Тести для класу Solver."""

	def __init__(self, *args, **kwargs):
		super(TestCommonLinearMethods, self).__init__(*args, **kwargs)
		self.input_info = {'data_type': 'string','data':test_input_string, "mute": True}
		self.input_info_main = {"data_type": "file", "data": "input", "mute": True}

	def test_making_unit_basis(self):
		"""Тест на перевірку коректної роботи методу зведення стовпчика до одиничного вектора."""

		dummy = SimplexSolver(self.input_info)
		dummy._make_basis_column()
		test_matrix = np.array([
		 	[1, 2, -3, -1],
		 	[0, 1, 1, 1],
		 	[0, 7, -2, -4],
		 	[0, -6, 13, 1]
		 ])
		self.assertTrue(np.array_equal(test_matrix, dummy.matrix))

	def test_making_equalities_in_conditions(self):
		"""Тест на перевірку коректної роботи методу зведення нерівностей умов до рівностей."""

		dummy = SimplexSolver(self.input_info)
		for i in range(len(dummy.inequalities)):
			if len(dummy.inequalities[i]) == 1:
				dummy.inequalities[i] = ">=" if i % 2 == 0 else "<="
		before_test = dummy.matrix
		dummy._make_conditions_equalities()
		self.assertTrue(len(before_test[0]) + 4, len(dummy.matrix[0]))
		self.assertTrue(np.array_equal(np.array([
			[-1, -2, 3, 1, 1, 0, 0, 0],
			[1, 3, -2, 0, 0, 1, 0, 0],
			[4, 1, -10, 0, 0, 0, 1, 0],
			[-1, 4, -10, 0, 0, 0, 0, 1]
		]), dummy.matrix))

	def test_getting_basis_vectors_nums(self):
		"""Тест на перевірку коректної роботи методу отримання номерів змінних, що входять в базис."""

		dummy = SimplexSolver(self.input_info)
		correct_matrix = np.array([
			[2, 0, 0, 1],
			[2, 0, 1, 0],
			[2, 1, 0, 0]
		])
		incorrect_matrix = np.array([
			[3, 0, 0, 0],
			[3, 1, 1, 0],
			[3, 1, 2, 0],
			[3, 1, 0, 0]
		])
		dummy.matrix = correct_matrix
		self.assertTrue(np.array_equal(np.array([3, 2, 1]), dummy._get_basis_vectors_nums()))
		dummy.matrix = incorrect_matrix
		self.assertTrue(np.array_equal(np.array([-1, -1, -1, -1]), dummy._get_basis_vectors_nums()))

	def test_changing_variable_in_basis(self):
		"""Тест на перевірку коректної заміни змінної в базисі."""

		dummy = SimplexSolver(self.input_info)
		dummy.basis = [0, 1]
		dummy.basis_koef = [3, 3]
		dummy.objective_function = [3, 3, 4]
		dummy.col_num = 2
		dummy.row_num = 1
		dummy._set_basis_koef()
		self.assertTrue(np.array_equal([0, 2], dummy.basis))
		self.assertTrue(np.array_equal([3, 4], dummy.basis_koef))

	def test_objective_function_expanding(self):
		"""Тест на коректне додання змінної до цільової функції."""

		dummy = SimplexSolver(self.input_info)
		dummy.objective_function = [1, 1]
		new_matrix = np.array([
			[2, 2, 1, 0],
			[2, 2, 0, 1]
		])
		dummy.matrix = new_matrix
		dummy._expand_objective_function_if_needed()
		self.assertTrue(np.array_equal([1, 1, 0, 0], dummy.objective_function))

class TestSimplexMethod(unittest.TestCase):
	"""Тести для класу SimplexSolver."""

	def __init__(self, *args, **kwargs):
		super(TestSimplexMethod, self).__init__(*args, **kwargs)
		self.input_info = {'data_type': 'string','data':test_input_string, "mute": True}
		self.input_info_main = {"data_type": "file", "data": "input", "mute": True}

	def test_calculating_deltas(self):
		"""Тест на правильне розрахування відносних оцінок."""

		dummy = SimplexSolver(self.input_info_main)
		dummy.matrix = np.array([
			[2, 2, 1, 0],
			[2, 2, 0, 1]
		])
		dummy.objective_function = np.array([4, 6, 0, 1])
		dummy.basis_koef = np.array([1, 1])
		dummy._calculate_deltas()
		self.assertTrue(np.array_equal([0, 2, -1, 0], dummy.deltas))

	def test_calculating_thetas(self):
		"""Тест на правильне розрахування вектору з відношеннями "тета"."""

		dummy = SimplexSolver(self.input_info_main)
		dummy.matrix = np.array([
			[-1, 2, 1, 0],
			[3, 1, 0, 1]
		])
		dummy.row_num = 0
		dummy.constants = np.array([-2, 0])
		dummy.basis = [2, 3]
		dummy._calculate_thetas()
		self.assertTrue(np.array_equal([2, 0], dummy.thetas))

	def test_finding_min_theta(self):
		"""Тест на пошук індекса ведучого рядка."""

		dummy = SimplexSolver(self.input_info_main)
		dummy.basis = [1, 2, 3]
		incorrect_thetas = np.array([-1, -2, -3])
		correct_thetas = np.array([1, -2, 3])
		dummy.thetas = incorrect_thetas
		self.assertEqual(-1, dummy._find_ind_of_min_theta())
		dummy.thetas = correct_thetas
		self.assertEqual(0, dummy._find_ind_of_min_theta())

	def test_for_choosing_column(self):
		"""Тест на коректний вибір ведучого стовпчика."""

		dummy = SimplexSolver(self.input_info_main)
		correct_matrix = np.array([
			[1, -1, -2],
			[0, -1, -2],
			[0, -1, 2]
		])
		incorrect_matrix = np.array([
			[1, -1, -2],
			[0, -1, -2],
			[0, -1, -2]
		])
		dummy.constants = [0, 1, 1]
		dummy.matrix = correct_matrix
		self.assertEqual(2, dummy._get_col_num([1, 2]))
		dummy.matrix = incorrect_matrix
		self.assertEqual(-1, dummy._get_col_num([1, 2]))

	def test_making_constants_positive(self):
		"""Тест на перехід до нового базису якщо дельта не менші нуля, але серед вільних членів є від'ємні."""

		info1 = """
		3x[1] + 3x[2] => min
		|x[1] + 4x[2] >=4
		|4x[1] + x[2] >=4 
		x[1]>=0,x[2]>=0
		"""
		info2 = """
		3x[1] + 3x[2] => min
		|x[1] + 4x[2] >=4
		|4x[1] + x[2] <=4 
		x[1]>=0,x[2]>=0
		"""
		dummy = SimplexSolver({"data_type": "string", "data": info1, "mute": True})

		dummy.initial_variables_quantity = len(dummy.matrix[0])
		dummy._normalize_conditions()
		dummy._make_conditions_equalities()
		dummy.basis = dummy._get_basis_vectors_nums()
		for i in dummy.basis:
			if i == -1:
				dummy._add_artificial_basis()
				break
		dummy.basis_koef = np.array([0] * len(dummy.basis))
		dummy._expand_objective_function_if_needed()
		for i in range(len(dummy.basis)):
			dummy.basis_koef[i] = dummy.objective_function[dummy.basis[i]]

		old_matrix = np.array([
			[-1, -4, 1, 0],
			[-4, -1, 0, 1]
		])
		dummy._make_constants_positive_if_needed()
		self.assertFalse(np.array_equal(old_matrix, dummy.matrix))
		
		old_matrix[1][:2] *= -1
		dummy = SimplexSolver({"data_type": "string", "data": info2, "mute": True})

		dummy.initial_variables_quantity = len(dummy.matrix[0])
		dummy._normalize_conditions()
		dummy._make_conditions_equalities()
		dummy.basis = dummy._get_basis_vectors_nums()
		for i in dummy.basis:
			if i == -1:
				dummy._add_artificial_basis()
				break
		dummy.basis_koef = np.array([0] * len(dummy.basis))
		dummy._expand_objective_function_if_needed()
		for i in range(len(dummy.basis)):
			dummy.basis_koef[i] = dummy.objective_function[dummy.basis[i]]

		dummy._make_constants_positive_if_needed()
		self.assertTrue(np.array_equal(old_matrix, dummy.matrix))

	def test_adding_artificial_basis(self):
		"""Тест на коректне додання одиничної підматриці до основної."""

		dummy = SimplexSolver(self.input_info_main)
		dummy.matrix = np.array([
			[2, 2],
			[3, 3]
		])
		dummy.objective_function = np.array([1, 1])
		dummy.constants = np.array([4, 4])
		dummy.basis = [-1, -1]
		dummy._add_artificial_basis()
		self.assertTrue(np.array_equal([1, 1, 5, 5], dummy.objective_function))
		self.assertTrue(np.array_equal([
			[2, 2, 1, 0],
			[3, 3, 0, 1]
		], dummy.matrix))
		self.assertTrue(np.array_equal([2, 3], dummy.artificial_variables))

	def test_normalizing_conditions(self):
		"""Тест на коректне зведення змінних до невід'ємних."""

		info = """
		2x[1]+x[2]=>max
		|x[1]+x[2]>=-5
		|2x[1]+2x[2]<=10
		x[1]<=2
		"""
		dummy = SimplexSolver({"data_type": "string", "data": info, "mute": True})
		dummy.initial_variables_quantity = len(dummy.matrix[0])
		dummy._normalize_conditions()
		correct_matrix = np.array([
			[-1, 1, -1],
			[-2, 2, -2]
		])
		self.assertTrue(np.array_equal(correct_matrix, dummy.matrix))
		self.assertTrue(np.array_equal([2, -1, 1], dummy.objective_function))
		self.assertTrue(np.array_equal([-7, 6], dummy.constants))
		self.assertTrue(np.array_equal([(1, 2)], dummy.arbitrary_pairs))
		self.assertTrue(np.array_equal([(0, '+=-2'), (0, '*=-1')], dummy.substitution_queue))

	def test_for_correct_solving(self):
		"""Тест на правильне розв'язання різних задач симплекс методом."""

		with open("test_input") as f:
			inner_text = f.read()
			inner_text = inner_text.split("***")
			for i in inner_text:
				dummy = SimplexSolver({"data_type":"string", "data": i, "mute":True})
				try:
					dummy.solve()
				except SolvingError as err:
					self.assertEqual(dummy.expected_error, dummy.result_error)
				else:
					self.assertEqual(dummy.expected_result, dummy.result)
	
	def test_substitution(self):
		"""Перевірка на коректне повернення значень змінних після заміни."""

		dummy = SimplexSolver(self.input_info_main)
		dummy.substitution_queue = [(0, '+=-8'), (0, '*=-2')]
		dummy.objective_function = np.array([2, 2])
		dummy.final_result = [Q(0)] * 2
		dummy.arbitrary_pairs = []
		dummy._cancel_subtitution()
		self.assertTrue(np.array_equal([-4, 2], dummy.objective_function))

	def test_criterion(self):
		"""Перевірка на коректне виконання критерію оптимальності."""

		dummy = SimplexSolver(self.input_info_main)
		dummy.deltas = [1, 2, 3]
		dummy.constants = [5, 6]
		dummy.basis = [0, 1]
		dummy.basis_koef = [5, 5]
		self.assertTrue(dummy._check_criterion_of_optimality())
		
		dummy.constants = [5, -6]
		self.assertFalse(dummy._check_criterion_of_optimality())
		
		dummy.constants = [-5, 6]
		dummy.matrix = np.array([
			[1, 2],
			[-1, 2]
		])
		self.assertRaises(SolvingError, dummy._check_criterion_of_optimality)
		self.assertEqual("empty", dummy.result_error)

	def test_reset_deltas_n_thetas(self):
		"""Перевірка на правильне повернення векторів "дельта" та "тета" до вихідних значень."""
		dummy = SimplexSolver(self.input_info_main)
		dummy.thetas = [2, 2]
		dummy.deltas = [3, 3, 3]
		dummy._reset_deltas_n_thetas()
		self.assertTrue(np.array_equal(["-"] * 2, dummy.thetas))
		self.assertTrue(np.array_equal(["-"] * 3, dummy.deltas))

	def test_raising_exceptions(self):
		"""Перевірка на правильне виконання винятків у випадку помилки алгоритму."""

		dummy = SimplexSolver(self.input_info_main)
		dummy.basis = [0, 2]
		dummy.arbitrary_pairs = [(0, 3)]
		dummy.objective_function = np.array([2, 3, 4, -2])
		dummy.deltas = [1, 0, 1, 1]
		dummy.result = 8
		self.assertRaises(SolvingError, dummy._check_for_ambiguous_result)
		self.assertEqual("infinite|8", dummy.result_error)
		
		dummy.artificial_variables = [1, 2]
		dummy.final_result = [0, 2]
		self.assertRaises(SolvingError, dummy._check_for_empty_allowable_area)
		self.assertEqual("empty", dummy.result_error)

	def test_final_preparations(self):
		"""Тест на правильний запис результатів виконання алгоритму у відповідні атрибути."""

		dummy = SimplexSolver(self.input_info_main)
		dummy.final_result = np.array([1, 2, 3])
		dummy.objective_function = np.array([2, -3, 4])
		dummy.initial_variables_quantity = 2
		dummy.arbitrary_pairs = []
		dummy.deltas = [1, 1, 1]
		dummy._final_preparations()
		self.assertEqual(-4, dummy.result)


if __name__ == "__main__":
	# unittest.main()
	data_to_solve = """
	4x[1] +4x[2] +2x[3] +4x[4] +3x[5] +x[6]=>max

	|-3x[1] + 2x[3] + 3x[4] + 3x[5] + 4x[6] = 1
	|3x[1] - 2x[2] + x[3] + 4x[4] + 3x[5] - 3x[6] = 2
	|4x[1] - 2x[2] + x[3] - 4x[4] - x[5] - x[6] = 2
	|2x[1] + 3x[2] + 3x[3] + x[4] + 2x[5] - 3x[6] = 3 

	x[1]>=0,x[2]>=0,x[3]>=0,x[4]>=0,x[5]>=0,x[6]>=0
	"""
	data_to_solve1 = """
		x[1]=>max
	|x[1]>=0
	
	x[1]<=5
	"""
	dummy = SimplexSolver({"data_type":"string", "data": data_to_solve1, "mute":False})
	f = open("output.html", "w")
	errors = ""
	try:
		dummy.solve()
	except SolvingError as err:
		errors = err
	else:
		print("OK")
	f.write("{}<br><br>{}".format(dummy.get_result(), errors))
