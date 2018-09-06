from django.shortcuts import render

# Create your views here.
from django.http import HttpResponse
from django.template import loader

from solver.modules.linearsolver import SimplexSolver, SolvingError

def index(request):
	return render(request, 'solver/index.html')

def result(request):
	# data = request.GET["input_text"]
	# reps = {"\r": "", "\t": "", " ": ""}
	# for k, v in reps.items():
	# 	data = data.replace(k, v)
	# input_info = {"data_type": "string", "data": data, "mute": False}
	
	data = request.GET
	parsed_data = {}

	parsed_data["obj_func"] = data["obj_func"][:-1].split(" ")
	parsed_data["ineq"] = data["ineq"][:-1].split(" ")
	parsed_data["constants"] = data["constants"][:-1].split(" ")
	parsed_data["task_type"] = data["task_type"]
	parsed_data["matrix"] = []
	for i in data["matrix"][:-1].split("|"):
		parsed_data["matrix"].append(i[:-1].split(" "))

	parsed_data["last_cond"] = []
	for i in data["last_cond"][:-1].split("|"):
		temp = i.split(" ")
		parsed_data["last_cond"].append([temp[0], temp[1]])

	solver = SimplexSolver("object", parsed_data)
	
	context = { 
		'result': solver.get_result()
	}
	return render(request, 'solver/result.html', context)