from django.shortcuts import render

# Create your views here.
from django.http import HttpResponse
from django.template import loader

from solver.modules.linearsolver import SimplexSolver, SolvingError

def index(request):
	return render(request, 'solver/index.html')

def result(request):
	data = request.GET["input_text"]
	reps = {"\r": "", "\t": "", " ": ""}
	for k, v in reps.items():
		data = data.replace(k, v)
	input_info = {"data_type": "string", "data": data, "mute": False}
	solver = SimplexSolver(input_info)
	errors = ""
	try:
		solver.solve()
	except SolvingError as err:
		errors = str(err).replace("\n", "<br>")

	context = { 
		'result': "{}<p>{}</p>".format(solver.get_result(), errors)
	}
	return render(request, 'solver/result.html', context)