$(document).ready(function(){
	String.prototype.format = function() {
	    var s = this,
	        i = arguments.length;

	    while (i--) {
	        s = s.replace(new RegExp('\\{' + i + '\\}', 'gm'), arguments[i]);
	    }
	    return s;
	};

	$.fn.textWidth = function(text, font) {
    
	    if (!$.fn.textWidth.fakeEl) $.fn.textWidth.fakeEl = $('<span>').hide().appendTo(document.body);
	    
	    $.fn.textWidth.fakeEl.text(text || this.val() || this.text() || this.attr('placeholder')).css('font', font || this.css('font'));
	    
	    return $.fn.textWidth.fakeEl.width();
	};

	function triggerFunctionInput(){
		isBelowZero = this.value[0] === "-" ? 1 : 0;
		if (this.value.length > 8)
	        this.value = this.value.slice(0, 8 + isBelowZero); 
	    var inputWidth = $(this).textWidth();
	    $(this).css({
	        width: inputWidth
	    })
	}

	$('input[type="number"]').on('input', triggerFunctionInput).trigger('input');

	counters = {
		"obj": 2,
		"matr1": 2,
		"matr2": 2,
		"last": 2
	}

	function eventAddDoublecell(){
		el = $(this).parent().parent().children('ul')
		type = el.find("li:first-child div input").attr("name").split("_")[0]

		var doubleCell = `
		<li>
			<div class="divider">+</div>
			<div class="double-cell">
				<input max="999999" type="number" name="{2}_{0}" placeholder="0">
				<div>x<sub>{1}</sub></div>
			</div>
		</li>`.format(++counters[type], counters[type], type)
		el.find("li:last-child").before(doubleCell)
		$('input[type="number"]').off().on('input', triggerFunctionInput).trigger('input');
		adjustLastConditions()
		if(counters[type] === 2)
			$(this).parent().children('button:disabled').prop("disabled", false)
	}

	function eventDeleteDoublecell(){
		el = $(this).parent().parent().find('ul li')
		type = el.find("div input").attr("name").split("_")[0]
		if(counters[type] === 1)
			return
		el = el.parent().children("li:nth-last-child(2)")
		el.remove()
		counters[type]--;
		adjustLastConditions()
		if(counters[type] === 1)
			$(this).prop("disabled", true)
	}

	function adjustLastConditions(){
		maxQuantity = 0;
		for (var key in counters) {
		    if (counters.hasOwnProperty(key) && key !== "last") {
		        if(counters[key] > maxQuantity)
		        	maxQuantity = counters[key]
		    }
		}
		if(counters["last"] < maxQuantity){
			block = `
			<li>
				<div class="triple-cell">
					<div>x<sub>{0}</sub></div>
					<select>
	    				<option value=">=">&ge;</option>
	    				<option value="<=">&le;</option>
	    				<option value="arbitrary">?</option>
	    			</select>
	    			<input type="number" max="999999" name="constant" placeholder="0">
				</div>
			</li>`.format(++counters["last"]);
			$("#lastConditions").append(block);
			$('input[type="number"]').off().on('input', triggerFunctionInput).trigger('input');
		}
		else if(counters["last"] > maxQuantity){
			el = $("#lastConditions li:last-child");
			el.remove();
			counters["last"]--;
		}
		$("#lastConditions select").off().change(selectEvent)
	}

	rowCount = 2;
	function eventAddRow(){
		rowEl = `
		<section>
    		<ul>
	    		<li>
	    			<div class="double-cell">
		    			<input type="number" max="999999" name="matr{0}_1" placeholder="0" value="">
		    			<div>x<sub>1</sub></div>
		    		</div>
		    	</li>
		    	<li>
		    		<div class="divider">+</div>
		    		<div class="double-cell">
		    			<input type="number" max="999999" name="matr{1}_2" placeholder="0" value="">
		    			<div>x<sub>2</sub></div>
		    		</div>
	    		</li>
	    		<li class="double-cell">
	    			<select>
	    				<option value=">=">&ge;</option>
	    				<option value="<=">&le;</option>
	    				<option value="=">=</option>
	    			</select>
	    			<input type="number" max="999999" name="constant" placeholder="0" value="">
	    		</li>
    		</ul>
    		<div class="quantity-control">
    			<button type="button" class="add-el">+</button>
    			<button type="button" class="del-el">-</button>
    		</div>
    	</section>`.format(++rowCount, rowCount)
    	counters["matr" + rowCount] = 2;
    	$("#matrControl").before(rowEl)
    	$("section .add-el").off().click(eventAddDoublecell)
		$("section .del-el").off().click(eventDeleteDoublecell)
		$('input[type="number"]').off().on('input', triggerFunctionInput).trigger('input');

		adjustLastConditions()
		if(rowCount == 2)
			$(this).parent().children(".del-el").prop("disabled", false)
	}

	function eventDelRow(){
		var container = $("#mainConditions section")
		if(rowCount > 1)
			var row = $(container[container.length - 1]);
			matrRowName = row.find("input").attr("name").split("_")[0];
			counters[matrRowName] = 0;
			adjustLastConditions()
			row.remove()
			--rowCount;
		if(rowCount === 1)
			$(this).prop("disabled", true)
	}

	function selectEvent(){
		el = $(this)
		if (el.val() === "arbitrary") {
			el.parent().find("input").prop('disabled', true);
			el.parent().find("input").val(0)
			el.parent().find("input").trigger('input');
		}
		else {
			if (el.parent().find("input").attr('disabled'))
				el.parent().find("input").prop("disabled", false)
		}
	}

	$("section .add-el").click(eventAddDoublecell)
	$("section .del-el").click(eventDeleteDoublecell)
	$("#matrControl .add-el").click(eventAddRow)
	$("#matrControl .del-el").click(eventDelRow)
	$("#lastConditions select").change(selectEvent)

	$("#submit").click(function(){
		var container = $("#objectiveFunc input[type='number']")
		var objString = "";
		for(var i = 0; i < container.length; i++) {
			if(container[i].value === "")
				$(container[i]).val(0)
			objString += container[i].value + " "
		}
		container = $("#mainConditions section")
		var matrString = "";
		var ineqString = "";
		var constString = "";
		for(var i = 0; i < container.length; i++) {
			var subCont = $(container[i]).find("input[type='number']")
			for(var j = 0; j < subCont.length; j++) {
				if(subCont[j].value === "")
					$(subCont[j]).val(0)
				if(j !== subCont.length - 1)
					matrString += subCont[j].value + " "
			}
			ineqString += $(container[i]).find("select").val() + " "
			constString += $(subCont[subCont.length - 1]).val() + " "
			matrString += "|"
		}
		container = $("#lastConditions .triple-cell")
		var lastString = ""
		for(var i = 0; i < container.length; i++) {
			if($(container[i]).find("input[type='number']").val() === "")
				$(container[i]).find("input[type='number']").val(0)
			lastString += $(container[i]).find("select").val() + " " + $(container[i]).find("input[type='number']").val() + "|"
		}

		var taskString = $("#objectiveFunc select").val()

		$("input[name='obj_func']").val(objString)
		$("input[name='matrix']").val(matrString)
		$("input[name='ineq']").val(ineqString)
		$("input[name='constants']").val(constString)
		$("input[name='last_cond']").val(lastString)
		$("input[name='task_type']").val(taskString)

		$("#finalForm").submit()
	});

	var examples = {
		"regular": [
			[[-1, 1], "max"],
			[[0, 1], ">=", 0],
			[[1, 1], "<=", 1],
			[[1, -4], ">=", -2],
			[["<=", 40], [">=", -100]]
		],
		"big": [
			[[4, 4, 2, 4 ,3, 1], "max"],
			[[-3, 0, 2, 3, 3, 4], "=", 1],
			[[3, -2, 1, 4, 3, -3], "=", 2],
			[[4, -2, 1, -4, -1, -1], "=", 2],
			[[2, 3, 3, 1, 2, -3], "=", 3],
			[[">=", 0],[">=", 0],[">=", 0],[">=", 0],[">=", 0],[">=", 0]]
		],
		"error": [
			[[1, 1], "max"],
			[[1, 1], "<=", 1],
			[[1, 1], ">=", 1],
			[["arbitrary", 0], ["arbitrary", 0]]
		],
		"small": [
			[[1, 1], "max"],
			[[1, 1], "<=", 1],
			[[1, 2], ">=", 1],
			[["arbitrary", 0], ["arbitrary", 0]]
		]
	}

	function setExample(exampleName){
		var conditions = examples[exampleName]
		var systemSize = conditions.length - 2
		var objSize = conditions[0][0].length
		var cont = $("#objectiveFunc")
		
		// Setting objective function
		while(cont.find("input[type='number']").length > 1)
			cont.parent().find(".del-el").click()

		for (var i = 1; i < objSize; ++i)
			cont.parent().find(".add-el").click()

		var subCont = cont.find("input")
		
		for (var i = 0; i < objSize; ++i)
			$(subCont[i]).val(conditions[0][0][i])

		$(cont.find("select")).val(conditions[0][1])

		//Setting main conditions 
		cont = $("#mainConditions")
		while(cont.find("section").length > 1)
			$("#matrControl").find(".del-el").click()
		while(cont.find("section div input").length > 1)
			cont.find("section .del-el").click()
		
		for (var i = 1; i < systemSize; ++i){
			$("#matrControl").find(".add-el").click()
			innerCont = $(cont.find("section"))
			innerCont = $(innerCont[innerCont.length - 1])
			while(innerCont.find("div input").length > 1)
				innerCont.find(".del-el").click()
		}
		for (var i = 1; i < systemSize + 1; ++i){
			innerCont = $(cont.find("section")[i - 1])
			for (var j = 1; j < conditions[i][0].length; ++j){
				innerCont.find(".add-el").click()
			}	
			subCont = innerCont.find("div input")
			for (var j = 0; j < conditions[i][0].length; ++j)
				$(subCont[j]).val(conditions[i][0][j])
			$(innerCont.find("select")).val(conditions[i][1])
			$(innerCont.find("li > input")).val(conditions[i][2])
		}

		// Setting last conditions
		cont = $("#lastConditions .triple-cell")
		var lastIndex = conditions.length - 1
		for (var i = 0; i < conditions[lastIndex].length; ++i){
			$(cont[i]).find("select").val(conditions[lastIndex][i][0])
			$(cont[i]).find("input").val(conditions[lastIndex][i][1])
			$(cont[i]).find("input").trigger("input")
		}
	}

	$("#exampleRegular").click(function(){
		setExample("regular")
	})
	$("#exampleBig").click(function(){
		setExample("big")
	})
	$("#exampleError").click(function(){
		setExample("error")
	})
	$("#exampleSmall").click(function(){
		setExample("small")
	})

	$("a").click(function(){
		$("input[name='method_name']").val($(this).attr("name"))
		$("a").removeClass("chosen-type")
		$(this).addClass("chosen-type")
	})

	function transposeArray(array, arrayLength){
	    var newArray = [];
	    for(var i = 0; i < arrayLength; i++){
	        newArray.push([]);
	    };

	    for(var i = 0; i < array.length; i++){
	        for(var j = 0; j < arrayLength; j++){
	            newArray[j].push(array[i][j]);
	        };
	    };

	    return newArray;
	}

	$("#convertToDual").click(function(){
		var objFunction = []
		var taskType = "min"
		var matrix = []
		var constants = []
		var conditions = []
		var inequalities = []

		var cont = $("#lastConditions .triple-cell")
		for (var i = 0; i < cont.length; ++i){
			var sign = ""
			var num = 0
			sign = $($(cont[i]).find("select")).val()
			num = $($(cont[i]).find("input")).val() === "" ? 0 : parseFloat($($(cont[i]).find("input")).val())
			conditions.push([sign, num])
		}
		
		for (var i = 0; i < conditions.length; ++i){
			if (conditions[i][0] === "<=" || conditions[i][1] !== 0){
				alert("Утворення двоїстої задачі вимагає обмеження змінних вигляду \">= 0\" або \"?\"")
				return false
			}
		}

		cont = $("#objectiveFunc").find("div input")
		for (var i = 0; i < cont.length; ++i){
			var num = $(cont[i]).val() === "" ? 0 : parseFloat($(cont[i]).val())
			objFunction.push(num)
		}
		cont = $("#objectiveFunc").find("select")
		if ($(cont).val() === "max")
			taskType = "max"

		cont = $("#mainConditions").find("section")
		for (var i = 0; i < cont.length; ++i){
			var innerCont = $(cont[i]).find("div input")
			var localArr = []
			for (var j = 0; j < innerCont.length; ++j){
				var num = $(innerCont[j]).val() === "" ? 0 : parseFloat($(innerCont[j]).val())
				localArr.push(num)
			}
			matrix.push(localArr)
			var num = $($(cont[i]).find("li > input")).val() === "" ? 0 : parseFloat($($(cont[i]).find("li > input")).val())
			constants.push(num)
			inequalities.push($($(cont[i]).find(".double-cell select")).val())
		}

		console.log(objFunction)
		console.log(taskType)
		console.log(matrix)
		console.log(constants)
		console.log(conditions)
		console.log(inequalities)

		for (var i = 0; i < inequalities.length; ++i){
			if ((inequalities[i] === "<=" && taskType === "min") || (inequalities[i] === ">=" && taskType === "max")) {
				matrix[i] = matrix[i].map(function(el) { return -el; });
				constants[i] *= -1
				inequalities[i] = "<="
			}
		}

		matrix = transposeArray(matrix, matrix[0].length)
		console.log("--------------")

		var newConditions = [], newConstants = [], newObjFunc = [], newInequalities = [], newTaskType = "max"

		for (var i = 0; i < inequalities.length; ++i){
			if (inequalities[i] === "=")
				newConditions.push(["arbitrary", 0])
			else 
				newConditions.push([">=", 0])
		}

		for (var i = 0; i < conditions.length; ++i){
			newConstants.push(objFunction[i])
			if (conditions[i][0] === ">="){
				var sign = taskType === "min" ? "<=" : ">="
				newInequalities.push(sign)
			}
			else 
				newInequalities.push("=")
		}

		for (var i = 0; i < constants.length; ++i){
			newObjFunc.push(constants[i])
		}

		if (taskType === "max")
			newTaskType = "min"

		console.log(newObjFunc)
		console.log(newTaskType)
		console.log(matrix)
		console.log(newConstants)
		console.log(newConditions)
		console.log(newInequalities)

		

		var result = []
		result.push([newObjFunc, newTaskType])
		for (var i = 0; i < matrix.length; ++i){
			result.push([matrix[i], newInequalities[i], newConstants[i]])
		}
		result.push(newConditions)

		console.log(result)

		examples["dual"] = result
		setExample("dual")
	})
});