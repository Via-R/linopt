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

	$("section .add-el").click(eventAddDoublecell)
	$("section .del-el").click(eventDeleteDoublecell)
	$("#matrControl .add-el").click(eventAddRow)
	$("#matrControl .del-el").click(eventDelRow)

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

	$("#example").click(function(){
		var cont = $("#objectiveFunc")
		while(cont.find("input[type='number']").length > 2)
			cont.parent().find(".del-el").click()
		if(cont.find("input[type='number']").length === 1)
			cont.parent().find(".add-el").click()
		cont.find("input[name='obj_1']").val(-1)
		cont.find("input[name='obj_2']").val(1)
		cont.find("select").val("max")

		cont = $("#mainConditions")
		while(cont.find("section").length > 1)
			$("#matrControl").find(".del-el").click()
		while(cont.find("section div input").length > 1)
			cont.find("section .del-el").click()
		
		cont.find("section .add-el").click()
		$("#matrControl").find(".add-el").click()

		cont.find("input[name='matr1_2']").val(1)
		cont.find("input[name='matr2_1']").val(1)
		cont.find("input[name='matr2_2']").val(1)
		cont.find("input[name='matr2_2']").parent().parent().parent().find("input[name='constant']").val(1)
		cont.find("input[name='matr2_2']").parent().parent().parent().find("select").val("<=")
		$("#matrControl .add-el").click()
		cont.find("input[name='matr3_1']").val(1)
		cont.find("input[name='matr3_2']").val(-4)
		cont.find("input[name='matr3_2']").parent().parent().parent().find("input[name='constant']").val(-2)
		
		cont = $("#lastConditions")
		$(cont.find(".triple-cell")[0]).find("input").val(40)
		$(cont.find(".triple-cell")[0]).find("select").val("<=")
		$(cont.find(".triple-cell")[1]).find("select").val(">=")
		$(cont.find(".triple-cell")[1]).find("input").val(-100)
		$(cont.find(".triple-cell")[1]).find("input").trigger("input")
	})
});