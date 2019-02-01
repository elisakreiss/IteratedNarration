// insert any functions that are useful throughout the experiment here
var shuffleComb = function(comb) {
    // while this one is trivial, this just to show that we CAN define a function here
    return _.shuffle(comb);
};

function show(obj,display="inline") {
	$("#"+obj).css({"visibility": "visible"});
    $("#"+obj).css({"display": display});

	// if (obj=="next"){
	// 	$("#"+obj).css({"display": "block"});
	// } else {
	// 	$("#"+obj).css({"display": "inline"});
	// }
};

function hide(obj) {
	$("#"+obj).css({"visibility": "hidden"});
	$("#"+obj).css({"display": "none"});
};