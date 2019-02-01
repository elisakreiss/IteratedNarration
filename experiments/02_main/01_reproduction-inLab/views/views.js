var intro = {
    // introduction title
    "title": "Welcome!",
    // introduction text
    "text": "Thank you for participating in our study. In this study, you will read five stories and are asked to reproduce them as accurately as you can.",
    "legal_info": "<strong>LEGAL INFORMATION</strong>:<br><br>We invite you to participate in a research study on language production and comprehension.<br>Your experimenter will ask you to do a linguistic task such as reading sentences or words, naming pictures or describing scenes, making up sentences of your own, or participating in a simple language game.<br><br>There are no risks or benefits of any kind involved in this study.<br><br>If you have read this form and have decided to participate in this experiment, please understand your participation is voluntary and you have the right to withdraw your consent or discontinue participation at any time without penalty or loss of benefits to which you are otherwise entitled. You have the right to refuse to do particular tasks. Your individual privacy will be maintained in all published and written data resulting from the study.<br>You may print this form for your records.<br><br>CONTACT INFORMATION:<br>If you have any questions, concerns or complaints about this research study, its procedures, risks and benefits, you should contact the Protocol Director Meghan Sumner at <br>(650)-725-9336<br><br>If you are not satisfied with how this study is being conducted, or if you have any concerns, complaints, or general questions about the research or your rights as a participant, please contact the Stanford Institutional Review Board (IRB) to speak to someone independent of the research team at (650)-723-2480 or toll free at 1-866-680-2906. You can also write to the Stanford IRB, Stanford University, 3000 El Camino Real, Five Palo Alto Square, 4th Floor, Palo Alto, CA 94306 USA.<br><br>If you agree to participate, please proceed to the study tasks.",
    // introduction's slide proceeding button text
    "buttonText": "Begin experiment",
    // render function renders the view
    render: function() {
        
        viewTemplate = $('#intro-view').html();
        $('#main').html(Mustache.render(viewTemplate, {
            title: this.title,
            text: this.text,
            legal_info: this.legal_info,
            button: this.buttonText
        }));

        // moves to the next view
        $('#next').on('click', function(e) {
            exp.findNextView();
        });

    },
    // for how many trials should this view be repeated?
    trials: 1
}

var instructions = {
     // instruction's title
    "title": "Instructions",
    // instruction's text
    "text": "In each of the five trials of this experiment, you will read a short story. You are then asked to retell this story. <strong>Please do not take notes.</strong> Make sure to retell the story immediately after reading the original. <br><br><p><strong>Notice:</strong> You may experience lags during the experiment, because the experiment communicates with a data base. Please just be patient, as the experiment should resume soon enough.",
    // instuction's slide proceeding button text
    "buttonText": "Let's go!",
    render: function() {

        viewTemplate = $("#instructions-view").html();
        $('#main').html(Mustache.render(viewTemplate, {
            title: this.title,
            text: this.text,
            button: this.buttonText
        }));

        // moves to the next view
        $('#next').on('click', function(e) {
            // due to loading times, disable button so experiment can't be advanced until last slide just here
            $('#next').html('Loading...');
            $('#next').prop('disabled', true);
            exp.findNextView();
        }); 

    },
    trials: 1
}


var main = {
	
	trials : 5,
	
    render : function(CT) {

        // record first database load starting time
        var startingTimeMain = Date.now();

        var current_seed_shuffled = _.shuffle(exp.trial_info.main_trials[CT]);

        // number of chains per seed
        var chain_max = 5;
        // max generations per chains (generation_max becomes deadend)
        var generation_max = 5;
        // minimum amount of characters that a reproduction should have; 
        // otherwise shows text prompting participant to make a more explicit answer
        var min_repro_length = 3;
        // reproductions with a total number of characters smaller than the final_repro_length
        // will automatically terminate the chain, since they are so short that they can be
        // recalled almost word-by-word
        var final_repro_length = 100;

        // data from database
		var retrieved_data;
		
		$.ajax({
				type: 'GET',
				url: "https://mcmpact.ikw.uni-osnabrueck.de/babe/api/retrieve_experiment/15",
				crossDomain: true,
				success: function (responseData, textStatus, jqXHR) {
                    retrieved_data = responseData;
					callback("success");
				},
                statusCode: {
                    404: function() {
                        console.log("404 error occurred, possibly due to empty database.")
                        callback("error");
                    }
                }
			  });
		
		function callback(state) {

            // record first database load ending time
            var endingTimeMain = Date.now();

            var current_condition = current_seed_shuffled[0];
			
            // 
            // HELPER FUNCTIONS
            //   
            // show() and hide() are in helpers.js

            // returns chain_end if chain number is already in chain_ends
            function checkChainPresence(chain) {
                for (var chain_end=0; chain_end<chain_ends.length; chain_end++) {
                    if (chain == chain_ends[chain_end]["chain"]) {
                        return chain_end;
                    }
                }
                return "not there";
            };

            function make_chainid() {
              var chain_id = "";
              var values = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789";

              for (var i = 0; i < 16; i++)
                chain_id += values.charAt(Math.floor(Math.random() * values.length));

              return chain_id;
            };

            // create and return a new chain with initial seed
            function initiate_chain() {
                var new_chain = {
                    chain: make_chainid(),
                    generation: 0,
                    deadend: false,
                    reproduction: current_condition.text
                };
                return new_chain;
            };


            // 
            // Decide on stimulus to show; define chain and generation
            // 
            // var story_kind = exp.trial_info.main_trials[CT].title;
            var story_kind = current_condition.title;
            // to save available chain_ends from database
            var chain_ends = [];
            var chain;
            var generation;
            var story_text;

            // if database was accessible (i.e., non-empty)
            if (state == 'success') {

                loadT_main = endingTimeMain-startingTimeMain;

                // if number of chains for story_title > chain_max, choose the other condition
                function get_number_of_filled_chains(retrieved_data,story_kind) {
                    console.log("get_number_of_unique_chains");
                    console.log("story_kind");
                    console.log(story_kind);
                    chains = [];
                    for (var participant=0; participant<retrieved_data.length; participant++){
                        for (var trial=0; trial<retrieved_data[participant].length; trial++){
                            var current_trial = retrieved_data[participant][trial];
                            if ((current_trial["story_title"] == story_kind) & (current_trial["deadend"] == true)){
                                chains.push(current_trial["chain"]);
                            }
                        }
                    }
                    // console.log("(_.uniq(chains)).length");
                    // console.log((_.uniq(chains)).length);
                    console.log("chains.length");
                    console.log(chains.length);
                    return chains.length;
                };

                if (get_number_of_filled_chains(retrieved_data,story_kind) >= chain_max){
                    console.log("get_number_of_unique_chains >= chain_max");
                    console.log("so change current condition from");
                    console.log(current_condition);
                    current_condition = current_seed_shuffled[1];
                    console.log("to");
                    console.log(current_condition);
                    story_kind = current_condition.title;
                    console.log("and use now story_kind");
                    console.log(story_kind);
                }

                // database is structured in one array for each participant that yields one object for each trial
                // go through all trials
                for (var participant=0; participant<retrieved_data.length; participant++){
                    for (var trial=0; trial<retrieved_data[participant].length; trial++){

                        var current_trial = retrieved_data[participant][trial];

                        //
                        // set all stories to deadend, that are shorter than final_repro_length
                        if (current_trial["reproduction"].length < final_repro_length) {
                            current_trial["deadend"] = true;
                        }
                        //


                        // if it is a trial with the wanted story
                        if (current_trial["story_title"] == story_kind) {

                            var new_chain_end = {
                                chain: current_trial["chain"],
                                generation: current_trial["generation"],
                                deadend: current_trial["deadend"],
                                reproduction: current_trial["reproduction"]
                            };

                            var pos_in_chain_ends = checkChainPresence(current_trial["chain"]);

                            // if chain not yet in chain_ends, add it
                            if (pos_in_chain_ends == "not there") {
                                chain_ends.push(new_chain_end);
                            } else {
                                // if generation of current trial is bigger than already saved one, replace it with current trial
                                if (current_trial["generation"] > chain_ends[pos_in_chain_ends]["generation"]) {
                                    chain_ends[pos_in_chain_ends] = new_chain_end;
                                // else if they are equal, keep the one that is not a deadend (if both are, keep saved one)
                                } else if ((current_trial["generation"] == chain_ends[pos_in_chain_ends]["generation"]) & (!current_trial["deadend"])) {
                                    chain_ends[pos_in_chain_ends] = new_chain_end;
                                }
                                // else keep saved one 
                            }
                        }
                    }
                };

                // add new seed to chain_ends to enable the creation of new chains even when others are still open
                // always add one initial chain
                // chain_ends.push(initiate_chain());

                // fill up until desired number of chains (chain_max) is reached
                while (chain_ends.length < chain_max){
                    chain_ends.push(initiate_chain());
                }
                console.log("chain_ends");
                console.log(chain_ends);
                
                // choose chain_end that will be continued with this participant
                var shuffled = _.shuffle(chain_ends);
                var found_chainend = false;

                while (!found_chainend){
                    for (var chain_end=0; chain_end<shuffled.length; chain_end++) {

                        var current_chainend = shuffled[chain_end];
                        // each chain_end is acceptable that is not a deadend
                        if (current_chainend["deadend"]==false) {
                            found_chainend = true;

                            story_text = current_chainend["reproduction"];
                            chain = current_chainend["chain"];
                            generation = current_chainend["generation"]+1;
                            break;
                        }
                    }
                    // if there is no story in our data base that fits and is not a deadend, start with new seed
                    if (found_chainend == false) {
                        found_chainend = true;
                        story_text = current_condition.text;
                        // if two people start at the same time, both can still be used, but system would fail if they get the same number
                        chain = make_chainid();
                        generation = 1;
                    }
                }
            // else (if ajax GET call throws 404 error, because, e.g., the database doesn't exist yet, create new chain)
            } else {
                console.log("start initial chain");
                story_text = current_condition.text;
                chain = make_chainid();
                generation = 1;
                loadT_main = "NA";
            }

            console.log("current story condition:");
            console.log(story_kind);

            // fill variables in view-template
            var viewTemplate = $('#main-view').html();
            $('#main').html(Mustache.render(viewTemplate, {
                recap_instruction: recap_instruction,
                story_text:        story_text,
                button1:           "Ready!",
                button2:           "Done!"
            }));

            // update the progress bar based on how many trials there are in this round
            var filled = exp.currentTrialInViewCounter * (180 / exp.views_seq[exp.currentViewCounter].trials);
            $('#filled').css('width', filled);

            var recap_instruction = "After you read the story, press 'Ready!'. Use the upcoming text field to reproduce the story, as best as you can. Then press 'Done!'.";

            // event listener for buttons; when an input is selected, the response
            // and additional information are stored in exp.trial_info
            $('#start_repro').on('click', function(e) {
                hide("story_text");
                hide("start_repro");
                show("reproduction");
                show("next","block");
            }); 

			// event listener for buttons; when an input is selected, the response
            // and additional information are stored in exp.trial_info
            $('#next').on('click', function(e) {
                // ensure that participants can't just click through
                if ($('#reproduction').val().length <= min_repro_length){
                    show('error_emptyrepro',"block");
                } else {
                    // due to loading times, disable button so experiment can't be advanced until last slide just here
                    $('#next').html('Sending...');
                    $('#next').prop('disabled', true);

                    hide('error_emptyrepro');

                    deadend = false;
                    if (generation >= generation_max) {
                        deadend = true;
                    }

                    // 
                    trial_data = {
                        trial_type: "reproduction",
                        trial_number: CT + 1,
                        story_title: story_kind,
                        story_text: story_text,
                        reproduction: $('#reproduction').val(),
                        deadend: deadend,
                        chain: chain,
                        generation: generation,
                        LoadT_main: loadT_main
                    };
                    exp.trial_data.push(trial_data);
                    exp.findNextView();
                }
            }); 
		}
    }
};

var postTest = {
    "title": "Additional Info",
    "text": "Answering the following questions is optional, but will help us understand your answers.",
    "buttonText": "Continue",
    render : function() {

        viewTemplate = $('#post-test-view').html();
        $('#main').html(Mustache.render(viewTemplate, {
            title: this.title,
            text: this.text,
            buttonText: this.buttonText
        }));

        $('#next').on('click', function(e) {
            // due to loading times, disable button so experiment can't be advanced until last slide just here
            $('#next').html('Sending...');
            $('#next').prop('disabled', true);
            // prevents the form from submitting
            e.preventDefault();

            // records the post test info
            exp.global_data.familiar = $('#familiar').val();
            exp.global_data.assess = $('#assess').val();
            exp.global_data.gender = $('#gender').val();
            exp.global_data.age = $('#age').val();
            exp.global_data.education = $('#education').val();
            exp.global_data.languages = $('#languages').val();
            exp.global_data.enjoyment = $('#enjoyment').val();
            exp.global_data.comments = $('#comments').val().trim();
            exp.global_data.endTime = Date.now();
            exp.global_data.timeSpent = (exp.global_data.endTime - exp.global_data.startTime) / 60000;

            // moves to the next view
            exp.findNextView();
        })

    },
    trials: 1
};

var thanks = {
    "message": "Thank you for taking part in this experiment!",
    render: function() {

        // record second database load starting time
        var startingTimeSubmit = Date.now();      

        var retrieved_data2;

        // check, if someone submitted this exact coninuation in the meantime
        $.ajax({
                type: 'GET',
                url: "https://mcmpact.ikw.uni-osnabrueck.de/babe/api/retrieve_experiment/15",
                crossDomain: true,
                success: function (responseData, textStatus, jqXHR) {
                    retrieved_data2 = responseData;
                    callback2("success");
                },
                statusCode: {
                    404: function() {
                        console.log("404 error occurred, possibly due to empty database.")
                        callback2("error");
                    }
                }
              });
        
        function callback2(state) {

            // record second database load starting time
            var endingTimeSubmit = Date.now(); 

            viewTemplate = $('#thanks-view').html();

            // what is seen on the screen depends on the used deploy method
            //    normally, you do not need to modify this
            if ((config_deploy.is_MTurk) || (config_deploy.deployMethod === 'directLink')) {
                // updates the fields in the hidden form with info for the MTurk's server
                $('#main').html(Mustache.render(viewTemplate, {
                    thanksMessage: this.message,
                }));
            } else if (config_deploy.deployMethod === 'Prolific') {
                var prolificURL = 'https://prolific.ac/submissions/complete?cc=' + config_deploy.prolificCode;

                $('main').html(Mustache.render(viewTemplate, {
                    thanksMessage: this.message,
                    extraMessage: "Please press the button below<br />" + '<a href=' + prolificURL +  ' class="prolific-url">Finished!</a>'
                }));
            } else if (config_deploy.deployMethod === 'debug') {
                $('main').html(Mustache.render(viewTemplate, {}));
            } else {
                console.log('no such config_deploy.deployMethod');
            };

            // if the database is still empty (or can't be called for other reasons), just set loading time to NA and submit
            if (state == "error") {
                exp.global_data.LoadT_submit = "NA";

            } else {
                exp.global_data.LoadT_submit = endingTimeSubmit-startingTimeSubmit;

                // for every trial in the database, check whether this is the same chain continuation that you want to submit
                // if so, declare this one as a deadend
                for (var participant=0; participant<retrieved_data2.length; participant++){
                    for (var trial=0; trial<retrieved_data2[participant].length; trial++){

                        var current_trial = retrieved_data2[participant][trial];
                        var number_of_trials = 1;

                        for (var trial=0; trial<main.trials; trial++){
                            if ((exp.trial_data[trial]["chain"] == current_trial["chain"]) & (exp.trial_data[trial]["generation"] == current_trial["generation"])) {
                                exp.trial_data[trial]["deadend"] = true;
                            };
                        };
                    };
                };
            };

            console.log("exp.trial_data");
            console.log(exp.trial_data);

            exp.submit();

        };


    },
    trials: 1
}