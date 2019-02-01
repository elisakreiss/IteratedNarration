var recaptcha = {
    name: "recaptcha",
    render: function(){
        var viewTemplate = $("#recaptcha-view").html();

        $("#main").html(
            Mustache.render(viewTemplate, {
                name: this.name
            })
        );
    },
    trials: 1
};

var intro = {
    name: "intro",
    // introduction title
    title: "ALPS lab Stanford",
    // introduction text
    text:
        "Thank you for participating in our study. In this study you’ll answer questions about a story that you’ll read. It will take approximately <strong>3</strong> minutes.<br>Note: The story might include sensitive content.",
    legal_info:
        "<strong>LEGAL INFORMATION</strong>:<br><br>We invite you to participate in a research study on language production and comprehension.<br>Your experimenter will ask you to do a linguistic task such as reading sentences or words, naming pictures or describing scenes, making up sentences of your own, or participating in a simple language game.<br><br>You will be paid for your participation at the posted rate.<br><br>There are no risks or benefits of any kind involved in this study.<br><br>If you have read this form and have decided to participate in this experiment, please understand your participation is voluntary and you have the right to withdraw your consent or discontinue participation at any time without penalty or loss of benefits to which you are otherwise entitled. You have the right to refuse to do particular tasks. Your individual privacy will be maintained in all published and written data resulting from the study.<br>You may print this form for your records.<br><br>CONTACT INFORMATION:<br>If you have any questions, concerns or complaints about this research study, its procedures, risks and benefits, you should contact the Protocol Director Meghan Sumner at <br>(650)-725-9336<br><br>If you are not satisfied with how this study is being conducted, or if you have any concerns, complaints, or general questions about the research or your rights as a participant, please contact the Stanford Institutional Review Board (IRB) to speak to someone independent of the research team at (650)-723-2480 or toll free at 1-866-680-2906. You can also write to the Stanford IRB, Stanford University, 3000 El Camino Real, Five Palo Alto Square, 4th Floor, Palo Alto, CA 94306 USA.<br><br>If you agree to participate, please proceed to the study tasks.",
    // introduction's slide proceeding button text
    buttonText: "Begin experiment",
    // render function renders the view
    render: function() {
        var viewTemplate = $("#intro-view").html();

        $("#main").html(
            Mustache.render(viewTemplate, {
                picture: "images/alpslogo.png",
                title: this.title,
                text: this.text,
                legal_info: this.legal_info,
                button: this.buttonText
            })
        );

        var prolificId = $("#prolific-id");
        var IDform = $("#prolific-id-form");
        var next = $("#next");

        var showNextBtn = function() {
            if (prolificId.val().trim() !== "") {
                next.removeClass("nodisplay");
            } else {
                next.addClass("nodisplay");
            }
        };

        if (config_deploy.deployMethod !== "Prolific") {
            IDform.addClass("nodisplay");
            next.removeClass("nodisplay");
        }

        prolificId.on("keyup", function() {
            showNextBtn();
        });

        prolificId.on("focus", function() {
            showNextBtn();
        });

        // moves to the next view
        next.on("click", function() {
            if (config_deploy.deployMethod === "Prolific") {
                exp.global_data.prolific_id = prolificId.val().trim();
            }

            exp.findNextView();
        });
    },
    // for how many trials should this view be repeated?
    trials: 1
};

var beginMainExp = {
    name: "beginMainExp",
    // introduction title
    title: "Read the Story",
    // introduction text
    text:
        "Please, read the following text. When you're done, press <strong>Next</strong> to show the first question.",
    // introduction's slide proceeding button text
    // buttonText: "Next",
    // render function renders the view
    render: function() {
        var viewTemplate = $("#begin-exp-view").html();

        $("#main").html(
            Mustache.render(viewTemplate, {
                title: this.title,
                text: this.text,
                story_text: exp.trial_info.stories.reproduction
            })
        );

        // moves to the next view
        $("#next").on("click", function() {
            exp.findNextView();
        });
    },
    // for how many trials should this view be repeated?
    trials: 1
};

var main = {
    name: "main",
    title: "Questions",
    text: "<br><strong>Note</strong>: The underlined in the story may be referred to as suspects in the question.",
    render: function(CT) {
        // fill variables in view-template
        var viewTemplate = $("#main-view").html();

        $("#main").html(
            Mustache.render(viewTemplate, {
                title: this.title,
                text: this.text,
                story_text: exp.trial_info.stories.reproduction,
                question: exp.trial_info.main_trials[CT].question,
                slider_left: exp.trial_info.main_trials[CT].slider_left,
                slider_right: exp.trial_info.main_trials[CT].slider_right,
            })
        );

        // $("#story_comments").val(exp.global_data.story_comments);

        var slider = $('#slider');
        var slider_changed = false;
        slider.on('click', function() {
            slider_changed = true;
            $("#error").css({"visibility": "hidden"});
            console.log("Yey, you clicked the slider and possibly changed the value");
        });

        var box_checked = false;
        $('input[id=checkbox]').change(function(){
            if($(this).is(':checked')) {
                box_checked = true;
                $('#slider_box').css("opacity", "0.2");
                $("#error").css({"visibility": "hidden"});
                console.log("Yey, you checked the box!");
                console.log("$('#checkox')");
                console.log($('#checkbox').prop('checked'));
            } else {
                box_checked = false;
                $('#slider_box').css("opacity", "1");
                console.log("Yey, you unchecked the box!");
                console.log("$('#checkox')");
                console.log($('#checkbox').prop('checked'));
            }
        });


        // event listener for buttons; when an input is selected, the response
        // and additional information are stored in exp.trial_info
        $("#next").on("click", function() {
            if (slider_changed == true || box_checked == true){
                var RT = Date.now() - startingTime; // measure RT before anything else
                var trial_data = {
                    trial_type: exp.trial_info.main_trials[CT].question_id,
                    trial_number: CT + 1,
                    question: exp.trial_info.main_trials[CT].question,
                    slider_left: exp.trial_info.main_trials[CT].slider_left,
                    slider_right: exp.trial_info.main_trials[CT].slider_right,
                    slider_val: $('#slider').val(),
                    box_checked: $('#checkbox').prop('checked')
                };
                // exp.global_data.story_comments = $("#story_comments").val();
                exp.trial_data.push(trial_data);
                exp.findNextView();
            } else {
                console.log($("#error"));    
                $("#error").css({"visibility": "visible"});
            };
        });

        // record trial starting time
        var startingTime = Date.now();
    },
    trials: 15
};


var postTest = {
    name: "postTest",
    title: "Additional Info",
    text:
        "Answering the following questions is optional, but will help us understand your answers.",
    buttonText: "Continue",
    render: function() {
        var viewTemplate = $("#post-test-view").html();
        $("#main").html(
            Mustache.render(viewTemplate, {
                title: this.title,
                text: this.text,
                buttonText: this.buttonText
            })
        );

        $("#next").on("click", function(e) {
            // prevents the form from submitting
            e.preventDefault();

            // records the post test info
            exp.global_data.HitCorrect = $("#HitCorrect").val();
            exp.global_data.age = $("#age").val();
            exp.global_data.gender = $("#gender").val();
            exp.global_data.education = $("#education").val();
            exp.global_data.languages = $("#languages").val();
            exp.global_data.enjoyment = $("#enjoyment").val();
            exp.global_data.comments = $("#comments")
                .val()
                .trim();
            exp.global_data.endTime = Date.now();
            exp.global_data.timeSpent =
                (exp.global_data.endTime - exp.global_data.startTime) / 60000;

            // moves to the next view
            exp.findNextView();
        });
    },
    trials: 1
};

var thanks = {
    name: "thanks",
    message: "Thank you for taking part in this experiment!",
    render: function() {
        var viewTemplate = $("#thanks-view").html();

        // what is seen on the screen depends on the used deploy method
        //    normally, you do not need to modify this
        if (
            config_deploy.is_MTurk ||
            config_deploy.deployMethod === "directLink"
        ) {
            // updates the fields in the hidden form with info for the MTurk's server
            $("#main").html(
                Mustache.render(viewTemplate, {
                    thanksMessage: this.message
                })
            );
        } else if (config_deploy.deployMethod === "Prolific") {
            $("main").html(
                Mustache.render(viewTemplate, {
                    thanksMessage: this.message,
                    extraMessage:
                        "Please press the button below to confirm that you completed the experiment with Prolific<br />" +
                        "<a href=" +
                        config_deploy.prolificURL +
                        ' class="prolific-url">Confirm</a>'
                })
            );
        } else if (config_deploy.deployMethod === "debug") {
            $("main").html(Mustache.render(viewTemplate, {}));
        } else {
            console.log("no such config_deploy.deployMethod");
        }

        exp.submit();
    },
    trials: 1
};
