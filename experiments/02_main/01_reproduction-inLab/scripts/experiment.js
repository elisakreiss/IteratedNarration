// customize the experiment by specifying a view order and a trial structure
exp.customize = function() {

	// record current date and time in global_data
    this.global_data.startDate = Date();
    this.global_data.startTime = Date.now();
    this.global_data.browser = BrowserDetect.browser;
    this.global_data.os = BrowserDetect.OS;

    // specify view order
    this.views_seq = [intro, 
					 instructions,
					 main,
                     postTest,
                     thanks];
	
    // prepare information about trials (procedure)
	// randomize main trial order, but keep practice trial order fixed
    // chosen_stories = _.shuffle(main_trials).slice(0,5);
    // first_seed = _.shuffle(chosen_stories[0])[0];
    // second_seed = _.shuffle(chosen_stories[1])[0];
    // third_seed = _.shuffle(chosen_stories[2])[0];
    // fourth_seed = _.shuffle(chosen_stories[3])[0];
    // fifth_seed = _.shuffle(chosen_stories[4])[0];
    // this.trial_info.main_trials = [first_seed,second_seed,third_seed,fourth_seed,fifth_seed];

    this.trial_info.main_trials = _.shuffle(main_trials);
	
};
