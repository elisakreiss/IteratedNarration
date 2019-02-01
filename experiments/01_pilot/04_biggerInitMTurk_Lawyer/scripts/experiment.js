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
    this.trial_info.main_trials = _.shuffle(main_trials)
	
};
