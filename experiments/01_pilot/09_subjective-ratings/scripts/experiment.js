// customize the experiment by specifying a view order and a trial structure
exp.customize = function() {
    // record current date and time in global_data
    this.global_data.startDate = Date();
    this.global_data.startTime = Date.now();
    // specify view order
    this.views_seq = [
        recaptcha,
        intro,
        beginMainExp,
        main,
        postTest,
        thanks
    ];

    // prepare information about trials (procedure)
    // randomize main trial order, but keep practice trial order fixed
    this.trial_info.main_trials = _.shuffle(main_trials);
    this.trial_info.stories = _.shuffle(story)[0];

    this.global_data.story_reproduction = this.trial_info.stories.reproduction;
    this.global_data.story_title = this.trial_info.stories.story_title;
    this.global_data.generation = this.trial_info.stories.generation;
    this.global_data.chain = this.trial_info.stories.chain;

    // this.global_data.story_comments = "";

    // adds progress bars to the views listed
    // view's name is the same as object's name
    this.progress_bar_in = ["main"];
    // this.progress_bar_in = ['practice', 'main'];
    // styles: chunks, separate or default
    this.progress_bar_style = "default";
    // the width of the progress bar or a single chunk
    this.progress_bar_width = 100;
};
