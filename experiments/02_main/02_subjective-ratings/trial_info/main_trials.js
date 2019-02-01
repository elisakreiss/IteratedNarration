var main_trials = [
	{
		question_id: "evidence",
		// ADD INSERT
		question: "How strong is the evidence for the suspect's / suspects' guilt?",
		slider_left: "not at all",
		slider_right: "very much"
	},
	{
		question_id: "suspect_committedCrime",
		// question: "How likely is it that the suspect is / the suspects are guilty? (i.e., the person/people underlined in the story)",
		question: "How likely is it that the suspect is / the suspects in the crime are guilty? --insert--",
		slider_left: "very unlikely",
		slider_right: "very likely"
	},
	{
		question_id:"suspect_conviction",
		// question: "How likely is a conviction of the suspect(s), i.e., the person/people underlined in the story?",
		question: "How likely is a conviction of the suspect(s) in the crime --insert--?",
		slider_left: "very unlikely",
		slider_right: "very likely"
	},
	{
		question_id: "suspect_convictionJustified",
		// question: "How justified would a conviction of the suspect(s), i.e., the person/people underlined in the story, be?",
		question: "How justified would a conviction of the suspect(s) in the crime --insert-- be?",
		slider_left: "not justified",
		// at all
		slider_right: "very justified"
	},
	// {
	// 	question_id: "suspect_likability",
	// 	question: "How likable do you find the suspect(s), i.e., the person/people underlined in the story?",
	// 	slider_left: "not likable",
	// 	// at all
	// 	slider_right: "very likable"
	// },
	// 
	// Not that relevant
	// {
	// 	question_id: "author_likability",
	// 	question: "How likable do you find the author of this story?",
	// 	slider_left: "not likable",
	// 	slider_right: "very likable"
	// },
	// 
	// 
	{
		question_id: "author_belief",
		question: "How much does the author believe that the suspect --insert-- is guilty?",
		slider_left: "not at all",
		slider_right: "very much"
	},
	// Strong correlation with info_reliability
	{
		question_id: "author_trust",
		question: "How much do you trust the author?",
		slider_left: "not at all",
		slider_right: "very much"
	},
	// correlates with story_subjectivity and story_emotion
	// {
	// 	question_id: "author_judgmental",
	// 	question: "How judgmental is the author?",
	// 	slider_left: "not judgmental",
	// 	slider_right: "very judgmental"
	// },
	{
		question_id: "story_subjectivity",
		question: "How objectively / subjectively written is the story?",
		slider_left: "very objective",
		slider_right: "very subjective"
	},
	// correlates with reader_emotion and author_judgmental
	// {
	// 	question_id: "story_emotion",
	// 	question: "How emotionally is the story told?",
	// 	slider_left: "not emotional",
	// 	slider_right: "very emotional"
	// },
	{
		question_id: "reader_emotion",
		question: "How affected do you feel by the story?",
		slider_left: "not affected",
		slider_right: "very affected"
	},
	// {
	// 	question_id: "info_reliability",
	// 	question: "How reliable do you consider the presented information to be?",
	// 	slider_left: "very unreliable",
	// 	slider_right: "very reliable"
	// },
	{
		question_id: "control1_false",
		question: "How likely is it that this story is a Greek fairy tale?",
		slider_left: "very unlikely",
		slider_right: "very likely"
	},
	{
		question_id: "control2_false",
		question: "How likely is it that this passage is a Bible quote?",
		slider_left: "very unlikely",
		slider_right: "very likely"
	},
	{
		question_id: "control3_true",
		question: "How likely is it that this story contains more than five words?",
		slider_left: "very unlikely",
		slider_right: "very likely"
	},
	{
		question_id: "control4_storydependent",
		question: "How reasonable is it to say that this is a story involving xxx?",
		slider_left: "not reasonable",
		slider_right: "very reasonable"
	}
];

