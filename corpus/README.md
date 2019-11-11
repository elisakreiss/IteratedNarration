# The Annotated Iterated Narration Corpus (AINC) -- Versions

For information about the corpus collection, you can find an overview [here](https://github.com/elisakreiss/iteratednarration).

We provide three versions of the corpus, differing in the complexity of and information provided in the corpus.

## Version 1: Reproductions

This corpus only consists of the seed stories and reproductions, without any ratings. There are 260 rows, where each row is either a seed story (where *generation* and *chain* are set to 0) or a reproduction. The columns in the corpus contain the following information:

- **story_topic**: The original topic of the story. (levels: *arson, bees, professor, scam, smuggler*)
- **story_reproduction**: The exact story the reader saw.
- **condition**: The evidence manipulation that the story resulted from. (levels: *weak evidence, strong evidence*)
- **chain**: The unique chain ID that this particular story belongs to.
- **generation**: The generation the story belongs to. (levels: *0,1,2,3,4,5*, where 0 is the seed story and 1-5 are the reproductions)

## Version 2: Ratings (simple)

This corpus version includes the mean ratings provided by participants for each of the eight guilt related questions for each seed story and reproduction. This results in 2080 rows, where each row represents the mean of all human judgments for a question given a story. This corpus extends version 1 by the following columns:

- **question_topic**: The kind of question, the reader responded to. (levels: *suspect_committedCrime, suspect_conviction, suspect_convictionJustified, evidence, author_trust, reader_emotion, story_subjectivity, author_belief*)
- **mean_rating**: The rating that indicates the reader's response on a slider scale, underlyingly coded as ranging from 0 to 100.
- **question**: The exact wording of the question.

Plus the columns from the original corpus version 1.

- **story_topic**: The original topic of the story. (levels: *arson, bees, professor, scam, smuggler*)
- **story_reproduction**: The exact story the reader saw.
- **condition**: The evidence manipulation that the story resulted from. (levels: *weak evidence, strong evidence*)
- **chain**: The unique chain ID that this particular story belongs to.
- **generation**: The generation the story belongs to. (levels: *0,1,2,3,4,5*, where 0 is the seed story and 1-5 are the reproductions)

## Version 3: Ratings (full)

While version 2 provides the mean ratings, this full version has the raw ratings each participant gave. In addition to the raw ratings, it also shows which person was marked as the suspect, and provides additional information about the participants.
Each row represents a human judgment for a question given a story. This corpus has some columns additional to the ones in version 1.

- **story_reproduction_suspectmarking**: Like *story_reproduction*, but with the original html marking for underlining.
- **sliderlabel_left**/**sliderlabel_right**: The labels that participants saw on the left and right of the slider.
- **suspect_underlined**: A boolean indicating whether a suspect was underlined in the story.
- **trial_number**: When a participant saw a particular question in the experiment.
- **anon_worker_id**: Worker id that was randomly assigned to the participants.

And some information we collected about the participants:

- **age**
- **gender**
- **education**
- **enjoyment**
- **languages**
- **comments**

Plus the columns from the original corpus version 2.

- **question_topic**: The kind of question, the reader responded to. (levels: *suspect_committedCrime, suspect_conviction, suspect_convictionJustified, evidence, author_trust, reader_emotion, story_subjectivity, author_belief*)
- **question**: The exact wording of the question.
- **story_topic**: The original topic of the story. (levels: *arson, bees, professor, scam, smuggler*)
- **story_reproduction**: The exact story the reader saw.
- **condition**: The evidence manipulation that the story resulted from. (levels: *weak evidence, strong evidence*)
- **chain**: The unique chain ID that this particular story belongs to.
- **generation**: The generation the story belongs to. (levels: *0,1,2,3,4,5*, where 0 is the seed story and 1-5 are the reproductions)