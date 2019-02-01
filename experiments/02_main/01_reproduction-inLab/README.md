# Iterated narration experiment inspired by Bartlett (1932)

## Cloning and running the experiment

```
# clone the repo, e.g.:
git clone https://github.com/babe-project/IteratedNarration

# go to 'IteratedNarration'

# open 'index.html' in the browser to see the experiment
```

## Description

Participants read a story and are then asked to reproduce it. The stories that are presented are retrieved from a data base and can either be the original story or a reproduction from another participant. The idea is to be able to run experiments involving iterations (where one participant responds to submissions from another participant) on MTurk without getting too much redundant data.<br><br>
The experimenter can define the maximum length of these chains of reproductions (`generation_max` in views.js).<br>
The number of chains is not explicitely specified beforehand. On each trial, the participant advances one open chain for one generation. An open chain is defined as a chain, which hasn't reached its `generation_max` yet and where the stories still have a predefined "reasonable" length. The chain for the current participant is selected randomly from all chains that are still open plus one new chain, which starts with the original story.