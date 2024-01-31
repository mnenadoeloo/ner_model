### Dataset summary
The shared task of CoNLL-2003 concerns language-independent named entity recognition. We will concentrate on four types of named entities: persons, locations, organizations and names of miscellaneous entities that do not belong to the previous three groups.

The CoNLL-2003 shared task data files contain four columns separated by a single space. Each word has been put on a separate line and there is an empty line after each sentence. The first item on each line is a word, the second a part-of-speech (POS) tag, the third a syntactic chunk tag and the fourth the named entity tag. The chunk tags and the named entity tags have the format I-TYPE which means that the word is inside a phrase of type TYPE. Only if two phrases of the same type immediately follow each other, the first word of the second phrase will have tag B-TYPE to show that it starts a new phrase. A word with tag O is not part of a phrase. Note the dataset uses IOB2 tagging scheme, whereas the original dataset uses IOB1.

### Progress of the decision
First, to solve the problem, we chose to additionally train on our bert-base-uncased dataset. I monitored the model training results in wandb.
Results on the validation set:
Validation Loss: 0.06703685227233634
Validation Metric: 0.9365910413030832

After the first method, I used distillation of knowledge. Knowledge distillation is a learning paradigm in which the knowledge of the teacher model is distilled into the student model. The student can be an arbitrary smaller model that solves the same problem.
As a teacher I used the retrained BERT from the previous method
I monitored the model training results in wandb.
Results on the validation set:
Validation Loss: 1.4200858639450658
Validation Metric: 0.8213484080139661

The results are worse than in the first approach. Most likely, if you choose a much larger model, the metrics will be at the same level.
