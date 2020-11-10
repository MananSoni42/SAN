# SAN
Stochastic answering network for Squad 2.0

## Objectives
* [ ] Reproduce results of the joint SAN and joint SAN + classifier. (Note the training of these models is the same so computationally, it shouldn’t be very expensive to reproduce both results as compared to just one.)
* [ ] Plot Lspan, Lclassifier, and overall loss and also training accuracy and f1 with epochs.
* [ ] Reproduce the classifier accuracy.
* [ ] Report other metrics - precision, recall, accuracy f1 for joint, joint+classifier, and on classifier alone as well.
* [ ] Experiment with different lambdas in the loss function and see how accuracy of classifier changes. (In case this gets computationally very expensive, we can discuss what to do about it later)

## Modules

* [x] ```preprocess.py```: Pre-processes the entire dataset
* [ ] ```lexicon_embed.py```
* [ ] ```contextual_embed.py```
* [ ] ```mem.py```
* [ ] ```san.py```
* [ ] ```classifier.py```
* [ ] ```train.py```
* [ ] ```plot_graphs.py```

## Usage

* Requires Python3+ and pip3 to be installed
* Run ```pip3 install -r requirements.txt```
* Run ```python3 -m spacy download "en"```
* Run ```get_data.sh```

Now you are ready to go!

* Run ```preprocess.py``` with relevant arguements to Pre-process and generate train and dev sets
* Run ```train.py``` to train the model and store the results
* Run ```plot_results.py``` with relevant arguements to get trainig graphs for various metrics and the loss function
