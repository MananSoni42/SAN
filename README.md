# SAN
Stochastic answering network for Squad 2.0

## Objectives
* [x] Reproduce results of the joint SAN and joint SAN + classifier. (Note the training of these models is the same so computationally, it shouldn’t be very expensive to reproduce both results as compared to just one.)
* [x] Plot Lspan, Lclassifier, and overall loss and also training accuracy and f1 with epochs.
* [ ] Reproduce the classifier accuracy **(differs by ~10%)**
* [x] Report other metrics - precision, recall, accuracy f1 for joint, joint+classifier, and on classifier alone as well.  
**(Have not trained on the classifier alone)**
* [x] Experiment with different lambdas in the loss function and see how accuracy of classifier changes. (In case this gets computationally very expensive, we can discuss what to do about it later)

### Installation and setup

* Requires Python3+ and pip3 to be installed
* To install all dependancies:
    * ```pip3 install -r requirements.txt```
    * ```python3 -m spacy download "en"```
    * ```get_data.sh```

### Scripts for running the different modules
> **NOTE**: You must run the get_data.sh script completely to execute any of the other Scripts

* Download the datasets and our model weights: ```./get_data.sh``` (~3.5 GB)
* Pre-process the data ```./preprocess.sh```
* Train the model ```./train.sh``` (Recommended to use ```train.py``` with correct arguements instead)
* Evaluate the model ```./evaluate.sh path_to_checkpoint```
* Plot graphs ```./plot_results.py``` (graphs are stored in plot/)
* Run the model on a custom sample: ```./predict_sample.py```
> **NOTE**: To run the last script, put your custom paragraph in paragraph.txt and your questions (1 per line) in questions.txt (samples have been provided in these files)
