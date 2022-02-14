# Nodule Detection Algorithm

This codebase implements our final solution for the nodule detection track in [NODE21](https://node21.grand-challenge.org/). 
It contains files to build a docker image which can be submitted as an algorithm on the [grand-challenge](https://www.grand-challenge.org) platform. The final algorithm can be found [here](https://grand-challenge.org/algorithms/final-submission-mtec/).

The code in this repository is based on [this](https://grand-challenge.org/algorithms/final-submission-mtec/) template repository with docker and evalutils.  

For the generation of additional Nodules, only the process.py script from the [generation track](https://github.com/node21challenge/node21_generation_baseline) template was modified (see /src/nodule_generation/) to generate the nodules given in nodules_to_generate.csv. In addition to the csv file, the algorithm requires the images to work with as input.
