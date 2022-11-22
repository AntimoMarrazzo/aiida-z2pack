# README #

# aiida-z2pack

Official Z2pack plugin for AiiDA. 
 The repository as of aiida-z2pack>=3.0 is compatible with aiida-core>=2.0.0 (tested up to 2.1.2). 
The repository as of aiida-z2pack>=2.0 is compatible with aiida-core>=1.0.0 (tested up to 1.6.5). 
For compatibility with older versions use aiida-z2pack==1.0.  
The plugin supports Quantum ESPRESSO only.

[![Build Status](https://travis-ci.com/AntimoMarrazzo/aiida-z2pack.svg?branch=master)](https://travis-ci.com/AntimoMarrazzo/aiida-z2pack)

### How do I get set up? ###

The Z2pack plugin has the following dependencies:
* numpy
* z2pack==2.1.1
* aiida-core~=2.1.2
* aiida_quantumespresso~=3.0
* aiida_wannier90~=3.0.0 

Installing:
* `pip install .`
* or `pip install .[pre-commit,tests,docs]` to install the dependencies for developers (pre-commit, ...)

### Contribution guidelines ###

* Never commit to the master branch!
* Fork the upstream repository and create a branch from develop for the feature you want to add.
* Make a pull-request to have your changes reviewed and merged into the upstream

### Who do I talk to? ###
Contact of repository owner :
* Antimo Marrazzo (THEOS & NCCR MARVEL,EPFL), antimo.marrazzo@epfl.ch
* Davide Grassano (THEOS & NCCR MARVEL,EPFL), davide.grassano@epfl.ch
