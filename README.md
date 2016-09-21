# Construe

*Construe* is a knowledge-based abductive framework for time series interpretation. It provides a knowledge representation model and a set of algorithms for the interpretation of temporal information, implementing a hypothesize-and-test cycle guided by an attentional mechanism. The framework is fully described in the following paper:

 [1]: T. Teijeiro and P. Félix: *On the adoption of abductive reasoning for time series interpretation*, 2016,  [arXiv:1609.05632](http://arxiv.org/abs/1609.05632).

In this repository you will find the complete implementation of the data model and the algorithms, as well as a knowledge base for the interpretation of electrocardiogram (ECG) signals, from the basic waveforms (P, QRS, T) to complex rhythm patterns (Atrial fibrillation, Bigeminy, Trigeminy, Ventricular flutter/fibrillation, etc.). In addition, we provide some utility scripts to reproduce the interpretation of all the ECG strips shown in paper [1], and to allow the interpretation of any ECG record in the [MIT-BIH format](https://www.physionet.org/faq.shtml#file_types).

Recently, we have also included an algorithm for [automatic heartbeat classification on ECG signals](Beat_Classification.md), described in the paper:

 [2]: T. Teijeiro, P. Félix, J.Presedo and D. Castro: *Heartbeat classification using abstract features from the abductive interpretation of the ECG*

## Installation

This project is implemented in pure python, so no installation is required. However, the core algorithms have strong dependencies with the following python packages:

1. [blist](https://pypi.python.org/pypi/blist)
2. [numpy](https://pypi.python.org/pypi/numpy)

In addition, the knowledge base for ECG interpretation depends on the following packages:

3. [scipy](https://pypi.python.org/pypi/scipy)
4. [scikit-learn v0.16](https://pypi.python.org/pypi/scikit-learn/0.16.1)
5. [PyWavelets](https://pypi.python.org/pypi/PyWavelets/)

To support visualization of the interpretation results and the interpretations tree and run the usage examples, the following packages are also needed:

6. [matplotlib](https://pypi.python.org/pypi/matplotlib)
7. [networkx](https://pypi.python.org/pypi/networkx/)
8. [pygraphviz](https://pypi.python.org/pypi/pygraphviz)

Finally, to read ECG signal records it is necessary to have access to a proper installation of the [WFDB software package](http://www.physionet.org/physiotools/wfdb.shtml).

To make easier the installation of Python dependencies, we recommend the [Anaconda Python distribution](https://www.continuum.io/anaconda-overview). Alternatively, you can install them using pip with the following command:

```
 ~$ pip install -r requeriments.txt
```

Once all the dependencies are satisfied, it is enough to download the project sources and execute the proper python or bash scripts, as explained below.

## Getting started
### *Construe* as a tool for ECG analysis
Along with the general data model for knowledge description and the interpretation algorithms, a comprehensive knowledge base for ECG signal interpretation is provided with the framework, so the software can be directly used as a tool for ECG analysis.

#### Demo examples
All signal strips in [1] are included as interactive examples to make it easier to understand how the interpretation algorithms work. For this, use the `run_example.sh` script, selecting the figure for which you want to reproduce the interpretation process:

```
./run_example.sh fig3
```

Once the interpretation is finished, the resulting observations are printed to the terminal, and two interactive are shown. One plots the ECG signal with all the observations organized in abstraction levels (deflections, waves, and rhythms), and the other shows the interpretations tree explored to find the result. Each node in the tree can be selected to show the observations at a given time point during the interpretation, allowing to reproduce the *abduce*, *deduce*, *subsume* and *predict* reasoning steps [1].

#### Interpreting external ECG records

Any ECG record in [MIT-BIH format](https://www.physionet.org/physiotools/wag/header-5.htm) can be interpreted with the *Construe* algorithm. For this, we provide two convenient python modules that may be used as command-line tools. The first one (`fragment_processing.py`) is intended to visually show the result of the interpretation of a (small) ECG fragment, allowing to inspect and reproduce the interpretation process by navigating through the interpretations tree. The second one (`record_processing.py`) is intended to perform background interpretations of full records, resulting in a set of [annotations in the MIT format](https://www.physionet.org/physiotools/wag/annot-5.htm). Both tools follow the [WFDB Aplications](https://www.physionet.org/physiotools/wag/wag.htm) command-line interface, and usage details are available with the `-h` option.

### Using *Construe* in another problems and domains

We will be glad if you want to use *Construe* to solve problems different from ECG interpretation, and we will help you to do so. The first step is to understand what is under the hood, and the best reference is [1]. After this, you will have to define the **Abstraction Model** for your problem, based on the **Observable** and **Abstraction Pattern** formalisms. As an example, a high-level description of the ECG abstraction model is available in [2], and its implementation is in the [`knowledge`](construe/knowledge) subdirectory. A tutorial is also available in the project [wiki](https://github.com/citiususc/construe/wiki/How-to-define-abstraction-models).

Once the domain-specific knowledge base has been defined, the `fragment_processing.py` module should serve as a basis for the execution of the full hypothesize-and-test cycle with different time series and the new abstraction model.

## Repository structure

The source code is structured in the following main modules:

 - [`acquisition`](construe/acquisition): Modules for the acquisition of the raw time series data. Currently it is highly oriented to ECG data in the [MIT-BIH format](https://www.physionet.org/faq.shtml#file_types).
 - [`inference`](construe/inference): Definition of the interpretation algorithms, including the *construe* algorithm and the reasoning modes (*abduce*, *deduce*, *subsume*, *predict* and *advance*) [1].
 - [`knowledge`](construe/knowledge): Definition of the ECG abstraction model, including *observables* and *abstraction patterns*.
 - [`model`](construe/model): General data model of the framework, including the base class for all *observables* and classes to implement *abstraction grammars* as finite automata.
 - [`util`](construe/util): Miscellaneous utility modules, including signal processing and plotting routines.

## Known issues

- The ECG knowledge base is prepared to interpret records with any sampling frequency, but if frequency differs from 250 Hz the value has to be manually adjusted in `construe/utils/units_helper.py`, line 13.
- Abductive interpretation of time-series is NP-Hard [1]. This implementation includes several optimizations to make computations feasible, but still the running times are probably longer than you expect. Try the `-v` flag to get feedback and make the wait less painful ;-).

## License

This project is licensed under the terms of the [GPL v3 license](LICENSE).
