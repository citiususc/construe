# Construe

*Construe* is a knowledge-based abductive framework for time series interpretation. It provides a knowledge representation model and a set of algorithms for the interpretation of temporal information, implementing a hypothesize-and-test cycle guided by an attentional mechanism. The framework is fully described in the following paper:

T. Teijeiro, P. Félix and J.Presedo: *On the adoption of abductive reasoning for time series interpretation*.

In this repository you will find the complete implementation of the data model and the algorithms, as well as a knowledge base for the interpretation of electrocardiogram (ECG) signals, from the basic waveforms (P, QRS, T) to complex rhythm patterns (Atrial fibrillation, Bigeminy, Trigeminy, Ventricular flutter/fibrillation, etc.). In addition, we provide some utility scripts to reproduce the interpretation of all the ECG strips shown in the paper, and to allow the interpretation of any ECG record in the MIT-BIH format.

## Installation

This project is implemented in pure python, so no installation is required. However, there are strong dependencies with the following python packages:

1. [blist](https://pypi.python.org/pypi/blist)
2. [scipy](https://pypi.python.org/pypi/scipy)
3. [numpy](https://pypi.python.org/pypi/numpy)
4. [scikit-learn v0.16](https://pypi.python.org/pypi/scikit-learn/0.16.1)
5. [PyWavelets](https://pypi.python.org/pypi/PyWavelets/)

In addition, to support visualization of the interpretation results and the interpretation tree, the following packages are needed:

6. [matplotlib](https://pypi.python.org/pypi/matplotlib)
7. [networkx](https://pypi.python.org/pypi/networkx/)
8. [pygraphviz](https://pypi.python.org/pypi/pygraphviz)

Finally, to read ECG signal records it is necessary to have access to a proper installation of the [WFDB software package](http://www.physionet.org/physiotools/wfdb.shtml).

Once all these dependencies are satisfied, is enough to download the project sources and execute the proper python or bash scripts, as explained below.

## Reproducing the examples in the paper

Use the `run_example.sh` script, selecting the figure for which you want to reproduce the interpretation process:

```
./run_example.sh fig1
```

## Interpreting external ECG records

Any ECG record in [MIT-BIH format](https://www.physionet.org/physiotools/wag/header-5.htm) can be interpreted with the *Construe* algorithm. For this, we provide two convenient python modules that may be used as command-line tools. The first one (`fragment_processing.py`) is intended to visually show the result of the interpretation of a (small) ECG fragment, allowing to inspect and reproduce the interpretation process. The second one (`record_processing.py`) is intended to perform background interpretations of full records, resulting in a set of [annotations in the MIT format](https://www.physionet.org/physiotools/wag/annot-5.htm).

## Known issues

- The framework is prepared to interpret records with any sampling frequency, but if frequency differs from 250 Hz the value has to be manually adjusted in `construe/utils/units_helper.py`, line 13.

## License

This project is licensed under the terms of the [GPL v3 license](LICENSE).