#!/bin/bash

if [ $# -eq 0 ]
  then
cat <<EOF
    Usage: $0 [figure]

    Reproduces the interpretation process for a given figure in the paper,
    automatically choosing the proper parameters for fragment_processing.py.

    All record fragments are under the examples/ directory, with a name
    corresponding to the appropriate figure in the paper, from fig1 to fig10.

    Example: ~$ $0 fig3
EOF
else
  echo "Running: python -i fragment_processing.py -r examples/$1"
  WFDB=`pwd` python -i fragment_processing.py -r examples/$1
fi
