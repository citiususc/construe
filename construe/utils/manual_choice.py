# -*- coding: utf-8 -*-
# pylint: disable-msg=E1101, E0102, E0202
"""
Created on Fri Jun 15 09:14:12 2012

This module redefines the *choice* function of the *random* module, to permit
a manual performance of this operation. This is helpful to perform executions
in a guided way.

@author: T. Teijeiro
"""

def choice(seq):
    """
    Return a user-selected element from the non-empty sequence seq. If seq is
    empty, raises IndexError.
    """
    if len(seq) == 0:
        raise IndexError('Empty seq')
    elif len(seq) == 1:
        return seq[0]
    #If seq has more than one element, we show all the elements, and ask for an
    #index to return.
    for i in range(len(seq)):
        print(i,': ', seq[i])
    idx = eval(input('Select index: '))
    return seq[idx]
    
    
if __name__ == "__main__":
    import random
    seq = random.sample(range(10), 3)
    print(('Selected item: ' + str(choice(seq))))