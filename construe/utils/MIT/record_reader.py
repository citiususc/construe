# -*- coding: utf-8 -*-
"""
Created on Thu Feb  2 16:40:43 2012

Utility module to read MIT records.
"""


__author__ = "T. Teijeiro"
__date__ = "$30-nov-2011 18:01:49$"


import numpy
import construe.acquisition.signal_buffer as SIG
import dateutil.parser
from subprocess import check_output

class MITRecord(object):
    """
    This class includes the information related to a record in MIT-BIH format,
    including the number of signals and their sampling frequency.
    """
    def __init__(self):
        self.signal = None
        self.frequency = 0.0
        self.gain = 0.0
        self.leads = []

    @property
    def length(self):
        """Obtains the length, in samples, of the longest signal"""
        return max(len(self.signal[i]) for i in range(len(self.leads)))

def get_leads(record_path):
    """Obtains a list with the name of the leads of a specific record."""
    signals = check_output(['signame', '-r', record_path]).splitlines()
    signals = [s.decode("utf-8") for s in signals]
    return [s for s in signals if s in SIG.VALID_LEAD_NAMES]

def get_datetime(record_path):
    """Obtains the datetime representing the beginning of a record"""
    datestr = check_output(['wfdbtime', '-r', record_path, '0'])
    datestr = datestr[datestr.index('[')+1:datestr.index(']')]
    return dateutil.parser.parse(datestr, dayfirst=True)

def get_gain(record_path):
    """
    Obtains the ADC Gain of a specific record. Currently only a single
    Gain value is supported, so all signals in a record must have the same
    value.
    """
    sigdesc = [s.strip()
                 for s in check_output(['wfdbdesc', record_path]).splitlines()]
    sigdesc = [s.decode("utf-8") for s in sigdesc]
    gains = set()
    for desc in sigdesc:
        if desc.startswith('Gain:'):
            try:
                gains.add(float(desc.split()[1]))
            except ValueError:
                pass
    if len(gains) > 1:
        raise ValueError('Found more than one different ADC Gain value for '
                         'the given record. Currently only one value is '
                         'supported')
    elif len(gains) == 0:
        return 200.0
    else:
        return gains.pop()

def get_sampling_frequency(record_path):
    """Obtains the base sampling frequency of a record."""
    return float(check_output(['sampfreq', record_path]))


def load_MIT_record(record_path, physical_units=False, multifreq=False):
    """
    Loads a MIT-BIH record using rdsamp. The correct number of signals in the
    file must be passed as argument to ensure a correct load.

    Parameters
    ----------
    record_path:
        Path to the record header file.
    physical_units:
        Flag to indicate if the input signals have to be read in physical
        units instead of digital values.

    Returns
    -------
    out:
        Matrix with the signal, with one row for each signal, and a column
        for each sample.
    """
    #First we obtain the recognized signals in the record
    leads = get_leads(record_path)
    if not leads:
        raise ValueError('None of the signals in the {0} record is '
                           'recognizable as an ECG signal'.format(record_path))
    num_signals = len(leads)
    #We obtain the string representation of the record
    command = ['rdsamp', '-r', record_path]
    if physical_units:
        command.append('-P')
    if multifreq:
        command.append('-H')
    #We load only the recognized signal names.
    command.append('-s')
    command.extend(leads)
    string = check_output(command).decode('utf-8')
    if physical_units:
        #HINT Bug in some cases with physical units conversion in rdsamp.
        string = string.replace('-', '-0')
    #Convert to matrix
    mat = numpy.fromstring(string, sep='\t')
    #We reshape it according to the number of signals + 1 (the first column)
    #is the number of sample, but it is not of our interest.
    mat = mat.reshape(((len(mat) // (num_signals + 1)), num_signals + 1))
    result = MITRecord()
    #We remove the first column, and transpose the result
    result.signal = mat[:, 1:].T
    #We include the loaded leads
    result.leads = leads
    #The sampling frequency
    result.frequency = get_sampling_frequency(record_path)
    #And the ADC Gain
    result.gain = 1.0 if physical_units else get_gain(record_path)
    return result


if __name__ == "__main__":
    pass
