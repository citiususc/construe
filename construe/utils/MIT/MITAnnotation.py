# -*- coding: utf-8 -*-
# pylint: disable-msg=E1101, E0102, E0202
"""
Created on Mon Feb 27 12:08:29 2012

Module with definition of classes to work with annotations in the MIT format.

@author: T. Teijeiro
"""
import struct

class MITAnnotation(object):
    """
    This class represents an annotation in the MIT format. Currently only
    standard 2-byte annotations are supported.
    """

    #We use slots instead of __dict__ to save memory space when a lot of
    #annotations are created and managed.
    __slots__ = ('code', 'time', 'subtype', 'chan', 'num', 'aux')

    def __init__(self, code=0, time=0, subtype=0, chan=0, num=0, aux=None):
        self.code = code
        self.time = time
        self.subtype = subtype
        self.chan = chan
        self.num = num
        self.aux = aux


    def __str__(self):
        return '{0} {1} {2} {3} {4} {5}'.format(self.time,    self.code,
                                                self.subtype, self.chan,
                                                self.num,     repr(self.aux))

    def __repr__(self):
        return str(self)

    def __lt__(self, other):
        return self.time < other.time

#MIT format special codes
SKIP_TIME = 1023
AUX_CODE = 63
SKIP_CODE = 59
NUM_CODE = 60
SUB_CODE = 61
CHN_CODE = 62


def is_qrs_annotation(annot):
    """
    Checks if an annotation corresponds to a QRS annotation.
    NORMAL
    UNKNOWN
    SVPB
    FUSION
    VESC
    LEARN
    AESC
    NESC
    SVESC
    PVC
    BBB
    LBBB
    ABERR
    RBBB
    NPC
    PACE
    PFUS
    APC
    RONT
    """
    return annot.code in (1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13,
                          25, 30, 34, 35, 38, 41)


def read_annotations(path):
    """
    Reads an annotation file in the MIT format.
    See: http://www.physionet.org/physiotools/wag/annot-5.htm

    Parameters
    ----------
    path:
        Path to the file containing the annotations.

    Returns
    -------
    out:
        List for annotation objects
    """
    result = []
    f = open(path, 'rb')
    num = 0
    chan = 0
    displ = 0
    while True:
        bann = f.read(2)
        if not bann:
            break
        (b0, b1) = struct.unpack('bb', bann)
        A = (b1 & 0xff) >> 2
        I = ((b1 & 0x03) << 8) | (0xff & b0)
        #Special codes parsing
        if A == SKIP_CODE and I == 0:
            (b0, b1, b2, b3) = struct.unpack('4b', f.read(4))
            displ = ((b1 << 24) | ((b0 & 0xff) << 16) |
                                  ((b3 & 0xff) << 8)  | (b2 & 0xff))
        elif A is NUM_CODE:
            num = I
            result[-1].num = num
        elif A is SUB_CODE:
            result[-1].subtype = I
        elif A is CHN_CODE:
            chan = I
            result[-1].chan = chan
        elif A is AUX_CODE:
            result[-1].aux = f.read(I)
            if I % 2 != 0:
                f.read(1)
        elif A == I == 0:
            break
        else:
            result.append(MITAnnotation(code=A, time=I+displ, chan=chan, num=num))
            displ = 0
    f.close()
    #Now, for each annotation we put the absolute time
    abs_time = 0
    for annot in result:
        abs_time += annot.time
        annot.time = max(0, abs_time)
    return result

def save_annotations(annots, path):
    """
    Saves a list of annotations in a file, in the MIT format.
    See: http://www.physionet.org/physiotools/wag/annot-5.htm

    Parameters
    ----------
    annots: List of MITAnnotation objects to be saved. It is sorted before
    the writing operation.
    path: Path to the file where the list is saved.
    """
    annots = sorted(annots)
    f = open(path, 'wb')
    prev_time = 0
    prev_num = 0
    prev_chn = 0


    for anot in annots:
        rel_time = anot.time - prev_time
        #If the distance is greater than 1023 (what we can express with 10
        #bits), we should write an skip code in the file.
        if rel_time > SKIP_TIME:
            #A = SKIP_CODE, I = 0; Then 4 byte PDP-11 long integer
            f.write(struct.pack('>H', SKIP_CODE << 2))
            f.write(struct.pack('<H', rel_time >> 16))
            f.write(struct.pack('<H', rel_time & 0xFFFF))
            #The next written position is 0
            rel_time = 0
        #We write the annotation code and the timestamp
        f.write(struct.pack('<H', anot.code << 10 | rel_time))
        prev_time = anot.time
        #Write the NUM annotation, if changes
        if anot.num != prev_num:
            f.write(struct.pack('<H', NUM_CODE << 10 | anot.num))
            prev_num = anot.num
        #Write the SUBTYPE annotation, if != 0
        if anot.subtype != 0:
            f.write(struct.pack('<H', SUB_CODE << 10 | anot.subtype))
        #Write the CHAN annotation, if changes
        if anot.chan != prev_chn:
            f.write(struct.pack('<H', CHN_CODE << 10 | anot.chan))
            prev_chn = anot.chan
        #Write the AUX field, if present
        if anot.aux != None:
            f.write(struct.pack('<H', AUX_CODE << 10 | len(anot.aux)))
            aux = (anot.aux if isinstance(anot.aux, bytes)
                                        else bytes(anot.aux, encoding='utf-8'))
            f.write(aux)
            if len(anot.aux) % 2 != 0:
                f.write(struct.pack('<b', 0))
    #Finish the file with a 00
    f.write(struct.pack('<h', 0))
    f.close()


def convert_annots_freq(spath, sfreq, dpath, dfreq):
    """
    Converts the frequency of an annotations file.

    Parameters
    ----------
    spath:
        Path of the input annotations file.
    sfreq:
        Frequency in Hz used in the input annotations timing.
    dpath:
        Path where the new annotations will be stored.
    dfreq:
        Frequency in Hz used in the output annotations timing.
    """
    annots = read_annotations(spath)
    for ann in annots:
        ann.time = int(ann.time/float(sfreq)*float(dfreq))
    save_annotations(annots, dpath)



if __name__ == "__main__":
    pass
