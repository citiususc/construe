# -*- coding: utf-8 -*-
# pylint: disable-msg=
"""
Created on Tue Feb  4 17:59:59 2014

This module contains the definition of different knowledge-related constants,
like the limits in the duration of waves or intervals in different ECG patterns

@author: T. Teijeiro
"""

import construe.acquisition.signal_buffer as SIG
from construe.model import Interval as Iv
from construe.utils.units_helper import (msec2samples as m2s,
                                         phys2digital as p2d)
import math

########################
### Global constants ###
########################

#Time span after which we can forget old observations, but always keeping a
#minimum number.
FORGET_TIMESPAN = m2s(10000)
MIN_NOBS = 30

#####################################
### Precision related constraints ###
#####################################

#Temporal margin for measurement discrepancies (1 mm in standard scale)
TMARGIN = int(math.ceil(m2s(40)))

############################################
### Beat Annotations related constraints ###
############################################

RDEFLECTION_MIN_DIST = m2s(80)         #Minimum distance for different annotations


#######################################
### QRS-complex related constraints ###
#######################################

NQRS_DUR = Iv(m2s(15), m2s(200))    #Normal QRS complex duration
VQRS_DUR = Iv(m2s(120), m2s(400))   #Ventricular QRS complex duration
QRS_DUR = NQRS_DUR.hull(VQRS_DUR)   #QRS complex duration
SPIKE_DUR = int(round(m2s(30)))              #Maximum duration of a pace spike.
SPIKE_EDGE_AMP = p2d(0.2)      #Minimum amplitude of each edge of a pace spike.
SPIKE_ECGE_DIFF = p2d(0.1)      #Maximum amplitude differences for spike edges.
#Minimum distances between the begin of the complex and the peak, and between
#the peak and the end
QRS_START_PK = m2s(5)
QRS_PK_END = m2s(40)
#Distances between R-Deflections and the starting of the complex.
QRS_RDEF_DMAX = m2s(80)
#Maximum difference in the amplitude to consider a signal fragment as a missed
#QRS (compared with an actual detected QRS)
MISSED_QRS_MAX_DIFF = 0.5


################################
### P Wave related constants ###
################################
PW_DURATION = Iv(m2s(40), m2s(250)) #P Wave duration limits
PQ_INTERVAL = Iv(m2s(20), m2s(200)) #PQ Interval duration limits
PR_INTERVAL = Iv(m2s(80), m2s(400)) #PR Interval duration limits
PW_DEF_DUR = Iv(0, m2s(200))       # Deflection duration for a P Wave
PW_DEF_OVER = m2s(350)             #P and Deflection overlapping limits
PQ_DEF_SEP = Iv(m2s(20), m2s(240)) #P Deflection and QRS separation
PR_DEF_SEP = Iv(m2s(20), m2s(400))

#Leads with visible P waves
PWAVE_LEADS = (SIG.Leads.MLI, SIG.Leads.MLII, SIG.Leads.MLIII, SIG.Leads.V1,
               SIG.Leads.V2, SIG.Leads.V3, SIG.Leads.V4, SIG.Leads.V5)

#P wave amplitude limits
PWAVE_AMP = {SIG.Leads.MLI : p2d(0.75), SIG.Leads.MLII : p2d(0.75),
             SIG.Leads.MLIII: p2d(0.75),
             SIG.Leads.V1 : p2d(0.5), SIG.Leads.V2 : p2d(0.5),
             SIG.Leads.V3 : p2d(0.5), SIG.Leads.V4 : p2d(0.5),
             SIG.Leads.V5 : p2d(0.5)}
PWAVE_MIN_AMP = p2d(0.05)
#Temporal environment of a P wave to check baseline stability.
PWAVE_ENV = m2s(80)


################################
### T Wave related constants ###
################################
TW_DURATION = Iv(m2s(80), m2s(450)) #T Wave duration limits
ST_INTERVAL = Iv(m2s(0), m2s(250))  #ST segment duration limits
QT_INTERVAL = Iv(m2s(250), m2s(900))#QT maximum limits (not normal)
#Maximum limits from the end of the QRS to the end of the T Wave (not normal)
SQT_INTERVAL = Iv(m2s(150), m2s(750))
#Minimum interval between the end of a T Wave and the beginning of the next QRS
TQ_INTERVAL_MIN = m2s(40)
TW_DEF_OVER_MIN = m2s(300)         #T and Energy interval overlapping limits
TW_DEF_OVER_MAX = m2s(450)
TW_DEF_ENDIFF = m2s(80)
TW_RDEF_MIN_DIST = m2s(80)    #Minimum distance from an abstracted Bann to Tend
#Factor limiting the maximum slope of a T Wave wrt the corresponding QRS.
TQRS_MAX_DIFFR = 0.7
#Maximum amplitude distance between the end of the T wave and the baseline if
#the delineation begins at the end of the temporal support of the T wave.
TWEND_BASELINE_MAX_DIFF = p2d(0.3)
#Constant values for the Kalman Filter used for QT (actually RT) measure
QT_ERR_STD = m2s(58) #Standard deviation of the QT error (R matrix)
MIN_QT_STD = m2s(20) #Minimum standard deviation of the QT error
KF_Q = m2s(40) #Dynamic noise of the Kalman filter (Q matrix)
#Upper and lower limit of the RR intervals
QTC_RR_LIMITS = Iv(m2s(300), m2s(1200))


#####################################################
### Regular rhythms related constants (normality) ###
#####################################################

N_PR_INTERVAL = Iv(m2s(100), m2s(300))  #Normal PR segment duration limits.
N_QT_INTERVAL = Iv(0, m2s(520))         #Normal QT interval duration limits.
SINUS_RR = Iv(m2s(475), m2s(1200))      #Normal rhythm RR limits (50-120 bpm)
BRADY_RR = Iv(m2s(1000), m2s(2000))     #Bradycardia RR limits (30-60 bpm)
TACHY_RR = Iv(m2s(200), m2s(600))       #Tachycardia RR limits (100-300 bpm)
RR_MAX_DIFF = m2s(200)                  #Maximum instantaneous RR variation.
RR_MAX_CV = 0.15                        #Maximum RR coefficient of variation.
#Normal QT interval duration limits. It takes as parameter the RR interval, and
#it applies the linear regression model from:
#http://www.ncbi.nlm.nih.gov/pmc/articles/PMC1767037/
def QT_FROM_RR(rr):
    """
    Returns the interval of acceptable QT durations with the given RR
    intervals. It applies a linear regression model with the coefficients
    obtained from the referenced study.
    """
    return Iv(m2s(220)+0.1*rr.start, m2s(240)+0.25*rr.end)

######################################
### Extrasystole-related constants ###
######################################

COMPAUSE_MIN_DUR = m2s(100)   #Minimum RR duration of the compensatory pause
COMPAUSE_MIN_F = 1.5          #Min and max factor on the preceding RR for the
COMPAUSE_MAX_F = 2.5          #compensatory pause.
COMPAUSE_RREXT_MIN = m2s(225)  #Minimum RR extension of the compensatory pause
COMPAUSE_RREXT_MIN_F = 1.25   #Min and max RR extension factors of the
COMPAUSE_RREXT_MAX_F = 4.0    #compensatory pause wrt the advanced beat.
ICOUPLET_MIN_RREXT_F = 1.25   #Min RR extension factor after a couplet.
ICOUPLET_MIN_RREXT = m2s(225) #Minimum pause after a couplet.
ICOUPLET_MAX_DIFF = m2s(150)  #Maximum RR variation inside a couplet.
ICOUPLET_RCHANGE = m2s(100)   #Minimum RR variation of the second extrasystole.

##################################
### Asystole related constants ###
##################################

ASYSTOLE_RR = Iv(m2s(2000), m2s(30000))

#############################################
### Ventricular flutter related constants ###
#############################################

VFLUT_MIN_DUR = m2s(2000)   #Minimum duration of a ventricular flutter
VFLUT_WW_INTERVAL = Iv(m2s(180), m2s(500)) #Flutter wave separation.
VFLUT_LIM_INTERV = Iv(m2s(180), m2s(1000)) #Begin and end transition duration

#############################################
### Atrial fibrillation related constants ###
#############################################

AFIB_RR_MINSTD = 0.08         #Minimum coefficient of variation of RR in afib
#Maximum temporal separation wrt a previous afib to use its parameters
AFIB_MAX_DELAY = m2s(10000)
AFIB_MIN_NQRS = 8

####################################
### Deflection related constants ###
####################################

DEF_DUR = Iv(0, m2s(500))


