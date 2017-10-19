# -*- coding: utf-8 -*-
# pylint: disable-msg=
"""
Created on Thu Dec 19 18:02:37 2013

This module contains the model definition for the automata-based abstraction
patterns. Since abstraction patterns are defined by regular grammars, the
internal representation is based on finite automatas.

@author: T. Teijeiro
"""

from .FreezableObject import FreezableObject
from .observable import Observable
from .constraint_network import verify
from collections import defaultdict
import itertools as it


ABSTRACTED = True
ENVIRONMENT = False

def NULL_CONST(pattern, obs):
    """Default constraint definition"""
    return None

def NULL_PROC(pattern):
    """Default observation procedure"""
    return None

def BASIC_TCONST(pattern, obs):
    """
    This function defines the basic temporal constraints that should contain
    any observation within an abstraction pattern. This constraints set that
    the beginning of an observation has to occur before its ending.
    """
    if obs.start is not obs.time:
        pattern.last_tnet.set_before(obs.start, obs.time)
    if obs.time is not obs.end:
        pattern.last_tnet.set_before(obs.time, obs.end)


class Transition(FreezableObject):
    """
    Model for the transition of an automata, allowing the attribute
    specification as a function on a partial-recognized pattern. It also
    includes a specific attribute to indicate the property of the observable
    that allows the transition as an Abstracted observation or an Environment
    observation.
    """
    __slots__ = ('istate', 'fstate', 'observable', 'abstracted', 'tconst',
                 'gconst')

    def __init__(self, istate=None, fstate=None, observable=None,
                       abstracted=ABSTRACTED, tconst=BASIC_TCONST,
                                                            gconst=NULL_CONST):
        """
        Creates a new transition that can be added to a DFA definition. All the
        attributes of the transition must be set on the creation, and no
        posterior modifications are allowed.

        Parameters
        ----------
        istate:
            Initial state of the transition.
        fstate:
            Final state of the transition.
        observable:
            Observable type that allows the transition. It can be None, to
            determine such transitions that simply add constraints, but no
            new observations.
        abstracted:
            Flag that indicates if the observation that allows the transition
            should be considered as an Abstracted observation or an Environment
            observation.
        tconst:
            Function that receives as a parameter the AbstractionPattern object,
            and adds the temporal constraints introduced by this transition
            with the rest of the variables of the pattern, including the
            hypothesis. These constraints are added before the matching of the
            predicted finding with an actual observation.
        gconst:
            Function that receives as a parameter the AbstractionPattern object,
            and checks any additional constraint in the value of the
            observations. These constraints are checked after the matching of
            the finding with an actual observation.
        """
        super(Transition, self).__init__()
        if istate is None or fstate is None:
            raise ValueError('Initial and final states must be != None')
        self.istate = istate
        self.fstate = fstate
        self.observable = observable
        self.abstracted = abstracted
        self.tconst = tconst
        self.gconst = gconst
        #We don't allow any posterior modification of a transition.
        self.freeze()


    def __str__(self):
        """Obtains the representation of a transition as a string"""
        return '{0} --{1}{2}--> {3}'.format(self.istate, self.observable,
                                 '@' if self.abstracted == ABSTRACTED else '#',
                                                                   self.fstate)

    def __repr__(self):
        return str(self)


class PatternAutomata(FreezableObject):
    """
    This class represents an automata created by adding *Transition* instances.
    It also includes the definition of the final states of the pattern, and
    the Observable class that represents the hypothesis of the pattern.
    """
    __slots__ = ('name', 'Hypothesis', 'transitions', 'abstractions',
                 'final_states', 'obs_proc')

    def __init__(self):
        """
        Creates a new empty pattern automata.

        Instance properties
        -------------------
        name:
            String representing the name of the pattern.
        Hypothesis:
            Observable type that constitutes the hypothesis of the pattern.
        transitions:
            List of transitions of the pattern automata.
        abstractions:
            Dictionary that maps each abstracted observable with the
            transitions that can be used to perform that abstraction in an
            abductive way.
        final_states:
            Set of states of the automata that are final or accepting states.
        obs_proc:
            Observation procedure that is executed once the necessary evidence
            for the pattern has been observed. It receives a single parameter,
            the abstraction pattern to apply the procedure.
        """
        super(PatternAutomata, self).__init__()
        self.name = ''
        self.Hypothesis = Observable
        self.transitions = []
        self.abstractions = defaultdict(tuple)
        self.final_states = set()
        self.obs_proc = NULL_PROC

    def __str__(self):
        return self.name

    def __repr__(self):
        return self.name if len(self.name) > 0 else object.__repr__(self)

    def freeze(self):
        """We override the freeze method to check additional constraints"""
        verify(self.final_states, 'The automata does not have any final state')
        #We use a tuple to avoid posterior changes on the transitions
        self.transitions = tuple(self.transitions)
        super(PatternAutomata, self).freeze()

    def add_transition(self, istate=None, fstate=None, observable=None,
                             abstracted=ABSTRACTED, tconst=NULL_CONST,
                                                            gconst=NULL_CONST):
        """
        Adds a new *Transition* to this automata.

        Parameters
        ----------
        istate:
            Initial state of the transition.
        fstate:
            Final state of the transition.
        observable:
            Observable type that allows the transition.
        abstracted:
            Flag that indicates if the observation that allows the transition
            should be considered as an Abstracted observation or an Environment
            observation.
        tconst:
            Function that receives as a parameter the AbstractionPattern object,
            and adds the temporal constraints introduced by this transition
            with the rest of the variables of the pattern, including the
            hypothesis. These constraints are added before the matching of the
            predicted finding with an actual observation.
        gconst:
            Function that receives as a parameter the AbstractionPattern object,
            and checks any additional constraint in the value of the
            observations. These constraints are checked after the matching of
            the finding with an actual observation.
        """
        assert abstracted in (ABSTRACTED, ENVIRONMENT), 'Invalid abstraction'
        transition = Transition(istate, fstate, observable, abstracted, tconst,
                                                                        gconst)
        self.transitions.append(transition)
        #All states (except the initial one) must be reached by at least one
        #transition.
        for state in self.states:
            if state != self.start_state:
                verify(self.tr_to(state))

    @property
    def states(self):
        """Obtains the set of states of this automata"""
        return set(it.chain.from_iterable((t.istate, t.fstate)
                                                    for t in self.transitions))

    @property
    def start_state(self):
        """Obtains the initial state of this automata"""
        if not self.transitions:
            raise IndexError('This automata does not contain any state')
        return self.transitions[0].istate

    @property
    def abstracted(self):
        """Obtains the set of observables abstracted by this pattern"""
        return {t.observable for t in self.transitions
                    if t.abstracted is ABSTRACTED and t.observable is not None}

    @property
    def environment(self):
        """
        Obtains the set of observables that are the environment of this
        pattern.
        """
        return {t.observable for t in self.transitions
                   if t.abstracted is ENVIRONMENT and t.observable is not None}
    @property
    def manifestations(self):
        """
        Obtains the set of observables over which this pattern is created. It
        corresponds to the alphabet of the pattern in the classical DFA
        nomenclature.
        """
        return set.union(self.abstracted, self.environment)

    def tr_from(self, state):
        """
        Obtains the list of transitions starting in a given state.
        """
        return [t for t in self.transitions if t.istate == state]

    def tr_to(self, state):
        """
        Obtains the list of transitions finishing in a given state.
        """
        return [t for t in self.transitions if t.fstate == state]

if __name__ == "__main__":
    #Small test for automata creation.
    DFA = PatternAutomata()
    DFA.add_transition('S', 'A', 'a', ENVIRONMENT)
    DFA.add_transition('A', 'A', 'a')
    DFA.add_transition('A', 'B', 'b')
    DFA.add_transition('B', 'B', 'b')
    DFA.add_transition('B', 'C', 'c')
    DFA.add_transition('C', 'D', 'c')
    DFA.add_transition('D', 'E', 'd')
    DFA.add_transition('E', 'E', 'd')
    DFA.add_transition('E', 'F', 'e')
    DFA.add_transition('F', 'G', 'f')
    DFA.add_transition('G', 'F', 'e')
    DFA.add_transition('F', 'H', 'f', ENVIRONMENT)
    DFA.final_states.add('H')
    DFA.freeze()
