# -*- coding: utf-8 -*-
# pylint: disable-msg=E0202, E0102, E1101, E1103, E1001
"""
Created on Fri Jan 24 09:46:28 2014

This module contains the definition of the interpretation class, which is the
basic unit of the search process which tries to solve interpretation problems.

@author: T. Teijeiro
"""
from .observable import (Observable, EventObservable, non_consecutive, between,
                         overlap_any)
from .interval import Interval as Iv
from .constraint_network import verify
import construe.knowledge.abstraction_patterns as ap
import construe.knowledge.constants as C
import construe.acquisition.obs_buffer as obsbuf
import blist
import weakref
import copy
import numpy as np
import itertools as it
from collections import deque, namedtuple as nt


class PastMetrics(nt('PastMetrics', 'time, unexp, total, abstime, nhyp')):
    """
    Tuple to store relevant information to evaluate an interpretation until
    a specific time, to allow discard old observations.
    """
    __slots__ = ()

    def diff(self, other):
        """
        Obtains the difference between two PastMetrics tuples, returned
        as a numpy array with three components. *time* attribute is excluded
        from diff.
        """
        return np.array((self.unexp - other.unexp,
                         self.total - other.total,
                         self.abstime - other.abstime,
                         self.nhyp - other.nhyp))

    def patch(self, patch):
        """
        Obtains a new PastMetrics object by applying a difference array,
        obtained by the *diff* method.

        Parameters
        ----------
        patch:
            Array, list or tuple with exactly three numerical values.
        """
        return PastMetrics(self.time, *np.array(self[1:]+patch))


class Focus(object):
    """
    This class represents the focus of attention of an interpretation, and it
    encapsulates all data and functionality related with its management.
    """
    __slots__ = ('_lst')

    def __init__(self, parent_focus=None):
        """
        Initializes a new empty focus of attention, or a shallow copy of an
        existing focus.

        Instance Properties
        -------------------
        _lst:
            Stack containing a number of tuples (observation_or_finding,
            pattern). If 'observation_or_finding' is a finding, then 'pattern'
            is the abstraction pattern generating such finding. If is an
            observation, then 'pattern' is the pattern for which the
            observation is its hypothesis, or None if it is an initial
            observation.
        """
        if parent_focus is None:
            self._lst = []
        else:
            self._lst = parent_focus._lst[:]

    def __len__(self):
        return len(self._lst)

    def __contains__(self, key):
        return any(key is v for v, _ in self._lst)

    def __nonzero__(self):
        return bool(self._lst)

    def push(self, obs, pattern):
        """
        Inserts a new observation or finding in the focus of attention.
        """
        self._lst.append((obs, pattern))

    def pop(self, n=1):
        """Removes 'n' elements from the focus of attention (1 by default)"""
        del self._lst[-n]

    @property
    def top(self):
        """Obtains the element at the top of the focus of attention"""
        return self._lst[-1]

    @top.setter
    def top(self, value):
        """Modifies the element at the top of the focus of attention"""
        self._lst[-1] = value

    @property
    def patterns(self):
        """
        Obtains an iterator over the patterns supporting the observations or
        findings in the focus of attention, starting at the top of the stack.
        """
        return (p for _, p in reversed(self._lst))

    def get_delayed_finding(self, observation):
        """
        Obtains the finding that will be matched with an observation once
        the observation is fully observed, or None if the observation will
        not be matched with a finding.
        """
        for i in xrange(len(self._lst)-1, 0, -1):
            if self._lst[i][0] is observation:
                f, p = self._lst[i-1]
                if p is not None and f is p.finding:
                    return f
                break
        return None

    def match(self, finding, obs):
        """
        Performs a matching operation between the finding at the top of the
        focus with a given observation, checking the time and value consistency
        of the matching. After consistency is checked, the finding is removed
        from the focus by means of a pop() operation.
        """
        f, pat = self._lst[-1]
        assert finding is f
        verify(obs not in pat.evidence[pat.get_evidence_type(f)[0]],
                  'Observation {0} is already in the evidence of {1} pattern',
                                                                    (obs, pat))
        patcp = copy.copy(pat)
        patcp.match(f, obs)
        #The hypothesis generating the finding is updated
        self._lst[-2] = (patcp.hypothesis, patcp)
        #And the matched finding removed from the focus
        del self._lst[-1]



class Interpretation(object):
    """
    This class represents the interpretation entity, which is a consistent
    group of abstraction hypotheses combined by the knowledge expressed in
    abstraction patterns. It is the basic entity in our search process, and
    the result of an interpretation process.
    """
    __slots__ = ('name', '_parent', 'child', 'observations', 'unintelligible',
                 'singletons', 'abstracted', 'nabd', 'focus', 'past_metrics',
                 'pred_suc', '__weakref__')

    counter = 0

    def __init__(self, parent=None):
        """
        Creates a new empty interpretation, initializing its attributes as a
        shallow copy or a direct assigment of the attributes of the parent. If
        parent is None, the attributes will be empty.

        Instance Properties
        -------------------
        name:
            Unique identificator of the interpretation.
        parent:
            Interpretation from which this one is derived, or None if this is
            a root interpretation.
        child:
            List of interpretations derived from this one.
        past_metrics:
            Summary of old information used for heuristics calculation.
        observations:
            Sortedlist containing all the observations in the interpretation,
            ordered by their start time. NOTE: This property is directly
            assigned from parent interpretation by default.
        singletons:
            Set with all Singleton hypotheses that are present in this
            interpretation. NOTE: This property is directly assigned
            from parent interpretation by default.
        abstracted:
            Set containing all the observations that are abstracted by some
            abstraction pattern in this interpretation. NOTE: This property is
            directly assigned from parent interpretation by default.
        unintelligible:
            Set containing all the observations that cannot be abstracted by
            any abstraction pattern. NOTE: This property is directly assigned
            from parent interpretation by default.
        nabd:
            Number of hypotheses in the interpretation that can be abstracted
            by a higher-level hypothesis. This value is used for the evaluation
            of the interpretation.
        focus:
            Stack containing the focus of attention of the interpretation. Each
            element in this stack is an observation or a non-matched finding
            of a pattern.
        pred_suc:
            Structure to store predecessor-successor information for
            consecutive observations. NOTE: This property is directly assigned
            from parent interpretation by default.
        """
        self.name = str(Interpretation.counter)
        if parent is None:
            self._parent = None
            self.child = []
            self.observations = blist.sortedlist()
            self.singletons = set()
            self.abstracted = set()
            self.unintelligible = set()
            self.nabd = 0
            self.past_metrics = PastMetrics(0, 0, 0, 0, 0)
            self.focus = Focus()
            self.pred_suc = {}
        else:
            self._parent = weakref.ref(parent, self._on_parent_deleted)
            self.child = []
            self.parent.child.append(self)
            self.observations = parent.observations
            self.singletons = parent.singletons
            self.abstracted = parent.abstracted
            self.unintelligible = parent.unintelligible
            self.nabd = parent.nabd
            self.past_metrics = parent.past_metrics
            self.focus = Focus(parent.focus)
            self.pred_suc = parent.pred_suc
        Interpretation.counter += 1

    def __str__(self):
        """
        Obtains the representation of the interpretation as a character string.
        """
        return self.name

    def __repr__(self):
        return self.name

    def _on_parent_deleted(self, _):
        """
        Callback function called when the parent interpretation is deleted.
        """
        self._parent = None

    def _get_types(self, obs):
        """
        Obtains a tuple with the types that are used respect to an observation,
        both as hypothesis and as evidence of different patterns.
        """
        types = {type(obs)}.union({p.get_evidence_type(obs)[0]
                                                for p in self.pat_map[obs][1]})
        dmatch = self.get_delayed_finding(obs)
        if dmatch is not None:
            types = types.union({type(dmatch)}, {p.get_evidence_type(dmatch)[0]
                                             for p in self.pat_map[dmatch][1]})
        return tuple(types)

    def _get_proper_obs(self, clazz=Observable, start=0, end=np.inf,
                                                        filt=lambda obs: True):
        """
        Obtains a list of observations matching the search criteria, ordered
        by the earliest time of the observation.

        Parameters
        ----------
        clazz:
            Only instances of the *clazz* class (or any subclass) are returned.
        start:
            Only observations whose earlystart attribute is after or equal this
            parameter are returned.
        end:
            Only observations whose lateend attribute is lower or equal this
            parameter are returned.
        filt:
            General filter provided as a boolean function that accepts an
            observation as a parameter. Only the observations satisfying this
            filter are returned.
        """
        dummy = EventObservable()
        dummy.start.value = Iv(start, start)
        idx = self.observations.bisect_left(dummy)
        if end == np.inf:
            udx = len(self.observations)
        else:
            dummy.start.value = Iv(end, end)
            udx = self.observations.bisect_right(dummy)
        return (obs for obs in it.islice(self.observations, idx, udx)
                if obs.lateend <= end and isinstance(obs, clazz) and filt(obs))

    @property
    def is_firm(self):
        """
        Checks if an interpretation is firm, that is, there are no unmatched
        findings and all the abstraction patterns involved have a sufficient
        set of evidence to support their hypothesis.
        """
        return all(p is None or p.sufficient_evidence
                                                  for p in self.focus.patterns)

    @property
    def time_point(self):
        """
        Obtains the time point of an interpretation, that is, the end time
        value of the last base evidence being considered as evidence of any
        hypothesis in the interpretation. If there are no hypothesis in the
        interpretation, then the time point is just before the first available
        observation.
        """
        lastfocus = (self.focus.top[0].earlystart - 1 if self.focus
                             else next(self.get_observations()).earlystart - 1)
        try:
            return max(max(o.lateend for o in self.abstracted), lastfocus)
        except ValueError:
            return lastfocus

    @property
    def parent(self):
        """
        Obtains the parent of an interpretation.
        """
        return self._parent() if self._parent is not None else None

    @parent.setter
    def parent(self, interpretation):
        """
        Establishes the parent of this interpretation, changing the
        corresponding references in the old and new parents.
        """
        if self._parent is not None and self in self.parent.child:
            self.parent.child.remove(self)
        if interpretation is not None:
            self._parent = weakref.ref(interpretation, self._on_parent_deleted)
            self.parent.child.append(self)
        else:
            self._parent = None

    @property
    def ndescendants(self):
        """Obtains the number of descendants of this interpretation"""
        stack = [self]
        ctr = 0
        while stack:
            ctr += 1
            interp = stack.pop()
            stack.extend(interp.child)
        return ctr

    #TODO reimplement
#    def is_mergeable(self, other):
#        """
#        Checks if two interpretations can be merged, that is, they represent
#        exactly the same interpretation from the time point in the past_metrics
#        structure.
#        """
#        nobs = len(self.observations)
#        nfocus = len(self.focus)
#        return (self is not other and
#                len(other.observations) == nobs and
#                len(other.focus) == nfocus and
#                other.singletons == self.singletons and
#                all(other.observations[i] == self.observations[i]
#                                                    for i in xrange(nobs)) and
#                all(other.focus[i] == self.focus[i] for i in xrange(nfocus)))

    def is_ancestor(self, interpretation):
        """
        Checks if a given interpretation is an ancestor in the hierarchy of
        this interpretation. The same interpretation is not considered an
        ancestor.
        """
        parent = self.parent
        while True:
            if parent is interpretation:
                return True
            elif parent is None:
                return False
            parent = parent.parent

    def update_location(self, observation):
        """
        Updates the temporal location of a certain observation in the list.
        This operation may be necessary when an observation changes its
        temporal variables values.
        """
        try:
            idx = self.observations.index(observation)
            #If it is found in its proper location, there's no need to update
            return
        except ValueError:
            idx = next(i for i in xrange(len(self.observations))
                                        if self.observations[i] is observation)
        self.observations.pop(idx)
        self.observations.add(observation)

    def non_consecutive(self, obs1, obs2):
        """
        Checks if two observations are not consecutive. This function returns
        true if it is guaranteed that in this interpretation, *obs1* and *obs2*
        are not consecutive.
        """
        types = (type(obs1), type(obs2))
        llim = obs1.earlystart
        ulim = obs2.lateend
        obs_lst = self.get_observations(start=llim, end=ulim,
                                       filt=lambda obs: isinstance(obs, types))
        return non_consecutive(obs1, obs2, obs_lst)

    def verify_consecutivity(self, obs):
        """
        Checks if an observation violates the consecutivity constraints in this
        interpretation.
        """
        #First we get the declared predecessor and successor observations.
        cons = [pat.get_consecutive(obs) for pat in self.pat_map[obs][1]]
        pred = next((p for p, _ in cons if p is not None), None)
        suc = next((s for _, s in cons if s is not None), None)
        verify(pred is None or not self.non_consecutive(pred, obs),
         'Consecutivity constraint violation between {0} and {1}', (pred, obs))
        verify(suc is None or not self.non_consecutive(obs, suc),
         'Consecutivity constraint violation between {0} and {1}', (obs, suc))
        #Now we check if the observation violates third-party consecutivities
        #and if there is any overlapping of same-observable observations.
        types = self._get_types(obs)
        #TODO filter more to reduce the size of this list. If we need two lists,
        #its OK, but for the overlapping
        #TODO test line
        start = obs.earlystart - 2560
        obs_lst = list(self.get_observations(start=start, filt=lambda ev:
            obs is not ev and
                         (isinstance(obs, type(ev)) or isinstance(ev, types))))
        if not self.is_finding(obs):
            verify(not obs_lst or not overlap_any(obs, obs_lst),
                                   '{0} overlaps another observation', (obs, ))
        dmatch = self.get_delayed_finding(obs)
        for obs1 in obs_lst:
            suc = self.get_suc(obs1)
            verify(suc is None or suc is dmatch or not between(obs1, obs, suc),
         'Consecutivity constraint violation between {0} and {1}', (obs1, suc))

    def get_pred(self, obs):
        """
        Obtains the predecessor of an observation in this interpretation, or
        None if it does not exist.
        """
        return self.pred_suc.get(obs, (None, None))[0]

    def get_suc(self, obs):
        """
        Obtains the successor of an observation in this interpretation, or
        None if it does not exist.
        """
        return self.pred_suc.get(obs, (None, None))[1]

    def get_observations(self, clazz=Observable, start=0, end=np.inf,
                                                        filt=lambda obs: True):
        """
        Obtains a list of observations matching the search criteria, ordered
        by the earliest time of the observation.

        Parameters
        ----------
        clazz:
            Only instances of the *clazz* class (or any subclass) are returned.
        start:
            Only observations whose earlystart attribute is after or equal this
            parameter are returned.
        end:
            Only observations whose lateend attribute is lower or equal this
            parameter are returned.
        filt:
            General filter provided as a boolean function that accepts an
            observation as a parameter. Only the observations satisfying this
            filter are returned.
        """
        #We perform a combination of the observations from the global buffer
        #and from the interpretation.
        geng = obsbuf.get_observations(clazz, start, end, filt)
        genl = self._get_proper_obs(clazz, start, end, filt)
        dummy = EventObservable()
        dummy.start.value = Iv(np.inf, np.inf)
        nxtg = next(geng, dummy)
        nxtl = next(genl, dummy)
        while True:
            nxt = min(nxtg, nxtl)
            if nxt is dummy:
                return
            elif nxt is nxtg:
                nxtg = next(geng, dummy)
            else:
                nxtl = next(genl, dummy)
            yield nxt

    def remove_old(self, time=None):
        """Removes old observations from the interpretation."""
        if time is None:
            time = max(self.past_metrics.time,
                     min(o.earlystart for o in self.focus) - C.FORGET_TIMESPAN)
        i = 0
        unexp = total = abstime = 0.0
        while i < len(self.observations):
            obs = self.observations[i]
            if obs.lateend < time:
                hypat, _ = self.pat_map.pop(obs)
                if hypat is not None:
                    self.patterns.remove(hypat)
                total += 1
                if ap.get_obs_level(type(obs)) == 0:
                    abstime += obs.earlyend - obs.latestart + 1
                if self.observations is self.parent.observations:
                    self.observations = self.parent.observations[:]
                self.observations.pop(i)
            else:
                i += 1
        nhyp = total
        for obs in obsbuf.get_observations(end=time-1,
                         filt=lambda ob: ob.lateend >= self.past_metrics.time):
            total += 1
            if self.get_abstraction_pattern(obs) is not None:
                abstime += obs.earlyend - obs.latestart + 1
            else:
                unexp += 1
        self.unintelligible = {obs for obs in self.unintelligible
                                                        if obs.lateend >= time}
        self.past_metrics = PastMetrics(time, self.past_metrics.unexp + unexp,
                                           self.past_metrics.total + total,
                                           self.past_metrics.abstime + abstime,
                                           self.past_metrics.nhyp + nhyp)

    def recover_old(self):
        """
        Recovers old observations from the ancestor interpretations,
        in order to have the full interpretation from the beginning of the
        process.
        """
        allobs = set(self.observations)
        interp = self.parent
        while interp is not None:
            allobs |= set(interp.observations) - set(interp.focus)
            interp = interp.parent
        allobs = blist.sortedlist(allobs)
        #Duplicate removal (set only prevents same references, not equality)
        i = 0
        while i < len(allobs)-1:
            obs = allobs[i]
            while allobs[i+1] == obs:
                allobs.pop(i+1)
                if i == len(allobs)-1:
                    break
            i += 1
        self.observations = blist.sortedlist(allobs)
        self.unintelligible |= set(obsbuf.get_observations(
                                                    end=self.past_metrics.time,
                     filt=lambda ob: self.get_abstraction_pattern(ob) is None))

    def detach(self, reason=''):
        """
        Detachs this interpretation from the interpretations tree, being from
        that moment a new root.
        """
        #Uncomment to debug.
        #print str(self), reason
        if self.parent is not None:
            parent = self.parent
            parent.child.remove(self)
            self.parent = None

    def discard(self, reason=''):
        """
        Discards this interpretation, and recursively all the descendant
        interpretations.
        """
        self.detach(reason)
        stack = self.child[:]
        while stack:
            interp = stack.pop()
            interp.detach('Parent interpretation discarded')
            stack.extend(interp.child)

    def get_child(self, name):
        """
        Obtains the child interpretation of this one with the given id.
        """
        name = str(name)
        queue = deque([self])
        while queue:
            head = queue.popleft()
            if head.name == name:
                return head
            for subbr in head.child:
                queue.append(subbr)
        raise ValueError('No child interpretation with such name')
