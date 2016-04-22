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
from collections import deque, defaultdict, namedtuple as nt


class PastMetrics(nt('PastMetrics', 'time, unexp, total, abstime, nhyp')):
    """
    Tuple to store relevant information to evaluate an interpretation until
    a specific time, to allow discard old observations.
    """

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


class Interpretation(object):
    """
    This class represents the interpretation entity, which is a consistent
    group of abstraction hypotheses combined by the knowledge expressed in
    abstraction patterns. It is the basic entity in our search process, and
    the result of an interpretation process.
    """
    counter = 0

    def __init__(self, parent=None):
        """
        Creates a new empty interpretation, initializing its attributes as a
        shallow copy of the attributes of the parent. If parent is None,
        the attributes will be empty.

        Instance Properties
        -------------------
        name:
            Unique identificator of the interpretation.
        parent:
            Interpretation from which this one is derived, or None if this is
            a root interpretation.
        child:
            List of interpretations derived from this one.
        observations:
            Sortedlist containing all the observations in the interpretation,
            ordered by their start time.
        unintelligible:
            Sortedlist containing all the observations that cannot be
            abstracted by any abstraction pattern.
        patterns:
            List of *AbstractionPattern* instances that have generated the
            hypotheses in this interpretation.
        pat_map:
            Dictionary to map observations with abstraction patterns. Each
            value in this dictionary is a 2-tuple *(hyp_pat,{ev_pats})*, where
            hyp_pat is the pattern hypothesizing the key observation, and
            ev_pats is the set of patterns for which the key observation
            belongs to their evidence.
        focus:
            Stack containing the focus of attention of the interpretation. Each
            element in this stack is an observation or a non-matched finding
            of a pattern.
        delay_match:
            Associations between observations that are postponed until the
            matched observation has sufficient evidence supporting it.
        """
        self.name = str(Interpretation.counter)
        if parent is None:
            self._parent = None
            self.child = []
            self.observations = blist.sortedlist()
            self.singletons = set()
            self.unintelligible = set()
            self.past_metrics = PastMetrics(0, 0, 0, 0, 0)
            self.patterns = []
            self.focus = []
            self.pat_map = defaultdict(lambda: (None, frozenset()))
            self.delay_match = []
        else:
            self._parent = weakref.ref(parent, self._on_parent_deleted)
            self.child = []
            self.parent.child.append(self)
            self.observations = parent.observations[:]
            self.singletons = parent.singletons.copy()
            self.unintelligible = parent.unintelligible.copy()
            self.past_metrics = parent.past_metrics
            self.patterns = parent.patterns[:]
            self.focus = parent.focus[:]
            self.pat_map = parent.pat_map.copy()
            self.delay_match = parent.delay_match[:]
            #TODO remove this
            for obs in parent.pat_map:
                obs.freeze()
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
        both as hypothesis and as evidence of different patterns. If the
        observation is assigned to a delayed match
        """
        types = {type(obs)}.union({p.get_evidence_type(obs)[0]
                                                for p in self.pat_map[obs][1]})
        idx = self.delayed_idx(obs)
        if idx > -1:
            dmatch = self.delay_match[idx][0]
            types = types.union({type(dmatch)}, {p.get_evidence_type(dmatch)[0]
                                             for p in self.pat_map[dmatch][1]})
        return tuple(types)

    def _save(self, obs, memo=None):
        """
        Performs a safe local copy of an observation in this interpretation,
        ensuring that modifications to the attributes of the observation will
        not affect to any ancestor interpretation. The function returns the
        copy observation that replaces the argument. If the copy is unnecessary
        the function returns the same observation.

        **Note:** This function may have expensive side effects, since all
        patterns related to the observation must be copied, and that copies
        must be propagated.

        Parameters
        ----------
        obs:
            Observation to be saved.
        memo:
            Optional dictionary (not intended to be used by first caller) to
            avoid side effects in the recursive calls to the function.

        Returns
        -------
        out:
            Observation that is safe to modify withouth affecting to ancestor
            interpretations.
        """
        #TODO manage circular references between patterns
        memo = {} if memo is None else memo
        if obs in memo:
            return memo[obs]
        if not self.parent or not obs in self.parent.pat_map:
            obscp = obs
        else:
            assert obs in self.pat_map, self.parent
            ev_pat = self.get_evidence_patterns(obs)
            while True:
                hyp = next((p.hypothesis for p in ev_pat
                                            if p.hypothesis not in memo), None)
                if hyp is None:
                    break
                self._save(hyp, memo)
            to_update = frozenset(
                               {self.get_hypothesis_pattern(memo[p.hypothesis])
                                                              for p in ev_pat})
            hypat = self.get_hypothesis_pattern(obs)
            if hypat is None:
                obscp = copy.deepcopy(obs)
                self.pat_map[obscp] = (None, to_update)
            else:
                hypatcp = copy.copy(hypat)
                obscp = hypatcp.hypothesis
                self.pat_map[obscp] = (hypatcp, to_update)
                self.replace_pat(hypat, hypatcp)
            for pat in to_update:
                pat.replace(obs, obscp)
            self.replace_obs(obs, obscp)
        obscp.unfreeze()
        memo[obs] = obscp
        return obscp

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
        return all(p.sufficient_evidence for p in self.patterns)

    @property
    def time_point(self):
        """
        Obtains the time point of an interpretation, that is, the end time
        value of the last base evidence being considered as evidence of any
        hypothesis in the interpretation. If there are no hypothesis in the
        interpretation, then the time point is just before the first available
        observation.
        """
        lastfocus = (self.focus[-1].earlystart - 1 if self.focus
                             else next(self.get_observations()).earlystart - 1)
        try:
            return max(max(o.lateend for o, v in self.pat_map.iteritems()
                                                           if v[1]), lastfocus)
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

    def is_mergeable(self, other):
        """
        Checks if two interpretations can be merged, that is, they represent
        exactly the same interpretation from the time point in the past_metrics
        structure.
        """
        nobs = len(self.observations)
        nfocus = len(self.focus)
        return (self is not other and
                len(other.observations) == nobs and
                len(other.focus) == nfocus and
                other.singletons == self.singletons and
                all(other.observations[i] == self.observations[i]
                                                    for i in xrange(nobs)) and
                all(other.focus[i] == self.focus[i] for i in xrange(nfocus)))

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

    def conjecture(self, observation, hyp_pattern):
        """
        Adds an observation as a new conjecture to this interpretation. The
        abstraction pattern generating this conjecture as its hypothesis will
        be also added to the pattern list, and the hypothesis binding will
        be set.

        Parameters
        ----------
        observation:
            New observation that will be published in this interpretation.
        hyp_pattern:
            Abstraction pattern generating *observation* as its hypothesis.
        """
        assert hyp_pattern is not None
        self.observations.add(observation)
        self.pat_map[observation] = (hyp_pattern, frozenset())
        self.patterns.append(hyp_pattern)

    def update_location(self, observation):
        """
        Updates the temporal location of a certain observation in the list.
        This operation may be necessary when an observation changes its
        temporal variables values.
        """
        try:
            idx = self.observations.index(observation)
        except ValueError:
            idx = next(i for i in xrange(len(self.observations))
                                        if self.observations[i] is observation)
        self.observations.pop(idx)
        self.observations.add(observation)

    def replace_pat(self, old, new):
        """
        Replaces one abstraction pattern instance by another in the data
        structures of this interpretation, deleting all references to the
        replaced one.

        Parameters
        ----------
        old:
            Abstraction pattern to be replaced.
        new:
            Replacement abstraction pattern.
        """
        self.patterns[self.patterns.index(old)] = new
        self.pat_map[old.hypothesis] = (new, self.pat_map[old.hypothesis][1])
        for obs in it.chain.from_iterable(old.evidence.itervalues()):
            hyp, ev_set = self.pat_map[obs]
            self.pat_map[obs] = (hyp, ev_set.union({new}) - {old})

    def replace_obs(self, old, new):
        """
        Replaces one observation by another in the data structures of this
        interpretation, deleting all references to the replaced one.

        Parameters
        ----------
        old:
            Observation to be replaced.
        new:
            Replacement observation.
        """
        #We remove the old observation, if present
        self.pat_map.pop(old, None)
        try:
            self.observations.remove(old)
        except ValueError:
            #Maybe the observation is unsorted.
            try:
                idx = next(i for i in xrange(len(self.observations))
                                                if self.observations[i] is old)
                self.observations.pop(idx)
            except StopIteration:
                pass
        #We add or update the new observation, if not present
        try:
            self.observations.index(new)
        except ValueError:
            try:
                next(o for o in self.observations if o is new)
                self.update_location(new)
            except StopIteration:
                if not obsbuf.contains_observation(new):
                    self.observations.add(new)
        try:
            self.focus[self.focus.index(old)] = new
        except ValueError:
            pass
        for i in xrange(len(self.delay_match)):
            if self.delay_match[i][0] is old:
                self.delay_match[i] = (new, self.delay_match[i][1])
            elif self.delay_match[i][1] is old:
                self.delay_match[i] = (self.delay_match[i][0], new)

    def match(self, finding, obs):
        """
        Performs a matching for a finding, checking and propagating the
        constraints to the pattern generating the **finding**. This function
        assumes the reasoning scheme ensures that **finding** is only related
        to one hypothesis at most, and this hypothesis is not still connected
        with any other pattern.
        """
        ev_p = self.get_evidence_patterns(finding)
        assert len(ev_p) <= 1
        if ev_p:
            #The following condition prevents cycles between patterns.
            assert not self.get_evidence_patterns(list(ev_p)[0].hypothesis)
            pat = tuple(ev_p)[0]
            verify(pat.hypothesis is not obs, 'Observation {0} is at the same'
                          'time hypothesis and evidence of a pattern', (obs, ))
            verify(obs not in pat.evidence[pat.get_evidence_type(finding)[0]],
                  'Observation {0} is already in the evidence of {1} pattern',
                                                                    (obs, pat))
            patcp = copy.copy(pat)
            try:
                patcp.match(finding, obs)
                self.pat_map[patcp.hypothesis] = (patcp, frozenset())
                self.replace_pat(pat, patcp)
                self.replace_obs(pat.hypothesis, patcp.hypothesis)
            except AttributeError as err:
                conf = err.args[0]
                obj = (obs if conf is obs or obs.references(conf) else
                            next((o for o in pat.obs_seq
                                    if o is conf or o.references(conf)), None))
                #There has been a modification attempt of the original
                #observation, we copy the observation and retry the matching.
                assert pat in self.patterns
                memo = {}
                if obj is obs:
                    obs = self._save(obs, memo)
                else:
                    #TODO it is possible to have to copy more than one
                    #observation, so the matching has to be retried.
                    self._save(obj, memo)
                hypcp = memo.get(pat.hypothesis) or self._save(pat.hypothesis)
                patcp = self.get_hypothesis_pattern(hypcp)
                patcp.match(finding, obs)
                self.update_location(hypcp)
            self.verify_consecutivity(patcp.hypothesis)
            hyp, ev_set = self.pat_map[obs]
            self.pat_map[obs] = (hyp, ev_set.union({patcp}))
        self.replace_obs(finding, obs)
        self.verify_consecutivity(obs)
        if (obs in self.unintelligible and
                                self.get_abstraction_pattern(obs) is not None):
            self.unintelligible.remove(obs)

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
        cons = [pat.get_consecutive(obs)
                                       for pat in self.pat_map[obs][1]]
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
        didx = self.delayed_idx(obs)
        for obs1 in obs_lst:
            suc = self.get_suc(obs1)
            verify(suc is None or
                   (didx > -1 and self.delay_match[didx][0] is suc) or
                   not between(obs1, obs, suc),
         'Consecutivity constraint violation between {0} and {1}', (obs1, suc))

    def get_pred(self, obs):
        """
        Obtains the predecessor of an observation in this interpretation, or
        None if it does not exist.
        """
        pred = {p for p in (pat.get_consecutive(obs)[0]
                             for pat in self.pat_map[obs][1]) if p is not None}
        verify(len(pred) <= 1, 'More than one predecessor for {0}', (obs, ))
        return None if not pred else pred.pop()

    def get_suc(self, obs):
        """
        Obtains the successor of an observation in this interpretation, or
        None if it does not exist.
        """
        suc = {s for s in (pat.get_consecutive(obs)[1]
                             for pat in self.pat_map[obs][1]) if s is not None}
        verify(len(suc) <= 1, 'More than one successor for {0}', (obs, ))
        return None if not suc else suc.pop()

    def get_abstraction_pattern(self, observation):
        """
        Obtains the *AbstractionPattern* instance that abstracts the given
        observation. If not such pattern exists, returns None
        """
        try:
            return next(p for p in self.pat_map[observation][1]
                                                   if p.abstracts(observation))
        except StopIteration:
            return None

    def get_evidence_patterns(self, observation):
        """
        Obtains a set of *AbstractionPattern* instances for which the given
        observation belongs to their evidence.
        """
        return self.pat_map[observation][1].copy()

    def get_hypothesis_pattern(self, observation):
        """
        Obtains the *AbstractionPattern* instance for which the given
        observation is its hypothesis, or None if the observation is not an
        hypothesis.
        """
        return self.pat_map[observation][0]

    def delayed_idx(self, observation):
        """
        Obtains the index in the delayed_match list of associations of an
        observation, or -1 if the observation is not a delayed match.
        """
        return next((i for i in xrange(len(self.delay_match))
                                 if self.delay_match[i][1] is observation), -1)

    def is_finding(self, observation):
        """
        Determines whether an observation is a finding (a prediction of a
        pattern waiting for a consistent matching) or not.
        """
        return any(observation in p.findings for p in self.patterns)

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
                self.patterns.remove(hypat)
                total += 1
                if ap.get_obs_level(type(obs)) == 0:
                    abstime += obs.earlyend - obs.latestart + 1
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
