# -*- coding: utf-8 -*-
# pylint: disable-msg= C0103, W0622
"""
Created on Mon Jun 11 18:27:22 2012

This module contains the definition of the reasoning methods using to generate
the successor interpretations of a given one. Several different methods are
applied based on the internal state of the interpretation, but the result is
equivalent to apply always the abduction process while avoiding the generation
of many uninteresting interpretations.

@author: T. Teijeiro
"""
import construe.knowledge.abstraction_patterns as ap
import construe.knowledge.constants as C
from ..model.observable import is_singleton
from ..model.FreezableObject import clone_attrs
from ..model import AbstractionPattern
from ..model.interpretation import Interpretation
from ..model.automata import NULL_PROC
from collections import Counter
import copy
import itertools as it
import numpy as np
import weakref
import blist
from ..model.constraint_network import InconsistencyError
#FIXME debug import
import construe.knowledge.observables as OBS

#numpy.any behavior with generators is not as expected, so we require to modify
#it
if any is np.any:
    import __builtin__
    any = __builtin__.any
assert any is not np.any

#########################
### Module attributes ###
#########################

#Flag to avoid dead-end branch pruning, to allow debugging of all hypotheses
SAVE_TREE = False

#Counter to measure where the interpretation process concentrates its efforts.
STATS = Counter()

#Sorted list of interpretations to find merge situations.
_INCACHE = blist.sortedset(key=lambda interp: len(interp.observations))

#Dictionary to map merged interpretations
_MERGED = weakref.WeakKeyDictionary()

#Dictionary to obtain the successor generator of each interpretation.
_SUCCESSORS = weakref.WeakKeyDictionary()

#Set containing the interpretations with one or more firm successors. If an
#interpretation is exhausted and does not belong to this set, it is discarded
#as a dead-end reasoning path.
_FSUCC = weakref.WeakSet()

#Set containing the findings that have been successfully matched after a
#predictive reasoning
_MATCHED_FINDINGS = weakref.WeakSet()

###############################
### Global module functions ###
###############################

def reset():
    """
    Deletes the content of all the global data structures of the module,
    allowing a fresh restart of every reasoning operations.
    """
    global _SUCCESSORS
    _INCACHE.clear()
    _MERGED.clear()
    #We cannot clear this dict, it raises many KeyError exceptions.
    _SUCCESSORS = weakref.WeakKeyDictionary()
    _FSUCC.clear()
    _MATCHED_FINDINGS.clear()

def clear_cache(time):
    """
    Deletes old information from interpretation cache. All interpretations
    with time_point < time are removed, and therefore cannot be eligible for
    merging.
    """
    global _INCACHE
    _INCACHE = blist.sortedset((i for i in _INCACHE if i.time_point >= time),
                               key=lambda interp: len(interp.observations))


#########################
### Utility functions ###
#########################

def _save_fsucc(interpretation):
    """
    This function is called when an interpretation is found to have a firm
    successor, and then all the hierarchy up is properly marked.
    """
    node = interpretation
    while node is not None:
        #When an interpretation is already found in the set, we can stop.
        if node in _FSUCC:
            break
        _FSUCC.add(node)
        node = node.parent

def _consecutive_valid(obs, rep, interpretation):
    """Checks if the consecutive restrictions of *obs* are satisfied by
    *rep*."""
    obs_p, obs_s = interpretation.get_pred(obs), interpretation.get_suc(obs)
    #We exclude the original observation from the checking operation
    if obs_p is not None:
        if interpretation.non_consecutive(obs_p, rep):
            return False
    if obs_s is not None:
        if interpretation.non_consecutive(rep, obs_s):
            return False
    return True

def _singleton_violation(pat, interpretation):
    """
    Checks if an abstraction pattern violates the singleton constraints in a
    given interpretation, that is, if the interpretation already contains an
    observation of the same type of the hypothesis of the pattern, and that
    hypothesis is a singleton.

    Parameters
    ----------
    pat:
        PatternAutomata.
    interpretation:
        Interpretation.
    """
    return issubclass(pat.Hypothesis, tuple(interpretation.singletons))

def _find_mergeable(interpretation):
    """
    Searches the interpretation cache in order to find mergeable
    interpretations.
    """
    #TODO temporal disable this feature
    return None
    idx = _INCACHE.bisect_left(interpretation)
    nobs = len(interpretation.observations)
    mergeable = interpretation.is_mergeable
    while idx < len(_INCACHE):
        other = _INCACHE[idx]
        if len(other.observations) > nobs:
            break
        if mergeable(other):
            return other
        idx += 1
    return None

###########################
### Reasoning functions ###
###########################

def firm_succ(interpretation):
    """
    Returns a generator of the successor interpretations of a given one, but it
    only yields those that are firm. To do this, it performs a depth-first
    search process.

    **Note:** This generator will not obtain **every** node in the tree, but
    only those that are firm. Thus, a walk using only the output of this
    generator will not be exhaustive. Completeness is only guaranteed by
    :func:`multicall_succ`.
    """
    #Check merge possibilities
    if interpretation not in _SUCCESSORS:
        mrg = _find_mergeable(interpretation)
        if mrg is not None:
            _SUCCESSORS[interpretation] = _merge_succ(interpretation, mrg)
            _MERGED[interpretation] = mrg
    stack = [(interpretation, multicall_succ(interpretation))]
    while stack:
        node, generator = stack[-1]
        finished = False
        while not finished:
            try:
                suc = generator.next()
                if suc.is_firm:
                    #Only original interpretations are cached.
                    if interpretation not in _MERGED:
                        _INCACHE.add(suc)
                    _save_fsucc(node)
                    yield suc
                else:
                    stack.append((suc, multicall_succ(suc)))
                    finished = True
            except StopIteration:
                finished = True
                #Remove interpretations with no firm descendants
                if not SAVE_TREE and node not in _FSUCC and not node.is_firm:
                    node.discard('Dead-end interpretation')
                stack.pop()

def multicall_succ(interpretation):
    """
    Returns a generator of the successor interpretations of a given one. The
    sequence is the same that using the **_succ** generator, but with the
    difference that it allows the creation and iteration of many generators
    ensuring each successor interpretation is created only once.
    """
    #Generator lookup or creation.
    successors = _SUCCESSORS.get(interpretation)
    if successors is None:
        merged = _MERGED.get(interpretation)
        successors = (_succ(interpretation) if merged is None
                                      else _merge_succ(interpretation, merged))
        _SUCCESSORS[interpretation] = successors
    #Main loop
    yielded = weakref.WeakSet()
    while True:
        nxt = next((n for n in interpretation.child if n not in yielded), None)
        nxt = nxt or successors.next()
        yielded.add(nxt)
        yield nxt

def _succ(interpretation):
    """
    Returns a generator of the successor interpretations of a given one, by the
    appropriate combination of the different implemented reasoning modes.
    **Warning**: This function can be called only one time for each
    interpretation. If you need multiple iterations over the successor nodes
    of an interpretation, use the *multicall_succ* generator.
    """
    sequences = []
    focus, pat = interpretation.focus.top
    if pat is not None and focus is pat.finding:
        sequences.append(subsume(interpretation, focus, pat))
        #Past environment findings are not predicted.
        if focus.lateend > interpretation.time_point or pat.abstracts(focus):
            sequences.append(predict(interpretation, focus, pat))
    else:
        if pat is not None:
            sequences.append(deduce(interpretation, focus, pat))
        sequences.append(abduce(interpretation, focus, pat))
        sequences.append(advance(interpretation, focus, pat))
    return it.chain.from_iterable(sequences)

def _merge_succ(interpretation, merged):
    """
    Returns a generator of the successor interpretations of a given one,
    assuming we have performed a merge operation with an existing
    interpretation.
    """
    diff = interpretation.past_metrics.diff(merged.past_metrics)
    for succ in multicall_succ(merged):
        #Successors are shallow copies of the merged successors, changing
        #the tree structure references and the past_metrics.
        nxt = copy.copy(succ)
        nxt.name = str(Interpretation.counter)
        Interpretation.counter += 1
        nxt.parent = interpretation
        nxt.child = []
        nxt.past_metrics = nxt.past_metrics.patch(diff)
        #We already flag the successors as merged.
        _MERGED[nxt] = succ
        STATS.update(['Merge'])
        yield nxt

def subsume(interp, finding, pattern):
    """
    Obtains the interpretations that can be derived from a given one, by
    establishing all the consistent matchings with the unmatched finding of
    the interpretation through a subsumption operation.
    """
    is_abstr = pattern.abstracts(finding)
    #First we test all the subsumption options
    opt = interp.get_observations(clazz=type(finding),
                                  start=finding.earlystart, end=finding.lateend,
            filt=lambda ev: (ev.earlystart in finding.start.value
                             and  ev.time.start in finding.time.value
                             and  ev.lateend in finding.end.value
                             and (ev not in interp.unintelligible and ev not in
                                  interp.abstracted if is_abstr else True)
                             and _consecutive_valid(finding, ev, interp)))
    for subs in opt:
        newint = Interpretation(interp)
        try:
            newint.focus.match(finding, subs)
            _MATCHED_FINDINGS.add(finding)
            #Consecutivity constraints are checked for the updated hypothesis,
            #now at the top of the stack, and for the matched observation.
            newint.verify_consecutivity(newint.focus.top[0])
            newint.verify_consecutivity(subs)
            newint.remove_old()
            if is_abstr:
                #Lazy copy of the parent reference
                newint.abstracted = newint.abstracted.copy()
                newint.abstracted.add(subs)
            STATS.update(['S+' + str(finding.__class__.__name__)])
            yield newint
        except InconsistencyError as error:
            newint.discard(str(error))

def predict(interp, finding, pattern):
    """
    Obtains the interpretations that can be derived from a given one, by
    performing a deduction operation taking as hypothesis the unmatched finding
    of the interpretation.
    """
    for pat in (p for p in ap.KNOWLEDGE if
                    issubclass(p.Hypothesis, type(finding))
                                      and not _singleton_violation(p, interp)):
        #The exploration stops at the first consistent matching of the finding
        if finding in _MATCHED_FINDINGS:
            _MATCHED_FINDINGS.remove(finding)
            return
        newint = Interpretation(interp)
        if is_singleton(pat.Hypothesis):
            newint.singletons = newint.singletons.copy()
            newint.singletons.add(pat.Hypothesis)
        pattern = AbstractionPattern(pat)
        #We set the known attributes of the new conjecture
        clone_attrs(pattern.hypothesis, finding)
        try:
            #The new predicted observation is focused.
            newint.focus.push(pattern.hypothesis, pattern)
            #And the number of abducible observations is increased.
            if ap.is_abducible(pat.Hypothesis):
                newint.nabd += 1
            STATS.update(['D+' + str(pat)])
            yield newint
        except InconsistencyError as error:
            newint.discard(str(error))

def deduce(interp, focus, pattern):
    """
    Extends the pattern whose hypothesis is the current inference focus of the
    interpretation, by testing all possible successor paths.
    """
    assert pattern.finding is None
    for suc in pattern.successors():
        #Once a sufficient set of evidence has ben obtained for this pattern
        #and all dependent patterns up in the abstraction hierarchy, we don't
        #look for other alternatives. Therefore, the successors() generator
        #should try to optimize the coverage of the generated patterns.
        if interp in _FSUCC:
            return
        newint = Interpretation(interp)
        try:
            newint.focus.top = (suc.hypothesis, suc)
            #We set the focus on the new predicted finding, if it exists.
            if suc.finding is not None:
                newint.focus.push(suc.finding, suc)
                newint.verify_consecutivity(suc.finding)
            newint.verify_consecutivity(suc.hypothesis)
            STATS.update(['X+' + str(pattern.automata)])
            yield newint
        except InconsistencyError as error:
            newint.discard(str(error))

def abduce(interp, focus, pattern):
    """
    Continues the inference by performing an abduction operation on the current
    inference focus of the interpretation.
    """
    #If the focus is an hypothesis of a pattern, we require that pattern to
    #have sufficient supporting evidence.
    if ((pattern is not None and not pattern.sufficient_evidence)
        or focus in interp.abstracted
        or interp.focus.get_delayed_finding(focus) is not None):
        return
    hypatcp = None
    if pattern is not None and pattern.automata.obs_proc is not NULL_PROC:
        hypatcp = copy.copy(pattern)
        try:
            hypatcp.finish()
        except InconsistencyError as error:
            return
    qobs = type(focus)
    for pat in (p for p in ap.KNOWLEDGE if
                                  issubclass(qobs, tuple(p.abstracted))
                                  and not _singleton_violation(p, interp)):
        types = [tp for tp in pat.abstracted if issubclass(qobs, tp)]
        for typ in types:
            for trans in pat.abstractions[typ]:
                try:
                    newint = Interpretation(interp)
                    if hypatcp is not None:
                        focus = hypatcp.hypothesis
                        newint.observations = newint.observations[:]
                        newint.observations.add(focus)
                    #Pattern consistency checking
                    pattern = AbstractionPattern(pat)
                    pattern.istate = trans.istate
                    pattern.fstate = trans.fstate
                    pattern.trseq.append((trans, focus))
                    pattern.evidence[typ].append(focus)
                    trans.tconst(pattern, focus)
                    pattern.check_temporal_consistency()
                    trans.gconst(pattern, focus)
                    #Interpretation updating
                    newint.abstracted = newint.abstracted.copy()
                    newint.abstracted.add(focus)
                    if is_singleton(pat.Hypothesis):
                        newint.singletons = newint.singletons.copy()
                        newint.singletons.add(pat.Hypothesis)
                    if ap.is_abducible(pat.Hypothesis):
                        newint.nabd += 1
                    newint.focus.pop()
                    newint.focus.push(pattern.hypothesis, pattern)
                    newint.verify_exclusion(pattern.hypothesis)
                    newint.verify_consecutivity(pattern.hypothesis)
                    STATS.update(['A+' + str(pat)])
                    yield newint
                except InconsistencyError as error:
                    newint.discard(str(error))

def advance(interp, focus, pattern):
    """
    Continues the inference by recovering the previous focus of attention, or
    by going to the next unexplained observation. It also resolves all the
    postponed matchings.
    """
    max_ap_level = ap.get_max_level()
    #We can not advance if the focus is an hypothesis of a pattern with
    #insufficient evidence.
    newint = None
    if pattern is not None:
        if not pattern.sufficient_evidence:
            return
        #If there is an observation procedure for the focused hypothesis, the
        #execution takes place now.
        elif pattern.automata.obs_proc is not NULL_PROC:
            patcp = copy.copy(pattern)
            try:
                patcp.finish()
                newint = Interpretation(interp)
                newint.observations = newint.observations[:]
                newint.observations.add(patcp.hypothesis)
                focus = patcp.hypothesis
            except InconsistencyError:
                return
    #If the new focus is a delayed matching, we solve it at this point.
    finding = interp.focus.get_delayed_finding(interp.focus.top[0])
    if finding is not None:
        newint = newint or Interpretation(interp)
        try:
            newint.focus.pop()
            if newint.focus.top[1].abstracts(finding):
                newint.abstracted = newint.abstracted.copy()
                newint.abstracted.add(focus)
            newint.focus.match(finding, focus)
            _MATCHED_FINDINGS.add(finding)
        except InconsistencyError as error:
            newint.discard(str(error))
            return
    else:
        #HINT with the current knowledge base, we restrict the advancement to
        #observations of the first or the last level, not allowing partial
        #interpretations in the abstraction level.
        if (len(interp.focus) == 1 and
                             0 < ap.get_obs_level(type(focus)) < max_ap_level):
            return
        #We just move on the focus.
        newint = newint or Interpretation(interp)
        if ap.get_obs_level(type(focus)) == 0:
            newint.unintelligible = newint.unintelligible.copy()
            newint.unintelligible.add(focus)
        newint.focus.pop()
    #If we have reached the top of the stack, we go ahead to the next
    #unexplained observation.
    if not newint.focus:
        try:
            unexp = newint.get_observations(start=focus.earlystart + 1, filt=
                 lambda ev: ev not in newint.unintelligible
                        and ev not in newint.abstracted
                        and ap.is_abducible(type(ev))).next()
            newint.focus.push(unexp, None)
        except StopIteration:
            newint.discard('No unexplained evidence after the current point')
            return
    #Finally, we remove old observations from the interpretation, since they
    #won't be used in following reasoning steps, and they affect the
    #performance of the searching procedure.
    oldtime = max(newint.past_metrics.time,
                  min(o.earlystart for o in newint.focus) - C.FORGET_TIMESPAN)
    #We always have to keep at least the last observation of the highest
    #abstraction level.
    max_ap_time = [o.earlystart for o in newint.observations
                                  if ap.get_obs_level(type(o)) == max_ap_level]
    if max_ap_time and max_ap_time[-1] < oldtime:
        oldtime = max_ap_time[-1]
    newint.remove_old(oldtime)
    yield newint

if __name__ == "__main__":
    pass
