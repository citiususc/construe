# -*- coding: utf-8 -*-
# pylint: disable-msg= C0103, W0622
"""
Created on Mon Jun 11 18:27:22 2012

This module contains the definition of the reasoning methods using to generate
the successor interpretations of a given one. Several different methods are
applied based on the internal state of the interpretation, but the result is
equivalent to apply always the abduction process, but it allows to avoid the
generation of many uninteresting interpretations.

@author: T. Teijeiro
"""
import construe.knowledge.abstraction_patterns as ap
import construe.knowledge.constants as C
from ..model.observable import is_singleton
from ..model import AbstractionPattern
from ..model.interpretation import Interpretation
from ..model.automata import NULL_PROC
from collections import deque, Counter
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

def _clone_attrs(obs, ref):
    """
    Performs a deep copy of the attributes of **ref**, setting them in **obs**.
    The procedure ensures that all the attributes of **obs** are not frozen.

    Parameters
    ----------
    obs, ref:
        Observations. The attributes of *obs* are set to be equal that the
        attributes of *ref*.
    """
    memo = {}
    frozen = ref.frozen
    if frozen:
        ref.unfreeze()
    for attr, value in vars(ref).iteritems():
        if attr != '__frozen__':
            if id(value) not in memo:
                memo[id(value)] = copy.deepcopy(value)
            setattr(obs, attr, memo[id(value)])
    if frozen:
        ref.freeze()

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

def _finding_matched(interpretation, finding):
    """
    Checks if in some successor of an interpretation a given finding has been
    successfully matched with another observation, by ensuring that the finding
    does not appear in the global evidence list of the successor. It performs
    a breadth-first search on the successors tree.
    """
    queue = deque([interpretation])
    while queue:
        interp = queue.popleft()
        if finding not in interp.pat_map:
            return True
        queue.extend(interp.child)
    return False

def _pattern_completed(interpretation, patidx):
    """
    Checks if in some successor of an interpretation a given pattern
    (identified by its index in the pattern list) has found a sufficient set
    of evidence to support its hypothesis. It performs a depth-first search
    on the successors tree.
    """
    stack = interpretation.child[:]
    while stack:
        interp = stack.pop()
        if interp.patterns[patidx].sufficient_evidence:
            return True
        stack.extend(interp.child)
    return False

def _find_mergeable(interpretation):
    """
    Searches the interpretation cache in order to find mergeable
    interpretations.
    """
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
    only returns those that are firm. To do this, it performs a depth-first
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
                #HINT this saves memory, but penalizes _finding_matched
                if node not in _FSUCC and not node.is_firm:
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
    focus = interpretation.focus[-1]
    if interpretation.is_finding(focus):
        sequences.append(subsume(interpretation))
        #Past environment findings are not predicted.
        pat = next(p for p in interpretation.patterns if focus in p.findings)
        if focus.lateend > interpretation.time_point or pat.abstracts(focus):
            sequences.append(predict(interpretation))
    else:
        if interpretation.get_hypothesis_pattern(focus) is not None:
            sequences.append(deduce(interpretation))
        sequences.append(abduce(interpretation))
        sequences.append(advance(interpretation))
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

def subsume(interpretation):
    """
    Obtains the interpretations that can be derived from a given one, by
    establishing all the consistent matchings with the unmatched finding of
    the interpretation through a subsumption operation.
    """
    focus = interpretation.focus[-1]
    assert interpretation.is_finding(focus)
    is_abstr = interpretation.get_abstraction_pattern(focus) is not None
    #First we test all the subsumption options
    opt = interpretation.get_observations(clazz=type(focus),
            filt=lambda ev: (ev.start.value.overlapm(focus.start.value) and
                               ev.time.value.overlapm(focus.time.value) and
                               ev.end.value.overlapm(focus.end.value) and
                            (ev not in interpretation.unintelligible and
                             interpretation.get_abstraction_pattern(ev) is None
                                                         if is_abstr else True)
                            and _consecutive_valid(focus, ev, interpretation)))
    for subs in opt:
        newint = Interpretation(interpretation)
        try:
            newint.match(focus, subs)
            newint.focus.pop()
            newint.remove_old()
            STATS.update(['S+' + str(focus.__class__.__name__)])
            yield newint
        except InconsistencyError as error:
            newint.discard(str(error))

def predict(interpretation):
    """
    Obtains the interpretations that can be derived from a given one, by
    performing a deduction operation taking as hypothesis the unmatched finding
    of the interpretation.
    """
    focus = interpretation.focus[-1]
    assert interpretation.is_finding(focus)
    for pat in (p for p in ap.KNOWLEDGE if
                                  issubclass(p.Hypothesis, type(focus)) and
                                  not _singleton_violation(p, interpretation)):
        #The exploration stops at the first consistent matching of the finding
        if _finding_matched(interpretation, focus):
            return
        newint = Interpretation(interpretation)
        if is_singleton(pat.Hypothesis):
            newint.singletons.add(pat.Hypothesis)
        pattern = AbstractionPattern(pat)
        #We set the known attributes of the new conjecture
        _clone_attrs(pattern.hypothesis, focus)
        try:
            newint.conjecture(pattern.hypothesis, pattern)
            #The matching of the finding is delayed until the new hypothesis
            #has sufficient evidence supporting it.
            newint.delay_match.append((focus, pattern.hypothesis))
            #We focus on the new predicted observation, replacing the predicted
            #finding by it.
            newint.focus.pop()
            newint.focus.append(pattern.hypothesis)
            STATS.update(['D+' + str(pat)])
            yield newint
        except InconsistencyError as error:
            newint.discard(str(error))

def deduce(interpretation):
    """
    Extends the pattern whose hypothesis is the current inference focus of the
    interpretation, by testing all possible successor paths.
    """
    focus = interpretation.focus[-1]
    assert not interpretation.is_finding(focus)
    hypat = interpretation.get_hypothesis_pattern(focus)
    pidx = interpretation.patterns.index(hypat)
    assert not hypat.findings
    for suc in hypat.successors():
        #Once a sufficient set of evidence has ben obtained, we don't look for
        #other alternatives. Therefore, the successors() generator should try
        #to optimize the coverage of the generated patterns.
        if _pattern_completed(interpretation, pidx):
            return
        newint = Interpretation(interpretation)
        newint.observations.add(suc.hypothesis)
        newint.pat_map[suc.hypothesis] = (suc, frozenset())
        newint.replace_pat(hypat, suc)
        try:
            #We set the focus on the new predicted finding, if it exists.
            if suc.findings:
                newint.replace_obs(focus, suc.hypothesis)
                finding = set(suc.findings).pop()
                newint.pat_map[finding] = (None, frozenset({suc}))
                newint.focus.append(finding)
                newint.verify_consecutivity(finding)
            else:
                newint.match(focus, suc.hypothesis)
            STATS.update(['X+' + str(hypat.automata)])
            yield newint
        except InconsistencyError as error:
            newint.discard(str(error))

def abduce(interpretation):
    """
    Continues the inference by performing an abduction operation on the current
    inference focus of the interpretation.
    """
    focus = interpretation.focus[-1]
    assert not interpretation.is_finding(focus)
    #If the focus is an hypothesis of a pattern, we require that pattern to
    #have sufficient supporting evidence.
    hypat = interpretation.get_hypothesis_pattern(focus)
    if ((hypat is not None and not hypat.sufficient_evidence) or
        interpretation.get_abstraction_pattern(focus) or
                                       interpretation.delayed_idx(focus) > -1):
        return
    hypatcp = None
    if hypat is not None and hypat.automata.obs_proc is not NULL_PROC:
        hypatcp = copy.copy(hypat)
        try:
            hypatcp.finish()
        except InconsistencyError as error:
            return
    qobs = type(focus)
    for pat in (p for p in ap.KNOWLEDGE if
                                  issubclass(qobs, tuple(p.abstracted)) and
                                  not _singleton_violation(p, interpretation)):
        types = [tp for tp in pat.abstracted if issubclass(qobs, tp)]
        for typ in types:
            for trans in pat.abstractions[typ]:
                try:
                    newint = Interpretation(interpretation)
                    if hypatcp is not None:
                        newint.observations.add(hypatcp.hypothesis)
                        newint.pat_map[hypatcp.hypothesis] = (hypatcp,
                                                                   frozenset())
                        newint.replace_pat(hypat, hypatcp)
                        newint.match(focus, hypatcp.hypothesis)
                        focus = hypatcp.hypothesis
                    if is_singleton(pat.Hypothesis):
                        newint.singletons.add(pat.Hypothesis)
                    pattern = AbstractionPattern(pat)
                    pattern.istate = trans.istate
                    pattern.fstate = trans.fstate
                    pattern.trseq.append((trans, focus))
                    pattern.evidence[typ].append(focus)
                    hyp, ev_pats = newint.pat_map[focus]
                    newint.pat_map[focus] = (hyp, ev_pats.union({pattern}))
                    newint.conjecture(pattern.hypothesis, pattern)
                    trans.tconst(pattern, focus)
                    pattern.check_temporal_consistency()
                    trans.gconst(pattern, focus)
                    newint.update_location(pattern.hypothesis)
                    newint.focus.pop()
                    newint.focus.append(pattern.hypothesis)
                    newint.verify_consecutivity(pattern.hypothesis)
                    STATS.update(['A+' + str(pat)])
                    yield newint
                except InconsistencyError as error:
                    newint.discard(str(error))

def advance(interpretation):
    """
    Continues the inference by recovering the previous inference focus of the
    interpretation, or going to the next unexplained observation. It also
    resolves all the postponed matchings.
    """
    max_ap_level = ap.get_max_level()
    focus = interpretation.focus[-1]
    assert not interpretation.is_finding(focus)
    #We can not advance if the focus is an hypothesis of a pattern with
    #insufficient evidence.
    hypat = interpretation.get_hypothesis_pattern(focus)
    newint = None
    if hypat is not None:
        if not hypat.sufficient_evidence:
            return
        elif hypat.automata.obs_proc is not NULL_PROC:
            hypatcp = copy.copy(hypat)
            try:
                hypatcp.finish()
                newint = Interpretation(interpretation)
                newint.observations.add(hypatcp.hypothesis)
                newint.pat_map[hypatcp.hypothesis] = (hypatcp, frozenset())
                newint.replace_pat(hypat, hypatcp)
                newint.match(focus, hypatcp.hypothesis)
                focus = hypatcp.hypothesis
            except InconsistencyError:
                return
    #If the new focus is a delayed matching, we solve it at this point.
    if interpretation.delayed_idx(interpretation.focus[-1]) > -1:
        newint = newint or Interpretation(interpretation)
        try:
            idx = newint.delayed_idx(focus)
            finding, obs = newint.delay_match.pop(idx)
            newint.match(finding, obs)
            newint.focus.pop()
        except InconsistencyError as error:
            newint.discard(str(error))
            return
    else:
        #HINT with the current knowledge base, we restrict the advancement to
        #observations of the first or the last level, not allowing partial
        #interpretations in the abstraction level.
        if (len(interpretation.focus) == 1 and
                       0 < ap.get_obs_level(type(focus)) < max_ap_level):
            return
        #We just move on the focus.
        newint = newint or Interpretation(interpretation)
        if ap.get_obs_level(type(focus)) == 0:
            newint.unintelligible.add(focus)
        newint.focus.pop()
    #If we have reached the top of the stack, we go ahead to the next
    #unexplained observation.
    if not newint.focus:
        try:
            unexp = newint.get_observations(start=focus.earlystart + 1, filt=
                 lambda ev: ev not in newint.unintelligible and
                             newint.get_abstraction_pattern(ev) is None).next()
            newint.focus.append(unexp)
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
