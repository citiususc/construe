# -*- coding: utf-8 -*-
# pylint: disable=C0103
"""
Created on Mon Apr 29 09:31:50 2013

This module contains the implementation of the searching routines for the
interpretation process. Specifically, it provides the proposed implementation
for the Construe Algorithm.

@author: T. Teijeiro
"""

from ..utils.predictable_iter import PredictableIter
import construe.acquisition.record_acquisition as IN
import construe.knowledge.abstraction_patterns as ap
import construe.inference.reasoning as reasoning
import numpy as np
import itertools as it
import weakref
from sortedcontainers import SortedList
from operator import attrgetter
from collections import namedtuple

#Tuple containing the search heuristic.
Heuristic = namedtuple('Heuristic', 'ocov, scov, time, nhyp')
#Tuple containing the information of each node (heuristic and interpretation)
Node = namedtuple('Node', 'h, node')

def ilen(iterator):
    """
    Counts the number of elements in an iterator, consuming it and making it
    unavailable for any other purpose.
    """
    return sum(1 for _ in iterator)

def valuation(node, time=None):
    """
    Obtains the heuristic evaluation of an interpretation (a smaller value is
    better). Currently this function checks the proportion of abstracted
    observations over the total number of observations that can be abstracted
    until a specific time point. If the time point is not specified, the time
    point of the interpretation is considered. See the *time_point* function
    for details. The function returns a tuple with three values, the first is
    the relation unexplained/total observations, the second is the amount of
    time being abstracted by at least one observation, and the third is the
    number of hypotheses in the interpretation.
    """
    time = time or node.time_point
    tp, abst, abstime, nhyp = node.past_metrics
    assert time >= tp
    if time > tp:
        abstime += sum(o.earlyend - o.latestart + 1 for o in node.abstracted
                       if ap.get_obs_level(type(o)) == 0)
        abst += len(node.abstracted)
        nhyp += len(node.observations) + node.focus.nhyp
    total = IN.BUF.nobs_before(time) + node.nabd
    if total == 0:
        return (0.0, 0.0, 0.0)
    else:
        return (1.0 - abst/float(total), -abstime, nhyp)

def goal(node):
    """
    Checks if a node is a goal node, it is, an enough good interpretation to
    immediately stop the search process.
    """
    return (IN.BUF.get_status() is IN.BUF.Status.STOPPED
            and valuation(node, np.inf)[0] == 0.0 and node.is_firm)

def _safe_discard(node, checklist, reason=''):
    """
    Discards a node checking if it is not an ancestor of any of the nodes in
    the checklist iterable. In fact, the node is not destroyed, but only
    detached from the main interpretations tree. The *reason* optional
    parameter is the argument passed as argument for the *detach()* method
    of the node.
    """
    if all(not subnode.is_ancestor(node) for _, subnode in checklist):
        node.detach(reason)

class Construe(object):
    """
    This class implements the **Construe** algorithm allowing fine-grained
    control of the steps of the algorithm.
    """
    def __init__(self, root_node, K):
        """
        Initializes a new algorithm execution, receiving as arguments the
        root node for the exploration and the K parameter, which determines the
        exploratory nature of the algorithm.

        Instance Properties
        -------------------
        K:
            Exploration parameter.
        successors:
            Dictionary storing the successor generator of each node.
        last_time:
            Interpretation time of the most advanced interpretation generated
            so far.
        open:
            Sorted list of the open nodes, that can still be expanded.
        closed:
            Sorted list of closed nodes.
        best:
            When a node satisfies the *goal()* function, this attribute is
            assigned to that node. While the finished() method returns False,
            this attribute may be refined with new better interpretations.
        """
        assert K > 0
        self.K = K
        self.successors = weakref.WeakKeyDictionary()
        root_succ = PredictableIter(reasoning.firm_succ(root_node))
        if not root_succ.hasnext():
            raise ValueError('The root node does not have valid successors')
        self.successors[root_node] = root_succ
        self.last_time = root_node.time_point
        ocov, scov, nhyp = valuation(root_node)
        heur = Heuristic(ocov, scov, -self.last_time, nhyp)
        self.open = SortedList([Node(heur, root_node)], key=attrgetter('h'))
        self.closed = SortedList(key=attrgetter('h'))
        self.best = None

    def _add_closed(self, interp, nodelist, tpoint=None):
        """
        Adds a new interpretation to the closed list, recalculating the
        valuation of the already closed nodes using the furthest interpretation
        time point.

        Parameters
        ----------
        interp:
            Interpretation to be added to the closed list.
        nodelist:
            List of nodes we want to keep safe in the discarding operations.
        tpoint:
            Time point of interp. If None, it is obtained from the *time_point*
            property of interp.
        """
        tpoint = tpoint if tpoint is not None else interp.time_point
        self.successors.pop(interp, None)
        ocov, scov, nhyp = valuation(interp, self.last_time)
        node = Node(Heuristic(ocov, scov, -tpoint, nhyp), interp)
        #First we discard the not interesting closed interpretations
        i = self.closed.bisect_right(node)
        while len(self.closed) > i:
            _, discarded = self.closed.pop()
            _safe_discard(discarded, set(it.chain((node,), self.open, nodelist,
                                                  self.closed[:i])),
                          'uninteresting interpretation')
        #And now we recalculate the value of all the still interesting
        newclosed = SortedList(key=attrgetter('h'))
        for (ocov, scov, ntime, nhyp), iclosed in self.closed:
            if -ntime < self.last_time:
                ocov, scov, nhyp = valuation(iclosed, self.last_time)
            newclosed.add(Node(Heuristic(ocov, scov, ntime, nhyp), iclosed))
        #We introduce the closed node in the new closed list, discarding the
        #interpretations that are worse after the valuation update.
        i = newclosed.bisect_right(node)
        while len(newclosed) > i:
            _, discarded = newclosed.pop()
            _safe_discard(discarded, set(it.chain((node,), self.open, nodelist,
                                                  newclosed[:i])),
                          'uninteresting interpretation')
        newclosed.add(node)
        self.closed = newclosed

    def step(self, filt=lambda _: True):
        """
        Performs a new step of the algorithm, by continuing the K-best nodes
        satisfying the *filt* function one step.

        Parameters
        ----------
        filt:
            Boolean function that receives an element of the open list and
            decides if the node can be expanded in this iteration. The first
            K nodes satisfying this filter are expanded.
        """
        newnodes = []
        optimal = False
        def anc(n):
            """
            In an optimal context, ancestor interpretations of the newly
            generated nodes are not expanded. The following filter is for that
            purpose.
            """
            return optimal and any(nn.is_ancestor(n.node) for _, nn in newnodes)

        for _ in xrange(self.K):
            node = next((n for n in self.open if filt(n) and not anc(n)), None)
            #The search stops if no nodes can be expanded or if, being in an
            #optimal context, we need to expand a non-optimal node.
            if node is None or (optimal and node.h.ocov > 0.0):
                break
            self.open.remove(node)
            #Go a step further
            nxt = self.successors[node.node].next()
            self.successors[nxt] = PredictableIter(reasoning.firm_succ(nxt))
            nxtime = nxt.time_point
            if nxtime > self.last_time:
                self.last_time = nxtime
            ocov, scov, nhyp = valuation(nxt, nxtime)
            nxt = Node(Heuristic(ocov, scov, -nxtime, nhyp), nxt)
            #Optimality is determined by the coverage of the successors.
            optimal = optimal or ocov == 0.0
            #Reorganize the open and closed list.
            for n in (node, nxt):
                if self.successors[n.node].hasnext():
                    newnodes.append(n)
                elif n.node.is_firm:
                    self._add_closed(n.node, newnodes, -n.h.time)
                    #Is there a full cover?
                    if (n is nxt and n.h.ocov == 0.0 and goal(n.node) and
                            (self.best is None or n.h < self.best.h)):
                        self.best = n
        for node in newnodes:
            self.open.add(node)
        if not self.open:
            if not self.closed:
                raise ValueError('Could not find a complete interpretation.')
            self.best = min(self.closed)

    def prune(self):
        """
        Perform a pruning operation by limiting the size of the *open* list
        only to the K best.
        """
        #Now we get the best nodes with a common valuation.
        newopened = SortedList(key=attrgetter('h'))
        for h, node in self.open:
            ocov, scov, nhyp = valuation(node, self.last_time)
            newopened.add(Node(Heuristic(ocov, scov, h.time, nhyp), node))
        self.open = newopened
        i = count = 0
        while i < len(self.open) and count < self.K:
            if self.open[i].node.is_firm:
                count += 1
            i += 1
        for _, discarded in self.open[i:]:
            _safe_discard(discarded, it.chain(self.closed, self.open[:i]),
                          'sacrificed interpretation')
        del self.open[i:]
        #We also clear the reasoning cache, since some interpretations cannot
        #be eligible for merging anymore.
        if self.open:
            earliestime = min(n.past_metrics.time for _, n in self.open)
            reasoning.clear_cache(earliestime)

    def finished(self):
        """
        Checks if the searching procedure is finished, that is, more
        iterations, even if possible, will probably not lead to better
        interpretations that the best one. This is considered true if in the
        open list there are no partial covers with less hypotheses than
        the current best interpretation and that are not ancestors of the
        current best interpretation.
        """
        return (self.best is not None and
                all(self.best.node.is_ancestor(n.node) for n in self.open
                    if n.h.ocov == 0.0 and n.h.nhyp < self.best.h.nhyp))
