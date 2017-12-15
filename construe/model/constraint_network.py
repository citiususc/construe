# -*- coding: utf-8 -*-
# pylint: disable-msg=E1101, E0102, E0202, C0103
"""
Created on Mon May 28 16:40:37 2012

This module provides some utility classes to create and manipulate dynamic
constraint networks.

@author: T. Teijeiro
"""

from .interval import Interval
from .FreezableObject import FreezableObject
import numpy as np


def verify(expression, messagestr = None, messageargs = None):
    """
    This function provides an equivalent functionality to the *assert* builtin,
    but it cannot be disabled at compile time. It is very useful to check if a
    proposition is true, raising an *InconsistencyError* otherwise.

    Parameters
    ----------
    expression:
        Boolean expression. If it is True, the function does nothing, else it
        raises an InconsistencyError.
    messagestr:
        Optional message string to explain the generated exception, if it is
        the case.
    messageargs:
        List or tuple of arguments that are passed to the *format* method of
        messagestr. The string is only built if the exception is generated.
    """
    messageargs = messageargs or []
    if not expression:
        raise (InconsistencyError() if messagestr is None
                      else InconsistencyError(messagestr.format(*messageargs)))


class Variable(FreezableObject):
    """
    This class represents a variable in a constraint network, which can take a
    value constrained by an interval that can be modified according constraints
    in a network. As value, this class assumes always an Interval object.
    """
    __slots__ = ('value', )

    def __init__(self, value = Interval(-np.inf, np.inf)):
        """
        Creates a new variable, with a value constrained by an interval. By
        default, this interval is (-Inf, Inf).
        """
        super(Variable, self).__init__()
        self.value = value

    def __lt__(self, other):
        """
        Checks if this variable is less than another one. It is based in the
        relation between *Interval* objects.
        """
        return bool(self.value < other.value)

    def __le__(self, other):
        return self < other or self == other

    def __eq__(self, other):
        return self.value == other.value

    def __gt__(self, other):
        return self.value > other.value

    def __ge__(self, other):
        return self > other or self == other

    def __deepcopy__(self, memo):
        """Performs a deep copy operation of this variable."""
        if id(self) in memo:
            return memo[id(self)]
        else:
            res = Variable(value = Interval(self.value.start, self.value.end))
            memo[id(self)] = res
            return res

    def __str__(self):
        return repr(self.value)

    @property
    def start(self):
        """Gets the start value of this variable"""
        return self.value.start

    @property
    def end(self):
        """Gets the end value of this variable"""
        return self.value.end


class Constraint(object):
    """
    This class represents a constraint between two variables in a constraint
    network. Constraints are represented as intervals.
    """
    __slots__ = ('name', 'va', 'vb', 'constraint')

    def __init__(self, name = '', va = Variable(), vb = Variable(),
                 constr = Interval(-np.inf, np.inf)):
        """
        Creates a new constraint between two variable instances va and vb, with
        a default interval (-Inf, Inf)
        """
        assert va is not vb
        self.name = name
        self.va = va
        self.vb = vb
        self.constraint = constr

    def __str__(self):
        return str(self.va) + '<-' + repr(self.constraint) + '->' + str(self.vb)


    def __deepcopy__(self, memo):
        result = Constraint(name = self.name)
        #va copy
        k = id(self.va)
        vacp = memo.get(k, None)
        if vacp is not None:
            result.va = vacp
        else:
            result.va = Variable(value = Interval(self.va.start, self.va.end))
            memo[k] = result.va
        #vb copy
        k = id(self.vb)
        vbcp = memo.get(k, None)
        if vbcp is not None:
            result.vb = vbcp
        else:
            result.vb = Variable(value = Interval(self.vb.start, self.vb.end))
            memo[k] = result.vb
        #Constraint copy
        result.constraint = Interval(self.constraint.start, self.constraint.end)
        return result



class ConstraintNetwork(object):
    """
    This class represents a constraint network, which contains a set of
    variables linked by a set of constraints. It provides functionality to
    check the consistency of the network, and to obtain the interval of
    possible values of each variable.

    Instance Properties
    -------------------
    unconstrained:
        Boolean variable to flag if the constraint network is unconstrained,
        that is, the involved temporal variables are not set to their minimum
        values, and consistency of the network is not ensured.
        **Note**: A False value of this variable DOES NOT IMPLY that the
        network is minimized. Specifically, variable value changes are not
        detected at network level. A True value of this property, however, does
        imply that the network is not constrained.
    """
    __slots__ = ('_constr', 'unconstrained')

    def __init__(self):
        """
        Initializes a new, empty network, with no constraints.
        """
        #Internally, the network is stored in a dictionary, keyed by
        #temporal variables, associating a set of Constraints in which
        #the variable is involved (starting or finishing)
        self._constr = {}
        self.unconstrained = False


    def __copy__(self):
        """
        Performs a shallow copy of this network, maintaining the references
        to the temporal variables, but copying the constraint objects.
        """
        result = ConstraintNetwork()
        if self._constr:
            for const in set.union(*self._constr.values()):
                result.add_constraint(const.va, const.vb, const.constraint)
        result.unconstrained = self.unconstrained
        return result


    def add_constraint(self, va, vb, const):
        """
        Adds a new constraint to this network, from its components. If there is
        also a constraint between the same variables, the existing one is
        modified to the most restrictive values. va and vb may be the same
        variable.

        Parameters
        ----------
        va:
            First Variable of the constraint
        vb:
            Second Variable of the constraint
        const:
            Interval representing the constraints between the two variables, so
            constraint.start <= vb-va <= constraint.end

        See also: Constraint, Interval
        """
        if va is vb:
            if not const.zero_in:
                raise InconsistencyError('Constraints between the same '
                                                  'variable must contain zero')
        else:
            if not self._constr.has_key(va):
                self._constr[va] = set()
            if not self._constr.has_key(vb):
                self._constr[vb] = set()
            #We look up for a possible duplicate
            try:
                candidate = self.get_constraint(va, vb)
                #If the constraint is in the same direction
                if candidate.va is va:
                    start = max(candidate.constraint.start, const.start)
                    end = min(candidate.constraint.end, const.end)
                #If the direction is the opposite, we change the sign
                else:
                    start = max(candidate.constraint.start, -const.end)
                    end = min(candidate.constraint.end, -const.start)
                newint = Interval(start, end)
                if candidate.constraint != newint:
                    candidate.constraint = newint
                    self.unconstrained = True
            #If no duplicates, we insert the new value
            except KeyError:
                constraint = Constraint(va = va, vb = vb, constr = const)
                self._constr[va].add(constraint)
                self._constr[vb].add(constraint)
                self.unconstrained = True


    def update_constraint(self, va, vb, const):
        """
        Updates the temporal constraint between *va* and *vb* to the *const*
        interval. Raises KeyError exception if such constraint is not found.

        Parameters
        ----------
        va:
            Start variable of the constraint.
        vb:
            End variable of the constraint
        const:
            Interval representing the constraint value
        """
        if va is vb:
            if not const.zero_in:
                raise InconsistencyError('Constraints between the same '
                                                  'variable must contain zero')
        else:
            constraint = (self._constr[va] & self._constr[vb]).pop()
            #If the constraint is in the same direction
            constraint.constraint = (const if constraint.va is va
                                       else Interval(-const.end, -const.start))
            self.unconstrained = True

    def get_constraint(self, va, vb):
        """
        Obtains the *Constraint* object involving two specific variables in
        this network. The endpoint variables are guaranteed to be *va* and *vb*
        but the order is not necessary kept, so the *va* parameter may be the
        *vb* variable in the resulting constraint. If no such constraint exists
        a KeyError exception is raised.
        """
        return (self._constr[va] & self._constr[vb]).pop()


    def remove_constraint(self, va, vb):
        """
        Removes the constraint in this network involving two specific variables.
        If there is no such constraint, an exception is raised.

        Parameters
        ----------
        va:
            One variable of the constraint to remove.
        vb:
            Other variable of the constraint to remove.
        """
        if va is not vb:
            #We get the constraint to remove
            constraint = self.get_constraint(va, vb)
            #And we remove it from the two involved sets
            self._constr[va].remove(constraint)
            self._constr[vb].remove(constraint)
            #If the variables have no constraints, we remove them from the
            #network.
            if not self._constr[va]:
                self._constr.pop(va)
            if not self._constr[vb]:
                self._constr.pop(vb)

    def set_equal(self, va, vb):
        """
        Adds a constraint to indicate that variables *va* and *vb* must be
        equal. Equivalent to:
            add_constraint(va,vb,Interval(0,0))
        """
        self.add_constraint(va, vb, Interval(0, 0))

    def set_before(self, va, vb):
        """
        Adds a constraint to indicate that variable *va* must be before or
        equals *vb*. Equivalent to:
            add_constraint(va,vb,Interval(0,inf))
        """
        self.add_constraint(va, vb, Interval(0, np.inf))

    def set_between(self, va, vb, vc):
        """
        Adds a constraint to indicate that *vb* must be between *va* and *vc*
        """
        self.set_before(va, vb)
        self.set_before(vb, vc)

    def get_variables(self):
        """
        Returns a tuple with all the temporal variables involved in this STP.
        """
        return self._constr.keys()

    def contains_variable(self, var):
        """Checks if a temporal variable is in this STP"""
        return var in self._constr

    def get_constraints(self):
        """
        Returns a list with all the constraint in this network.
        """
        return list(set.union(*self._constr.values()))

    def substitute_variable(self, old, new):
        """
        Substitutes all the ocurrences of the 'old' temporal variable by
        the 'new' temporal variable. It allows for a transparent variable
        modification from outside the network.

        Parameters
        ----------
        old - Variable to be substituted.
        new - Variable to set instead of 'old'.
        """
        constr = self._constr.pop(old)
        for const in constr:
            if const.va is old:
                const.va = new
            elif const.vb is old:
                const.vb = new
        self._constr[new] = constr
        self.unconstrained = old.value != new.value

    def connect(self, other):
        """
        This method joins another constraint network with this, by performing
        the union of the constraints of this network with the constraints of
        the other one.

        Parameters
        ----------
        other:
            ConstraintNetwork to be joined with this.
        """
        for var in other._constr:
            if not self._constr.has_key(var):
                self._constr[var] = set()
            self._constr[var] = self._constr[var].union(other._constr[var])
        self.unconstrained = True


    def minimize_network(self):
        """
        Minimizes the constraint network, detecting possible inconsistencies
        in the process. Returns the set of variables that have been modified
        in the minimization process.
        """
        modified = set()
        var = self.get_variables()
        #Number of variables (1 additional for the absolute 0)
        n = len(var) + 1
        #Adjacency matrix
        A = np.empty((n, n))
        B = np.empty_like(A)
        A.fill(np.inf)
        for i in xrange(n):
            A[i, i] = 0
        #We asign a integer key to each variable, starting in 1
        keys = {}
        i = 1
        #We add the absolute constraints
        for v in var:
            keys[v] = i
            A[0, i] = v.value.end
            A[i, 0] = -v.value.start
            i += 1
        #And now the relative constraints
        for const in self.get_constraints():
            a = keys[const.va]
            b = keys[const.vb]
            A[a, b] = const.constraint.end
            A[b, a] = -const.constraint.start
        #Floyd-Warshall
        for i in xrange(n):
            np.add(A[i, :].reshape(1, n), A[:, i].reshape(n, 1), B)
            np.minimum(A, B, A)
        #Rounding to avoid precision errors
        np.around(A, 3, A)
        #Consistency checking (the diagonal must be all 0)
        if np.any(np.diagonal(A)):
            raise InconsistencyError()
        #Variable interval updating (we know the network is consistent)
        for v in keys:
            key = keys[v]
            newval = Interval(-A[key, 0], A[0, key])
            if v.value != newval:
                modified.add(v)
                v.value = newval
        self.unconstrained = False
        return modified


class InconsistencyError(Exception):
    """Exception raised when the constraint network is inconsistent"""
    pass


if __name__ == "__main__":
    # pylint: disable-msg=C0103
    import time

    v0, v1, v2, v3 = [Variable() for _ in xrange(4)]
    v0.value = Interval(0, 0)
    print 'v0:' + str(v0) + ' v1:' + str(v1) +\
         ' v2:' + str(v2) + ' v3:' + str(v3)
    nw = ConstraintNetwork()
    nw.set_before(v0, v1)
    nw.add_constraint(v1, v2, Interval(3, 5))
    nw.add_constraint(v2, v3, Interval(2, 25))
    nw.add_constraint(v2, v3, Interval(3, 27))
    nw.add_constraint(v3, v2, Interval(-24, -2))
    nw.update_constraint(v2, v3, Interval(4, 25))
    nw.minimize_network()
    print 'v0:' + str(v0) + ' v1:' + str(v1) +\
         ' v2:' + str(v2) + ' v3:' + str(v3)
    v3.value = Interval(89, 89)
    nw.minimize_network()
    print 'v0:' + str(v0) + ' v1:' + str(v1) +\
         ' v2:' + str(v2) + ' v3:' + str(v3)
    v1.value = Interval(62, 65)
    nw.minimize_network()
    print 'v0:' + str(v0) + ' v1:' + str(v1) +\
         ' v2:' + str(v2) + ' v3:' + str(v3)
    #Known example assertion (Detcher STP example in TCN paper)
    v0, v1, v2, v3, v4 = [Variable() for _ in xrange(5)]
    v0.value = Interval(0, 0)
    nw = ConstraintNetwork()
    nw.add_constraint(v0, v1, Interval(10, 20))
    nw.add_constraint(v1, v2, Interval(30, 40))
    nw.add_constraint(v3, v2, Interval(10, 20))
    nw.add_constraint(v3, v4, Interval(40, 50))
    nw.add_constraint(v0, v4, Interval(60, 70))
    nw.minimize_network()
    assert v0.value == Interval(0, 0)
    assert v1.value == Interval(10, 20)
    assert v2.value == Interval(40, 50)
    assert v3.value == Interval(20, 30)
    assert v4.value == Interval(60, 70)
    nw.remove_constraint(v0, v1)
    try:
        nw.remove_constraint(v0, v3)
    except KeyError:
        pass
    else:
        raise ValueError('An exception should be thrown')
    #Performance test
    nvar = 500
    variables = [Variable() for _ in xrange(nvar)]
    variables[0].value = Interval(0, 0)
    nwl = ConstraintNetwork()
    for j in xrange(nvar-1):
        nwl.add_constraint(variables[j], variables[j+1], Interval(1, 10))
    t1 = time.clock()
    nwl.minimize_network()
    t2 = time.clock()
    print 'Time to process {0} constraints: {1:.5f}s'.format(nvar, t2-t1)
    #End of performance test
