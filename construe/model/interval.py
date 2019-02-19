# -*- coding: utf-8 -*-
"""Module defining Interval model and operations"""
#Based on ActiveState recipe 576816

class Interval(object):
    """
    Represents an interval.
    Defined as closed interval [start,end), which includes the start and
    end positions.
    Start and end do not have to be numeric types.
    **NOTE:** This class implements equality and hash operations in a way that
    violates the Python assumption that *a==b -> hash(a)==hash(b)*. Equality is
    implemented by attribute comparison, while hash is implemented by object id.
    """

    __slots__ = ('_start', '_end')
    
    __hash__ = object.__hash__

    def __init__(self, start, end):
        "Construct, start must be <= end"
        self.set(start, end)

    def set(self, start, end):
        """Sets the limits of this interval, start must be <= end"""
        if start > end:
            raise ValueError('Start ({0}) must not be greater than end '
                             '({1})'.format(start, end))
        self._start = start
        self._end = end

    @property
    def start(self):
        """The interval's start"""
        return self._start

    @property
    def end(self):
        """The interval's end"""
        return self._end

    def __iter__(self):
        yield self._start
        yield self._end

    def __str__(self):
        "As string."
        return '[{0},{1}]'.format(self._start, self._end)

    def __repr__(self):
        "String representation."
        return '[{0},{1}]'.format(self._start, self._end)
    
    def __eq__(self, other):
        if other is None:
            return False
        return self._start == other.start and self._end == other.end

    def __lt__(self, other):
        if other is None:
            return False
        return self._start < other.start or (self._start == other.start
                                             and self._end < other.end)
        
    def __le__(self, other):
        if other is None:
            return False
        return self < other or self == other


    def cpy(self, other):
        """Sets the limits of this interval, by copying them from another one"""
        self.set(other.start, other.end)

    def intersection(self, other):
        "Intersection. @return: An empty intersection if there is none."
        if other < self:
            other, self = self, other
        if self.end <= other.start:
            return Interval(self.start, self.start)
        return Interval(other.start, min(self.end, other.end))

    def hull(self, other):
        "@return: Interval containing both self and other."
        if other < self:
            other, self = self, other
        return Interval(self.start, max(self.end, other.end))

    def overlap(self, other):
        "@return: True iff self intersects other."
        if other < self:
            other, self = self, other
        return self.end > other.start

    def overlapm(self, other):
        "@return: True iff selfs overlaps or meets other."
        if other < self:
            other, self = self, other
        return self.end >= other.start

    def move(self, offset):
        "@return: Interval displaced offset to start and end"
        return Interval(self._start+offset, self._end+offset)

    def __contains__(self, item):
        "@return: True iff item in self."
        return self._start <= item <= self._end

    def subset(self, other):
        "@return: True iff self is subset of other."
        return self._start >= other.start and self._end <= other.end


    def proper_subset(self, other):
        "@return: True iff self is proper subset of other."
        return self._start > other.start and self._end < other.end

    @property
    def zero_in(self):
        "@return: True iff 0 in self."
        return self._start <= 0 <= self._end

    @property
    def empty(self):
        "@return: True iff self is empty."
        return self._start == self._end

    @property
    def length(self):
        """@return: Difference between end and start"""
        return self._end - self._start

    @property
    def singleton(self):
        "@return: True iff self.end - self.start == 1."
        return self._end - self._start == 1


    def separation(self, other):
        "@return: The distance between self and other."
        if other < self:
            other, self = self, other
        return 0 if self.end > other.start else other.start - self.end
