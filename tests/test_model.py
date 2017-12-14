from collections import Hashable
from unittest import TestCase

from construe.model import FreezableObject, Interval, Variable


class FreezableTest(FreezableObject):
    __slots__ = ('attr1', 'attr2', 'attr3')

    """Dummy class to test the FreezableObject hierarchy"""

    def __init__(self):
        super(FreezableTest, self).__init__()
        self.attr1 = "val1"
        self.attr2 = "val2"


class A(object):
    pass


class TestFreezableObject(TestCase):
    def test_frozen(self):
        freezable = FreezableTest()
        assert freezable.frozen is False
        freezable.freeze()
        assert freezable.frozen is True
        freezable.unfreeze()
        assert freezable.frozen is False

    def test_freeze(self):
        freezable = FreezableTest()
        freezable.freeze()
        try:
            freezable.attr1 = "val1_updated"
        except AttributeError:
            assert freezable.attr1 == "val1"
        else:
            self.fail("attribute is not frozen")

        freezable.unfreeze()
        freezable.attr2 = "val2_updated"
        assert freezable.attr2 == "val2_updated"

    def test_references(self):
        freezable1 = FreezableTest()
        freezable2 = FreezableTest()
        a = A()
        b = A()
        c = A()

        freezable1.attr1 = freezable2
        freezable1.attr2 = a
        freezable2.attr1 = c
        assert freezable1.references(a)
        assert not freezable1.references(b)
        assert freezable1.references(c)
        assert freezable1.references(freezable2)

    def test_hashable(self):
        freezable1 = FreezableTest()
        freezable2 = FreezableObject()
        assert isinstance(freezable1, Hashable)
        assert isinstance(freezable2, Hashable)

    def test_equal(self):
        freezable1 = FreezableTest()
        freezable2 = FreezableTest()

        freezable1.freeze()

        assert freezable1 == freezable2
        assert freezable1 is not freezable2

        freezable2.attr2 = "hello"

        assert freezable1 != freezable2


class TestInterval(TestCase):
    def test_basic_properties(self):
        inter = Interval(-1, 1)
        assert inter.start == -1
        assert inter.end == 1
        assert inter.length == 2

    def test_intersection(self):
        inter1 = Interval(0, 10)
        inter2 = Interval(15, 20)

        inter = inter1.intersection(inter2)
        assert inter.empty
        assert inter.start == 0

        inter = inter2.intersection(inter1)
        assert inter.empty
        assert inter.start == 0

        inter2 = Interval(5, 15)

        inter = inter2.intersection(inter1)
        assert inter.start == 5
        assert inter.end == 10

        inter = inter1.intersection(inter2)
        assert inter.start == 5
        assert inter.end == 10

        inter2 = Interval(10, 15)
        inter = inter1.intersection(inter2)
        assert inter.empty

        inter = inter2.intersection(inter1)
        assert inter.empty

        inter2 = Interval(-1, 0)
        inter = inter1.intersection(inter2)
        assert inter.empty

        inter = inter2.intersection(inter1)
        assert inter.empty

    def test_hull(self):
        inter1 = Interval(0, 10)
        inter2 = Interval(15, 20)

        inter = inter1.hull(inter2)
        assert inter.start == 0
        assert inter.end == 20

        inter = inter2.hull(inter1)
        assert inter.start == 0
        assert inter.end == 20

    def test_overlap(self):
        inter1 = Interval(0, 10)
        inter2 = Interval(15, 20)
        assert not inter1.overlap(inter2)
        assert not inter2.overlap(inter1)

        assert inter1.overlap(inter1)
        assert inter2.overlap(inter2)

        inter2 = Interval(-6, 0)
        assert not inter1.overlap(inter2)
        assert not inter2.overlap(inter1)

        inter2 = Interval(-6, 5)
        assert inter1.overlap(inter2)
        assert inter2.overlap(inter1)

        inter2 = Interval(10, 15)
        assert not inter1.overlap(inter2)
        assert not inter2.overlap(inter1)

        inter2 = Interval(4, 15)
        assert inter1.overlap(inter2)
        assert inter2.overlap(inter1)

    def test_overlapm(self):
        inter1 = Interval(0, 10)
        inter2 = Interval(15, 20)
        assert not inter1.overlapm(inter2)
        assert not inter2.overlapm(inter1)

        assert inter1.overlapm(inter1)
        assert inter2.overlapm(inter2)

        inter2 = Interval(-6, 0)
        assert inter1.overlapm(inter2)
        assert inter2.overlapm(inter1)

        inter2 = Interval(-6, 5)
        assert inter1.overlapm(inter2)
        assert inter2.overlapm(inter1)

        inter2 = Interval(10, 15)
        assert inter1.overlapm(inter2)
        assert inter2.overlapm(inter1)

        inter2 = Interval(4, 15)
        assert inter1.overlapm(inter2)
        assert inter2.overlapm(inter1)

    def test_contains(self):
        inter = Interval(0, 100)
        x = range(101)

        assert all([k in inter for k in x])

    def test_move(self):
        inter = Interval(0, 10)
        inter = inter.move(5)
        assert isinstance(inter, Interval)
        assert inter.start == 5
        assert inter.end == 15

        inter = inter.move(-5)
        assert inter.start == 0
        assert inter.end == 10

    def test_zero_in(self):
        inter1 = Interval(-6, 2)
        assert inter1.zero_in

        inter1 = Interval(0, 2)
        assert inter1.zero_in

        inter1 = Interval(-4, 0)
        assert inter1.zero_in

        inter1 = Interval(6, 10)
        assert not inter1.zero_in

        inter1 = Interval(-10, -3)
        assert not inter1.zero_in

        inter1 = Interval(float("-inf"), float("inf"))
        assert inter1.zero_in

    def test_subset(self):
        inter1 = Interval(0, 10)
        inter2 = Interval(4, 6)

        assert inter1.subset(inter1)
        assert inter2.subset(inter2)

        assert not inter1.subset(inter2)
        assert inter2.subset(inter1)

        inter2 = Interval(6, 12)
        assert not inter2.subset(inter1)
        inter2 = Interval(-6, 5)
        assert not inter2.subset(inter1)

        inter1 = Interval(float("-inf"), float("inf"))
        assert inter1.subset(inter1)
        assert inter2.subset(inter1)

    def test_proper_subset(self):
        inter1 = Interval(0, 10)
        inter2 = Interval(4, 6)

        assert not inter1.proper_subset(inter2)
        assert inter2.proper_subset(inter1)

        inter2 = Interval(6, 12)
        assert not inter2.proper_subset(inter1)
        inter2 = Interval(-6, 5)
        assert not inter2.proper_subset(inter1)

        inter1 = Interval(float("-inf"), float("inf"))
        assert inter2.proper_subset(inter1)

    def test_empty(self):
        inter = Interval(1, 1)
        assert inter.empty
        inter = Interval(-1, 1)
        assert not inter.empty

    def test_singleton(self):
        inter = Interval(0, 1)
        assert inter.singleton
        inter = Interval(1, 3)
        assert not inter.singleton

    def test_separation(self):
        inter1 = Interval(0, 3)
        inter2 = Interval(7, 10)
        assert inter1.separation(inter2) == 4
        assert inter2.separation(inter1) == 4

        inter1 = Interval(3, 5)
        inter2 = Interval(3, 7)
        assert inter1.separation(inter2) == 0
        assert inter2.separation(inter1) == 0

        inter1 = Interval(3, 5)
        inter2 = Interval(0, 5)
        assert inter1.separation(inter2) == 0
        assert inter2.separation(inter1) == 0

        inter1 = Interval(3, 5)
        inter2 = Interval(float("inf"), float("inf"))
        assert inter1.separation(inter2) == float("inf")
        assert inter2.separation(inter1) == float("inf")

        inter1 = Interval(3, 5)
        inter2 = Interval(float("-inf"), float("-inf"))
        assert inter1.separation(inter2) == float("inf")
        assert inter2.separation(inter1) == float("inf")

    def test_ordering(self):
        inter1 = Interval(0, 10)
        inter2 = Interval(10, 15)

        assert inter1 < inter2
        assert not inter1 > inter2
        assert inter1 != inter2

        inter2 = Interval(0, 10)
        assert inter1 == inter2
        assert not inter1 < inter2
        assert not inter1 > inter2

        inter2 = Interval(0, 5)
        assert inter1 > inter2
        assert inter1 != inter2
        assert not inter1 < inter2

        inter2 = Interval(0, 15)
        assert inter1 < inter2
        assert inter1 != inter2
        assert not inter1 > inter2

        inter2 = Interval(-5, 10)
        assert inter1 > inter2
        assert not inter1 < inter2
        assert inter1 != inter2

        inter2 = Interval(5, 10)
        assert inter1 < inter2
        assert not inter1 > inter2
        assert inter1 != inter2

        inter2 = Interval(-5, 6)
        assert inter1 > inter2
        assert not inter1 < inter2
        assert inter1 != inter2

        inter2 = Interval(5, 6)
        assert inter1 < inter2
        assert not inter1 > inter2
        assert inter1 != inter2

        inter2 = Interval(-5, 15)
        assert inter1 > inter2
        assert not inter1 < inter2
        assert inter1 != inter2

        inter2 = Interval(5, 15)
        assert inter1 < inter2
        assert not inter1 > inter2
        assert inter1 != inter2

        inter2 = None
        assert inter1 > inter2
        assert not inter1 < inter2
        assert inter1 != inter2

    def test_hash(self):
        inter = Interval(2, 20)
        assert hash(inter) == (2 ^ 20)
        x = [1, 7, 1, 23, 1, 7, 1, 23, 1, 7, 1, 23, 1, 7]

        for m in x:
            inter = inter.move(m)
            assert hash(inter) == (2 ^ 20)

        inter = Interval(float("-inf"), float("inf"))
        assert hash(inter) == hash(float("-inf")) ^ hash(float("inf"))


class TestVariable(TestCase):
    def test_properties(self):
        inter = Interval(0, 15)
        var = Variable(inter)
        assert var.start == 0
        assert var.end == 15

    def test_ordering(self):
        inter = Interval(0, 15)
        var1 = Variable(inter)

        var2 = Variable(inter.move(5))
        assert var1 < var2
        assert var1 != var2

        var2 = Variable(inter.move(-5))
        assert var1 > var2
        assert var1 != var2

        var2 = Variable(Interval(0, 5))
        assert var1 > var2
        assert var1 != var2

        var2 = Variable(Interval(0, 20))
        assert var1 < var2
        assert var1 != var2

        var2 = Variable(Interval(0, 15))
        assert var1 == var2
        assert not var1 < var2
        assert not var1 > var2

        var2 = Variable(None)
        assert var1 != var2
        assert var1 > var2
        assert not var1 < var2

        var2 = Variable()
        assert var1 != var2
        assert var1 > var2

    def test_hash(self):
        inter = Interval(0, 15)
        var = Variable(inter)
        assert isinstance(var, Hashable)
