from collections import Hashable
import copy
from unittest import TestCase

from construe.model import FreezableObject, Interval, Variable, ConstraintNetwork


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

    def test_hashable(self):
        var = Variable(Interval(10, 150))
        assert isinstance(var, Hashable)

        var = [Variable(Interval(0, x)) for x in range(100)]
        varset = set(var)

        assert len(varset) == 100
        assert all(v in varset for v in var)

    def test_deep_copy(self):
        var1 = Variable(Interval(10, 150))
        var2 = copy.deepcopy(var1)
        var3 = copy.deepcopy(var1)

        assert var1 is not var2 and var1 is not var3
        assert var1 == var2 and var1 == var3
        assert var2 is not var3
        assert var2 == var3

        assert var1.value is not var2.value and var1.value is not var3.value
        assert var1.value == var2.value and var1.value == var3.value
        assert var2.value is not var3.value
        assert var2.value == var3.value

        assert hash(var1.value) == hash(var2.value) == hash(var3.value)
        assert hash(var1.value) is hash(var1.value) is hash(var3.value)

        assert hash(var1) != hash(var2) and hash(var1) != hash(var3)
        assert hash(var2) != hash(var3)


class TestConstraintNetWork(TestCase):
    def test_set_before(self):
        v0, v1 = [Variable(Interval(-1, x)) for x in range(2)]
        nw = ConstraintNetwork()
        nw.set_before(v0, v1)
        nw.minimize_network()
        assert v0 < v1
        assert v0.value.start == v1.value.start
        assert v0.value.start == -1

        nw.set_before(v1, v0)
        nw.minimize_network()
        assert v0 == v1
        assert v0.value == v1.value
        assert v0.value.start == -1

    def test_add_constraint(self):
        # Known example assertion (Detcher STP example in TCN paper)
        v0, v1, v2, v3, v4 = [Variable() for _ in range(5)]
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

        # Testing if a stricker constraint is applied
        v0, v1, v2, v3, v4 = [Variable() for _ in range(5)]
        v0.value = Interval(0, 0)
        nw = ConstraintNetwork()
        nw.add_constraint(v0, v1, Interval(10, 20))
        nw.add_constraint(v1, v2, Interval(30, 40))
        nw.add_constraint(v3, v2, Interval(10, 20))
        nw.add_constraint(v3, v4, Interval(40, 50))
        nw.add_constraint(v0, v4, Interval(60, 70))

        nw.add_constraint(v0, v1, Interval(10, 15))
        nw.add_constraint(v1, v2, Interval(30, 35))
        nw.add_constraint(v3, v2, Interval(10, 15))
        nw.add_constraint(v3, v4, Interval(40, 45))
        nw.add_constraint(v0, v4, Interval(60, 65))
        nw.minimize_network()
        assert v0.value == Interval(0, 0)
        assert v1.value == Interval(10, 10)
        assert v2.value == Interval(40, 40)
        assert v3.value == Interval(25, 25)
        assert v4.value == Interval(65, 65)

    def test_equal(self):
        v0 = Variable(Interval(0, 10))
        v1 = Variable(Interval(7, 15))

        nw = ConstraintNetwork()
        nw.set_equal(v0, v1)
        nw.minimize_network()
        assert v0 == v1

    def test_between(self):
        v0 = Variable(Interval(0, 10))
        v1 = Variable(Interval(7, 15))
        v2 = Variable(Interval(4, 10))

        nw = ConstraintNetwork()
        nw.set_between(v0, v1, v2)
        nw.minimize_network()
        assert v0 <= v1 <= v2

        v0 = Variable(Interval(0, 10))
        v1 = Variable(Interval(7, 15))
        v2 = Variable(Interval(4, 10))

        nw = ConstraintNetwork()
        nw.set_between(v2, v1, v0)
        nw.minimize_network()
        assert v2 <= v1 <= v0
