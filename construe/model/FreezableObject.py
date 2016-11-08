# -*- coding: utf-8 -*-
# pylint: disable-msg=C0103
"""
Created on Tue Jul  2 19:44:31 2013

@author: T. Teijeiro
"""

from abc import ABCMeta

class FreezableMeta(ABCMeta):
    '''The metaclass for the abstract base class of Freezable objects'''
    def __new__(mcls, name, bases, namespace):
        fields = namespace.get('__slots__', ())
        for base in bases:
            oldfields = getattr(base, '_fields', None)
            if oldfields:
                fields = oldfields + fields
        #All freezable objects must use __slots__ for attribute definition.
        namespace.setdefault('__slots__', ())
        namespace['_fields'] = fields
        return ABCMeta.__new__(mcls, name, bases, namespace)


class FreezableObject(object):
    """
    This class provides utilities to "freeze" an object, this is, to guarantee
    that after the freeze operation no attributes of the object can be
    modified. If any of the attributes is also a FreezableObject, the freeze
    operation is called in depth-last order.
    """

    __metaclass__ = FreezableMeta

    __slots__ = ('__frozen__', )

    def __init__(self):
        self.__frozen__ = False

    def __eq__(self, other):
        """
        Implements equality comparison, by equality comparison of all the
        attributes but __frozen__
        """
        return (type(self) is type(other) and
                self._fields == other._fields and
                all(getattr(self, f, None) == getattr(other, f, None)
                                   for f in self._fields if f != '__frozen__'))


    @property
    def frozen(self):
        """
        Checks if this object is frozen, this is, no attributes nor
        methods can be set.
        """
        return getattr(self, '__frozen__', False)

    def __setattr__(self, name, value):
        if self.frozen and name != '__frozen__':
            raise AttributeError(self, 'Object {0} is now frozen'.format(self))
        return super(FreezableObject, self).__setattr__(name, value)

    def freeze(self):
        """
        Freezes the object, ensuring that no setting operations can be
        performed after that.
        """
        if not self.frozen:
            self.__frozen__ = True
            for field in self._fields:
                attr = getattr(self, field, None)
                if isinstance(attr, FreezableObject):
                    attr.freeze()

    def unfreeze(self):
        """
        Unfreezes the object, allowing for attribute modifications.
        """
        if self.frozen:
            self.__frozen__ = False
            for field in self._fields:
                attr = getattr(self, field, None)
                if isinstance(attr, FreezableObject):
                    attr.unfreeze()

    def references(self, obj):
        """
        Checks if this object references another one, this is, another object
        is an attribute of this object. If any attribute is a
        **FreezableObject** instance, then the property is checked recursively.
        """
        for attr in vars(self).itervalues():
            if attr is obj:
                return True
            if isinstance(attr, FreezableObject) and attr.references(obj):
                return True
        return False


if __name__ == "__main__":
    # pylint: disable-msg=W0201
    class FreezableTest(FreezableObject):
        __slots__ = ('attr1', 'attr2', 'attr3')

        """Dummy class to test the FreezableObject hierarchy"""
        def __init__(self):
            super(FreezableTest, self).__init__()
            self.attr1 = "val1"
            self.attr2 = "val2"

    freezable = FreezableTest()
    print(freezable.attr1, freezable.attr2, freezable.frozen)
    freezable.attr1 = "val1_updated"
    print(freezable.attr1, freezable.attr2, freezable.frozen)
    freezable.attr3 = FreezableTest()
    freezable.attr3.attr2 = "val2_updated"
    freezable.freeze()
    print(freezable.attr1, freezable.attr2, freezable.frozen)
    try:
        freezable.attr2 = "val2_updated"
    except TypeError:
        print(freezable.attr1, freezable.attr2, freezable.frozen)
