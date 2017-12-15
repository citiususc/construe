from unittest import TestCase
import numpy as np
from construe.knowledge.abstraction_patterns.segmentation.pwave import _CLASSIFIERS as classifier


class TestClassifier(TestCase):
    def test_classifier(self):
        limb = classifier[0]
        prec = classifier[1]

        X_test = np.loadtxt("pw_samples.csv", delimiter=",", skiprows=1)
        X_test, Y_test = X_test[:, 0:8], X_test[:, 8:]

        d1 = limb.decision_function(X_test)
        d2 = prec.decision_function(X_test)

        d = np.column_stack((d1, d2))

        np.testing.assert_almost_equal(d, Y_test)
