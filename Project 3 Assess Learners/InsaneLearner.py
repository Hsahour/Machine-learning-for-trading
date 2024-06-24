import numpy as np
from BagLearner import BagLearner
import LinRegLearner as lrl

class InsaneLearner(object):
    def __init__(self, verbose=False):
        self.learners = [BagLearner(learner=lrl.LinRegLearner, kwargs={}, bags=20, verbose=verbose) for _ in range(20)]
        self.verbose = verbose
    def add_evidence(self, Xtrain, Ytrain):
        for learner in self.learners:
            learner.add_evidence(Xtrain, Ytrain)
    def query(self, Xtest):
        predictions = np.array([learner.query(Xtest) for learner in self.learners])
        return predictions.mean(axis=0)
    def author(self):
        return "hsahour3"  

if __name__ == "__main__":
    print("InsaneLearner module ready.")
