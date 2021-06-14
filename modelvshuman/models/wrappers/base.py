from abc import ABC, abstractmethod

class AbstractModel(ABC):

    @abstractmethod
    def softmax(self, logits):
        pass

    @abstractmethod
    def forward_batch(self, images):
        pass

