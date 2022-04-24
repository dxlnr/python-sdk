from abc import ABC, abstractmethod


class BaselineTrainer(ABC):
    r"""Trainer class provides an API for feature-complete training in PyTorch"""
    
    model = abstract_attribute()
    dataset = abstract_attribute()

    @abstractmethod
    def train(self):
        r"""runs the training."""
        raise NotImplementedError()
