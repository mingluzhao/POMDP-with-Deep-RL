import collections
import numpy as np
import torch
from . import state as st
from torch.autograd import Variable
import logging
import warnings


class RandomVariable():
    """Base class for random variables. Supported methods:
        - sample(batch_size, num_particles)
        - sample_reparameterized(batch_size, num_particles)
        - logpdf(value, batch_size, num_particles)
    """

    def sample(self, batch_size, num_particles):
        """Returns a sample of this random variable."""

        raise NotImplementedError

    def sample_reparameterized(self, batch_size, num_particles):
        """Returns a reparameterized sample of this random variable."""

        raise NotImplementedError

    def pdf(self, value, batch_size, num_particles):
        """Evaluate the density of this random variable at a value. Returns
        Tensor/Variable [batch_size, num_particles].
        """

        raise NotImplementedError

    def logpdf(self, value, batch_size, num_particles):
        """Evaluate the log density of this random variable at a value. Returns
        Tensor/Variable [batch_size, num_particles].
        """

        raise NotImplementedError

    def kl_divergence(self, other_random_variable):
        """
        Compute the analytic KL-divergence between this and given random variable,
        i.e. KL(self||other_random_variable)
        """

        raise NotImplementedError


class StateRandomVariable(RandomVariable):
    """Collection of RandomVariable objects. Implements sample,
    sample_reparameterized, logpdf methods.

    E.g.

        state_random_variable = StateRandomVariable(random_variables={
            'a': Normal(
                mean=torch.zeros(3, 2),
                variance=torch.ones(3, 2)
            )
        })
        state_random_variable.b = MultivariateIndependentNormal(
            mean=torch.zeros(3, 2, 4, 5),
            variance=torch.ones(3, 2, 4, 5)
        )
        state = state_random_variable.sample(
            batch_size=3,
            num_particles=2
        )
        state_logpdf = state_random_variable.logpdf(
            value=state,
            batch_size=3,
            num_particles=2
        )
    """

    def __init__(self, **kwargs):
        # Needed because we overwrote normal __setattr__ to only allow torch tensors/variables
        object.__setattr__(self, '_items', {})

        for name in kwargs:
            self.set_random_variable_(name, kwargs[name])

    def __setitem__(self, name, value):
        self.__setattr__(name, value)

    def __getitem__(self, key):
        # Access elements
        if isinstance(key, str):
            return getattr(self, key)

        # Trick to allow state to be returned by nn.Module
        # Purposely only supports `0` to catch potential misuses
        if key == 0:
            if not self._values:
                raise KeyError("StateRandomVariable is empty")
            for key, value in self._values.items():
                return value

        raise KeyError('StateRandomVariable only supports slicing through the method slice_elements()')

    def __getattr__(self, name):
        if '_items' in self.__dict__:
            _items = self.__dict__['_items']
            if name in _items:
                return _items[name]
        raise AttributeError("'{}' object has no attribute '{}'".format(
            type(self).__name__, name))

    def __setattr__(self, name, value):
        if isinstance(value, RandomVariable):
            self.set_random_variable_(name, value)
        elif (
                ('_items' in self.__dict__) and
                (name in self__dict__['_items'])
        ):
            raise AttributeError(
                'cannot override assigned random variable {0} with a value '
                'that is not a RandomVariable: {1}'.format(name, value)
            )
        else:
            object.__setattr__(self, name, value)

    def named_random_variables(self):
        """Return a lazy iterator over random_variables"""
        for name, random_variable in self._items.items():
            yield name, random_variable

    def sample(self, batch_size, num_particles):
        state = st.State()
        for name, random_variable in self.named_random_variables():
            state[name] = random_variable.sample(
                batch_size=batch_size,
                num_particles=num_particles
            )

        return state

    def sample_reparameterized(self, batch_size, num_particles):
        state = st.State()
        for name, random_variable in self.named_random_variables():
            setattr(state, name, random_variable.sample_reparameterized(
                batch_size=batch_size,
                num_particles=num_particles
            ))
        return state

    def set_random_variable_(self, name, random_variable):
        if not isinstance(random_variable, RandomVariable):
            raise TypeError(
                'random_variable {} is not a RandomVariable'.
                    format(random_variable)
            )
        _items = self.__dict__['_items']
        _items[name] = random_variable

        return self

    def _find_common_keys(self, other):
        random_variable_keys = [key for key in self._items]
        other_keys = [key for key in other._items]
        common_keys = list(set(random_variable_keys) & set(other_keys))

        # logging.debug("Computing logpdf for states: {}".format(common_keys))

        if not set(common_keys) == set(random_variable_keys):
            logging.warning("Not all random variable key are used, only {} out of {}!".format(
                common_keys, random_variable_keys
            ))

        if not set(common_keys) == set(other_keys):
            logging.debug("Not all other keys are used/evaluated, only {} out of {}!".format(
                common_keys, other_keys
            ))

        return common_keys

    # add a name argument
    def logpdf(self, value, batch_size, num_particles):
        # assert(
        #     set([key for key, v in self.named_random_variables()]) ==
        #     set([key for key in value._values])
        # )

        common_keys = self._find_common_keys(value)
        print(common_keys)

        result = 0
        # for name, random_variable in self.named_random_variables():
        for name in common_keys:
            result += self._items[name].logpdf(
                # result += random_variable.logpdf(
                value=value[name],
                batch_size=batch_size,
                num_particles=num_particles
            )

        return result


class MultivariateIndependentNormal(RandomVariable):
    """MultivariateIndependentNormal random variable"""

    def __init__(self, mean, variance):
        """Initialize this distribution with mean, variance.

        input:
            mean: Tensor/Variable
                [batch_size, num_particles, dim_1, ..., dim_N]
            variance: Tensor/Variable
                [batch_size, num_particles, dim_1, ..., dim_N]
        """
        assert (mean.size() == variance.size())
        assert (len(mean.size()) > 2)
        self._mean = mean
        self._variance = variance

    def sample(self, batch_size, num_particles):
        assert (list(self._mean.size()[:2]) == [batch_size, num_particles])

        uniform_normals = torch.Tensor(self._mean.size()).normal_()
        return self._mean.detach() + \
               Variable(uniform_normals) * torch.sqrt(self._variance.detach())

    def sample_reparameterized(self, batch_size, num_particles):
        assert (list(self._mean.size()[:2]) == [batch_size, num_particles])

        standard_normal = MultivariateIndependentNormal(
            mean=Variable(torch.zeros(self._mean.size())),
            variance=Variable(torch.ones(self._variance.size()))
        )

        return self._mean + torch.sqrt(self._variance) * \
               standard_normal.sample(batch_size, num_particles)

    def logpdf(self, value, batch_size, num_particles):
        assert (value.size() == self._mean.size())
        assert (list(self._mean.size()[:2]) == [batch_size, num_particles])

        return torch.sum(
            (
                    -0.5 * (value - self._mean) ** 2 / self._variance -
                    0.5 * torch.log(2 * self._variance * np.pi)
            ).view(batch_size, num_particles, -1),
            dim=2
        )


# This class only needs logpdf because we don't sample from it.
# Test if we don't need this anymore
class MultivariateIndependentPseudobernoulli(RandomVariable):
    """MultivariateIndependentPseudobernoulli random variable"""

    def __init__(self, probability):
        """Initialize this distribution with probability.

        input:
            probability: Tensor/Variable
                [batch_size, num_particles, dim_1, ..., dim_N]
        """
        assert (len(probability.size()) > 2)
        self._probability = probability

    def logpdf(self, value, batch_size, num_particles, epsilon=1e-10):
        assert (value.size() == self._probability.size())
        assert (
                list(self._probability.size()[:2]) == [batch_size, num_particles]
        )

        return torch.sum(
            (
                    value * torch.log(self._probability + epsilon) +
                    (1 - value) * torch.log(1 - self._probability + epsilon)
            ).view(batch_size, num_particles, -1),
            dim=2
        )
