import dune.istl
# TODO: use the right include here
from dune.istl._istl import BlockVector as DuneVector
import ipyultraweak as uw

from pymor.vectorarrays.list import CopyOnWriteVector

class WrappedDuneVector(CopyOnWriteVector):
    """
    Wrapper class for the DUNE-internal vector type
    """

    def __init__(self, vector):
        assert isinstance(vector, dune.istl._istl.BlockVector)
        self._impl = vector

    @classmethod
    def from_instance(cls, instance):
        return cls(instance._impl)

    def to_numpy(self, ensure_copy=False):
        raise NotImplementedError

    def _copy_data(self):
        self._impl = DuneVector(self._impl)

    def _scal(self, alpha):
        self._impl *= alpha

    def _axpy(self, alpha, x):
        assert isinstance(x, WrappedDuneVector)
        self._impl.axpy(alpha, x._impl)

    def inner(self, other, product=None):
        assert isinstance(other, WrappedDuneVector)
        return self._impl.dot(other._impl)

    # This is the Euclidian norm on the coefficients!
    def norm(self):
        import numpy as np
        return np.sqrt(self.inner(self))

    def norm2(self):
        raise NotImplementedError

    def sup_norm(self):
        raise NotImplementedError

    def dofs(self, dof_indices):
        raise NotImplementedError

    def amax(self):
        raise NotImplementedError

    def __len__(self):
        return len(self._impl)


from pymor.vectorarrays.list import ListVectorSpace

class DuneVectorSpace(ListVectorSpace):
    """
    Wraps the WrappedDuneVector into a vector space
    """

    def __init__(self, dim):
        self.dim = dim

    def zero_vector(self):
        impl = DuneVector(self.dim)
        impl *= 0.0
        return WrappedDuneVector(impl)

    def make_vector(self, obj):
        return WrappedDuneVector(obj)

    def __eq__(self, other):
        return type(other) is DuneVectorSpace and self.dim == other.dim


from pymor.operators.interface import Operator

class DuneOperator(Operator):
    """
    The Operator holding the operator matrix/matrices. Needed for reduction later on.
    """

    def __init__(self, duneObj):
        self.duneObj = duneObj
        self.source = DuneVectorSpace(duneObj.dim_source)
        self.range = DuneVectorSpace(duneObj.dim_range)
        self.linear = True

    def apply(self, U, mu=None):
        assert U in self.source
        V = self.range.zeros(len(U))

        def apply_once(u, v):
            self.duneObj.apply(u._impl, v._impl)
            return v

        return self.range.make_array([apply_once(u, v) for (u, v) in zip(U.vectors, V.vectors)])

from pymor.models.basic import StationaryModel
from pymor.operators.constructions import VectorOperator

class DuneUltraweakTransportModel(StationaryModel):
    """
    The model where we can actually call .solve()
    """

    def __init__(self, operator, rhs, solver, solverConfig, output_functional=None,
                 products=None, error_estimator=None, name="dune-ultraweak transport model"):
        self.solver = solver
        self.solverConfig = solverConfig
        super().__init__(operator=operator, rhs=rhs,
                         output_functional=output_functional, products=products,
                         error_estimator=error_estimator, name=name)

    def _compute_solution(self, mu=None, **kwargs):
        def solve_once(mu):
            return WrappedDuneVector(self.solver.solve(mu.to_numpy(), self.solverConfig))

        if 'Acw' in mu:
            ret = [solve_once(mu)]
        else:
            ret = [solve_once(m) for m in mu]

        return self.operator.range.make_array(ret)

    def visualize(self, U, mu, filename="", **kwargs):
        for u in U.vectors:
            self.solver.visualize(u._impl, mu.to_numpy(), filename)


import numpy as np
class FakeErrorEstimator():
    """
    Calculates the true error in the mu-dependent norm for given mu
    """
    def __init__(self, fom, reductor):
        self.fom = fom
        self.reductor = reductor
        self.snapshots = {}

    def estimate_error(self, U, mu, m):
        U_fom = self.fom.solve(mu) # TODO: save these so they dont get recomputed every time
        U_rom = self.reductor.reconstruct(U)
        diff = U_rom - U_fom
        return np.sqrt(self.fom.operator.apply2(diff,diff,mu)).item()


from pymor.reductors.basic import StationaryRBReductor
class CustomReductor(StationaryRBReductor):
    """
    Slightly adapted StationaryRBReductor
    """
    def __init__(self, fom, RB=None, product=None, check_orthonormality=False, check_tol=None):
        super().__init__(fom, RB, product, check_orthonormality, check_tol)
        self.error_estimator = FakeErrorEstimator(self.fom, self)

    def assemble_error_estimator(self):
        return self.error_estimator


# TODO: generalize
from pymor.operators.interface import Operator
class DuneL2Product(Operator):
    def __init__(self, duneObj):
        self.duneObj = duneObj
        self.mat = duneObj.getL2MassMatrix()
        self.source = DuneVectorSpace(duneObj.dim_source)
        self.range = DuneVectorSpace(duneObj.dim_range)
        self.linear = True

    def apply(self, U, mu=None):
        assert U in self.source
        V = self.range.zeros(len(U))

        def apply_once(u, v):
            self.mat.mv(u._impl, v._impl)
            return v

        return self.range.make_array([apply_once(u, v) for (u, v) in zip(U.vectors, V.vectors)])

class DuneH1bProduct(Operator):
    def __init__(self, duneObj):
        self.duneObj = duneObj
        self.mat = duneObj.getH1bMassMatrix()
        self.source = DuneVectorSpace(duneObj.dim_source)
        self.range = DuneVectorSpace(duneObj.dim_range)
        self.linear = True

    def apply(self, U, mu=None):
        assert U in self.source
        V = self.range.zeros(len(U))

        def apply_once(u, v):
            self.mat.mv(u._impl, v._impl)
            return v

        return self.range.make_array([apply_once(u, v) for (u, v) in zip(U.vectors, V.vectors)])
