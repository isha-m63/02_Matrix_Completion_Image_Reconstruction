import numpy as np
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from src.operators import MeasurementOperator



def test_forward_shape():
    """
    forward() must return shape (m,) not (m,1) or (n,n).

    A shape mismatch here causes silent broadcasting errors
    in the ADMM updates that are very hard to trace.
    """
    op = MeasurementOperator(n=16, m=50, seed=1)
    X = np.random.randn(16, 16)
    #print(op.forward(X).shape)
    assert op.forward(X).shape == (50,), f"Expected (50,), got {op.forward(X).shape}"

def test_adjoint():
    """
    Core mathematical requirement: <A(X), u> == <X, A*(u)>

    This is the definition of the adjoint operator. If it fails,
    the Z-update gradient is wrong and the solver will diverge.
    """
    rng = np.random.default_rng(0)
    op = MeasurementOperator(n=16, m=50, seed=1)
    X = rng.standard_normal((16, 16))
    u = rng.standard_normal(50)

    lhs = np.dot(op.forward(X), u)
    rhs = np.dot(X.ravel(), op.adjoint(u).ravel())
    #print(lhs, rhs)
    np.testing.assert_allclose(lhs, rhs, rtol=1e-10,
        err_msg="Adjoint test failed — A* is not the true adjoint of A")


def test_adjoint_shape():
    """
    adjoint() must return shape (n, n).

    SVT expects a 2D matrix. If adjoint returns a flat vector,
    SVT will silently operate on the wrong shape.
    """
    op = MeasurementOperator(n=16, m=50, seed=1)
    u = np.random.randn(50)
    #print(u)
    assert op.adjoint(u).shape == (16, 16), f"Expected (16, 16), got {op.adjoint(u).shape}"


def test_measure_equals_forward():
    """
    measure() and forward() must return identical results.

    They are semantic aliases: forward() is the mathematical operator,
    measure() is what you call once on X_true to generate y.
    If they ever diverge (e.g. noise is added to measure()),
    this test will catch it before it corrupts an experiment.
    """
    op = MeasurementOperator(n=16, m=50, seed=1)
    X = np.random.randn(16, 16)
    #print(op.measure(X), op.forward(X))
    np.testing.assert_array_equal(op.measure(X), op.forward(X), 
                                  err_msg="measure() and forward() must be identical")



def test_z_update_satisfies_constraint():
    """
    After z_update(), A(Z) must equal y to within numerical tolerance.

    This is the feasibility condition the Z-update exists to enforce.
    If it fails, the matrix inversion lemma formula is wrong —
    either the derivation or the np.linalg.solve call.
    This is the mathematical correctness test for the most complex method.
    """
    rng = np.random.default_rng(7)
    op = MeasurementOperator(n=16, m=50, seed=1)

    X_true = rng.standard_normal((16, 16))
    y = op.measure(X_true)

    #Simulate a mid-iteration ADMM state
    X = rng.standard_normal((16, 16))
    lam = rng.standard_normal((16, 16))
    rho = 1.0
    Z = op.z_update(X, lam, y, rho)
    residual = np.linalg.norm(op.forward(Z) - y)
    #print(residual)
    np.testing.assert_allclose(residual, 0.0, atol=1e-8,
        err_msg="Z-update does not satisfy measurement constraint A(Z) = y")


def test_deterministic_with_same_seed():
    """
    Same seed must produce identical measurement matrices A across runs.

    Reproducibility is non-negotiable for phase transition plots and reported metrics to be corrext.
    """
    op1 = MeasurementOperator(n=16, m=50, seed=42)
    op2 = MeasurementOperator(n=16, m=50, seed=42)
    #print(op1.A[0], op2.A[0])
    np.testing.assert_array_equal(op1.A, op2.A, 
                                  err_msg="Same seed must produce identical A matrix")



if __name__ == "__main__":
    test_forward_shape()
    test_adjoint()
    test_adjoint_shape()
    test_measure_equals_forward()
    test_z_update_satisfies_constraint()
    test_deterministic_with_same_seed()
    print("All operator tests passed.")