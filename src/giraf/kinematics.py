"""GIRAF kinematic model: modified-DH forward kinematics and Jacobian.

The boom bends under its own weight and the end-effector load, so a
deflection transform is inserted between links 3 and 4. The symbolic model is
built and lambdified on first use.
"""

from functools import cache

import numpy as np
import sympy as sp

# Link lengths [m]
L1_CONST = 0.21
L2_CONST = 0.055
L4_CONST = 0.04325
L5_CONST = 0.14

# Boom deflection model
rho = 0.188  # boom mass per length [kg/m]
m_e = 0.5  # end-effector mass [kg]
g = 9.81  # [m/s^2]
EI_CONST = 91.24628715  # flexural rigidity [Nm^2]

JOINTS = th1, th2, d3, th4, th5, th6 = sp.symbols("th1 th2 d3 th4 th5 th6", real=True)

# Modified DH parameters: a(i-1), alpha(i-1), d(i), theta(i)
MDH_sym = {
    1: {"a": 0, "al": 0, "d": L1_CONST, "th": th1},
    2: {"a": 0, "al": sp.pi / 2, "d": 0, "th": th2},
    3: {"a": -L2_CONST, "al": sp.pi / 2, "d": d3, "th": 0},
    4: {"a": 0, "al": -sp.pi / 2, "d": 0, "th": th4},
    5: {"a": -L4_CONST, "al": sp.pi / 2, "d": 0, "th": th5},
    6: {"a": 0, "al": sp.pi / 2, "d": L5_CONST, "th": th6},
}


def sym_MDH_forward(dh_param):
    a, al, d, th = dh_param["a"], dh_param["al"], dh_param["d"], dh_param["th"]
    return sp.Matrix(
        [
            [sp.cos(th), -sp.sin(th), 0, a],
            [
                sp.sin(th) * sp.cos(al),
                sp.cos(th) * sp.cos(al),
                -sp.sin(al),
                -sp.sin(al) * d,
            ],
            [
                sp.sin(th) * sp.sin(al),
                sp.cos(th) * sp.sin(al),
                sp.cos(al),
                sp.cos(al) * d,
            ],
            [0, 0, 0, 1],
        ]
    )


def _link_transforms():
    return [sym_MDH_forward(MDH_sym[i]) for i in (1, 2, 3, 4, 5, 6)]


def _deflection_transform():
    """Cantilever bending of the extended boom under gravity."""
    L = d3 - 255 / 1000
    delta = (
        sp.cos(th2 - sp.pi / 2) / EI_CONST * (rho * g * L**4 / 8 + m_e * g * L**3 / 3)
    )
    phi = sp.cos(th2 - sp.pi / 2) / EI_CONST * (rho * g * L**3 / 6 + m_e * g * L**2 / 2)
    return sp.Matrix(
        [
            [1, 0, 0, 0],
            [0, sp.cos(-phi), -sp.sin(-phi), delta],
            [0, sp.sin(-phi), sp.cos(-phi), 0],
            [0, 0, 0, 1],
        ]
    )


def sym_forward_kinematics():
    T01, T12, T23, T34, T45, T56 = _link_transforms()
    return T01 @ T12 @ T23 @ _deflection_transform() @ T34 @ T45 @ T56


def sym_jacobian_linear(T):
    return T[:3, 3].jacobian(sp.Matrix(JOINTS))


def sym_jacobian_angular():
    # Specific to this joint layout: joint 3 is prismatic and contributes no rotation.
    T01, T12, T23, T34, T45, T56 = _link_transforms()
    T3d = _deflection_transform()
    T02 = T01 @ T12
    T03 = T02 @ T23
    T04 = T03 @ T3d @ T34
    T05 = T04 @ T45
    T06 = T05 @ T56
    return sp.Matrix.hstack(
        T01[:3, 2],
        T02[:3, 2],
        sp.zeros(3, 1),
        T04[:3, 2],
        T05[:3, 2],
        T06[:3, 2],
    )


@cache
def _numeric_model():
    T = sym_forward_kinematics()
    J = sp.Matrix.vstack(sym_jacobian_linear(T), sym_jacobian_angular())
    return (
        sp.lambdify(JOINTS, T, modules="numpy"),
        sp.lambdify(JOINTS, J, modules="numpy"),
    )


def num_forward_transform(joint_coords) -> np.ndarray:
    """Full 4x4 end-effector transform in the base frame."""
    return np.array(_numeric_model()[0](*joint_coords))


def num_jacobian(joint_coords) -> np.ndarray:
    """6x6 basic Jacobian (linear over angular)."""
    return np.array(_numeric_model()[1](*joint_coords))
