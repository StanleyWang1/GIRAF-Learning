import numpy as np
import sympy as sp

## -------------------- CONSTANTS (physical parameters) --------------------
# Link Lengths
L1_CONST = 0.21  # m
L2_CONST = 0.055  # m
L3_CONST = 0.133 # m

L4_CONST = 0.04325 # m
L5_CONST = 0.14 # m

# rho = 0.188 # [kg/m]
# m_e = 0.5 # [kg]
# g = 9.81 # m/s^2

# # Flexural rigidity of boom
# EI_CONST = 91.24628715 # Nm^2

## -------------------- KINEMATIC MODEL (dh_parameters) --------------------
th1, th2, d3, th4, th5, th6 = sp.symbols('th1 th2 d3 th4 th5 th6', real=True)
MDH_sym = {
    1: {'a': 0,         'al': 0,       'd': L1_CONST, 'th': th1},
    2: {'a': 0,         'al': sp.pi/2, 'd': 0,        'th': th2},
    3: {'a': -L2_CONST, 'al': sp.pi/2, 'd': d3,       'th': 0},
    4: {'a': 0,         'al': -sp.pi/2, 'd': 0,       'th': th4},
    5: {'a': -L4_CONST, 'al': sp.pi/2, 'd': 0,        'th': th5},
    6: {'a': 0,         'al': sp.pi/2, 'd': L5_CONST, 'th': th6}    
}

## Forward Kinematics for a link as defined by modified DH parameters
def sym_MDH_forward(dh_param):
    a = dh_param['a'] # a (i-1)
    al = dh_param['al'] # alpha (i-1)
    d = dh_param['d'] # d (i)
    th = dh_param['th'] # theta (i)
    return sp.Matrix([
        [sp.cos(th), -sp.sin(th), 0, a],
        [sp.sin(th)*sp.cos(al), sp.cos(th)*sp.cos(al), -sp.sin(al), -sp.sin(al)*d],
        [sp.sin(th)*sp.sin(al), sp.cos(th)*sp.sin(al), sp.cos(al), sp.cos(al)*d],
        [0, 0, 0, 1]
    ])

## SYMBOLIC Forward Kinematics for Modified DH Parameter Link Description
def sym_forward_kinematics(MDH):
    T = sp.eye(4)  # Initialize transform as identity matrix
    for i in sorted(MDH.keys()): # apply transforms for each link described by DH
        Ti = sym_MDH_forward(MDH[i])
        T = T @ Ti
    return T

# def sym_forward_kinematics_corrected(MDH):
#     # Rigid link transforms
#     T01 = sym_MDH_forward(MDH[1])
#     T12 = sym_MDH_forward(MDH[2])
#     T23 = sym_MDH_forward(MDH[3])
#     T34 = sym_MDH_forward(MDH[4])
#     T45 = sym_MDH_forward(MDH[5])
#     T56 = sym_MDH_forward(MDH[6])

#     # deflection compensation
#     L = d3 - 255/1000
#     delta = sp.cos(th2 - sp.pi/2) / EI_CONST * (rho * g * L**4 / 8 + m_e * g * L**3 / 3)
#     phi = sp.cos(th2 - sp.pi/2) / EI_CONST * (rho * g * L**3 / 6 + m_e * g * L**2 / 2)
#     T3d = sp.Matrix([[1, 0,             0,              0],
#                      [0, sp.cos(-phi),  -sp.sin(-phi),  delta],
#                      [0, sp.sin(-phi),  sp.cos(-phi),   0],
#                      [0, 0,             0,              1]])
    
#     T = T01 @ T12 @ T23 @ T3d @ T34 @ T45 @ T56
#     return T

## SYMBOLIC Linear Velocity Jacobian
def sym_jacobian_linear(T):
    x = T[0,3]
    y = T[1,3]
    z = T[2,3]
    Jv = sp.Matrix([[x.diff(th1), x.diff(th2), x.diff(d3), x.diff(th4), x.diff(th5), x.diff(th6)],
                    [y.diff(th1), y.diff(th2), y.diff(d3), y.diff(th4), y.diff(th5), y.diff(th6)],
                    [z.diff(th1), z.diff(th2), z.diff(d3), z.diff(th4), z.diff(th5), z.diff(th6)]])
    return Jv
    # return sp.simplify(Jv) # linear velocity Jacobian

## SYMBOLIC Angular Velocity Jacobian
def sym_jacobian_angular(MDH): # NOT OPTIMIZED FOR GENERAL MANIPULATOR STRUCTURES!
    # Get individual link transforms
    T01 = sym_MDH_forward(MDH[1])
    T12 = sym_MDH_forward(MDH[2])
    T23 = sym_MDH_forward(MDH[3])
    T34 = sym_MDH_forward(MDH[4])
    T45 = sym_MDH_forward(MDH[5])
    T56 = sym_MDH_forward(MDH[6])

    # deflection compensation
    # L = d3 - 255/1000
    # delta = sp.cos(th2 - sp.pi/2) / EI_CONST * (rho * g * L**4 / 8 + m_e * g * L**3 / 3)
    # phi = sp.cos(th2 - sp.pi/2) / EI_CONST * (rho * g * L**3 / 6 + m_e * g * L**2 / 2)
    # T3d = sp.Matrix([[1, 0,             0,              0],
    #                  [0, sp.cos(-phi),  -sp.sin(-phi),  delta],
    #                  [0, sp.sin(-phi),  sp.cos(-phi),   0],
    #                  [0, 0,             0,              1]])
    T3d = sp.eye(4) # NO deflection compensation (bypass)

    # Compute cumulative transforms (w/ deflection correction)
    T01_cum = T01
    T02_cum = T01 @ T12
    T03_cum = T01 @ T12 @ T23
    T04_cum = T01 @ T12 @ T23 @ T3d @ T34
    T05_cum = T01 @ T12 @ T23 @ T3d @ T34 @ T45
    T06_cum = T01 @ T12 @ T23 @ T3d @ T34 @ T45 @ T56

    # Extract z axes of joint 1, 2, 3 (in base frame)
    z1 = T01_cum[:3, 2]
    z2 = T02_cum[:3, 2]
    z3 = T03_cum[:3, 2] * 0 # prismatic joint!!!
    z4 = T04_cum[:3, 2]
    z5 = T05_cum[:3, 2]
    z6 = T06_cum[:3, 2]

    # Stack to form angular velocity Jacobian
    Jw = sp.Matrix.hstack(z1, z2, z3, z4, z5, z6)
    return Jw
    # return sp.simplify(Jw)

## NUMERICAL XYZ forward kinematics
def num_forward_kinematics(joint_coords):
    return np.array(FK_num(*joint_coords))
## NUMERICAL full SE(3) forward kinematics
def num_forward_transform(joint_coords):
    return np.array(FK_full_num(*joint_coords))
## NUMERICAL Basic Jacobian (6x6)
def num_jacobian(joint_coords):
    return np.array(J_num(*joint_coords))

print("Starting symbolic kinematic derivations")
T = sym_forward_kinematics(MDH_sym)
print("Computed FK")
# T_corr = sym_forward_kinematics_corrected(MDH_sym)
# print("Computed FK w/ deflection correction")
FK_num = sp.lambdify((th1, th2, d3, th4, th5, th6), T[:3,3], modules='numpy')
FK_full_num = sp.lambdify((th1, th2, d3, th4, th5, th6), T, modules='numpy')

Jv = sym_jacobian_linear(T)
print("Computed linear velocity jacobian")
Jw = sym_jacobian_angular(MDH_sym)
print("Computed angular velocity jacobian")
J = sp.Matrix.vstack(Jv, Jw)
J_num = sp.lambdify((th1, th2, d3, th4, th5, th6), J, modules='numpy')
