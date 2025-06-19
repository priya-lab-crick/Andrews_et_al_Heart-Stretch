import sys
sys.dont_write_bytecode = True
import os
SCRIPT_DIR = "../"
sys.path.append(os.path.dirname(SCRIPT_DIR))
import numpy as np
import jax.numpy as jnp
from jax import jacrev
from jax import jit
from functools import partial
from scipy.integrate import solve_ivp
import jax
from vertex_stretch_3d_periodic_rigid.mesh import Mesh,get_geometry,assemble_scalar
jax.config.update("jax_enable_x64", True)
jax.config.update('jax_platform_name', 'cpu')
from numpy.lib.scimath import sqrt as csqrt


class Tissue:
    def __init__(self,tissue_options=None,true_mesh_options=None):
        assert true_mesh_options is not None
        assert tissue_options is not None
        self.true_mesh_options = true_mesh_options

        ##Make an effective mesh in a box of size 1x1
        mesh_options = true_mesh_options.copy()
        mesh_options["L"] = 1
        mesh_options["A_init"] = true_mesh_options["A_init"]/true_mesh_options["L"]**2
        mesh_options["V_init"] = true_mesh_options["V_init"]/true_mesh_options["L"]**3
        mesh_options["init_noise"] = true_mesh_options["init_noise"]/true_mesh_options["L"]

        self.mesh = Mesh(mesh_options)
        self.tissue_options = tissue_options
        if "max_l_grad" not in self.tissue_options:
            self.tissue_options["max_l_grad"] = 100
        self.tissue_params = {}
        self.effective_tissue_params = {}
        self.jac = None

        self.initialize()

    def initialize(self):
        self.assign_notch_positive_cells()
        self.assign_tissue_parameters()
        self.initialize_energies()
        self.update_effective_tissue_params()
        self.initialize_L()

    def assign_notch_positive_cells(self):
        self.tissue_params["n_notch"] = int(np.round(self.tissue_options["p_notch"]*self.mesh.mesh_props["n_c"]))
        self.tissue_params["is_notch"] = np.zeros((int(self.mesh.mesh_props["n_c"])),dtype=np.bool_)
        self.tissue_params["is_notch"][:self.tissue_params["n_notch"]] = True
        np.random.seed(self.mesh.mesh_options["seed"])
        np.random.shuffle(self.tissue_params["is_notch"])

    def initialize_energies(self):
        self.jac = jit(jacrev(energy),static_argnums=(2,))

    def assign_tissue_parameters(self):
        self.tissue_params["A0"] = np.ones((int(self.mesh.mesh_props["n_c"])))*self.tissue_options["A0"][0]
        self.tissue_params["A0"][self.tissue_params["is_notch"]] = self.tissue_options["A0"][1]

        self.tissue_params["V0"] = np.ones((int(self.mesh.mesh_props["n_c"])))*self.tissue_options["V0"][0]
        self.tissue_params["V0"][self.tissue_params["is_notch"]] = self.tissue_options["V0"][1]

        self.tissue_params["F_bend"] = np.ones((int(self.mesh.mesh_props["n_c"]))) * self.tissue_options["F_bend"][0]
        self.tissue_params["F_bend"][self.tissue_params["is_notch"]] = self.tissue_options["F_bend"][1]

        self.tissue_params["kappa_A"] = np.ones((int(self.mesh.mesh_props["n_c"]))) * self.tissue_options["kappa_A"][0]
        self.tissue_params["kappa_A"][self.tissue_params["is_notch"]] = self.tissue_options["kappa_A"][1]

        self.tissue_params["kappa_P"] = np.ones((int(self.mesh.mesh_props["n_c"]))) * self.tissue_options["kappa_P"][0]
        self.tissue_params["kappa_P"][self.tissue_params["is_notch"]] = self.tissue_options["kappa_P"][1]

        self.tissue_params["P0"] = np.ones((int(self.mesh.mesh_props["n_c"]))) * self.tissue_options["P0"][0]
        self.tissue_params["P0"][self.tissue_params["is_notch"]] = self.tissue_options["P0"][1]


        self.tissue_params["kappa_V"] = np.ones((int(self.mesh.mesh_props["n_c"]))) * self.tissue_options["kappa_V"][0]
        self.tissue_params["kappa_V"][self.tissue_params["is_notch"]] = self.tissue_options["kappa_V"][1]

        self.tissue_params["mu_l"] = self.tissue_options["mu_l"]
        self.tissue_params["T_external"] = self.tissue_options["T_external"]
        self.tissue_params["l"] = 1.
        self.tissue_params["L0"] = float(self.true_mesh_options["L"])
        self.effective_tissue_params = self.tissue_params.copy()

    def update_effective_tissue_params(self):
        ##prefix parameters are rescaled by 1/L^2 to account for the effective damping coefficient of 1/L^2
        self.effective_tissue_params["L"] = self.tissue_params["L0"]*self.effective_tissue_params["l"]
        self.effective_tissue_params["Lz"] = self.tissue_params["L0"]/(self.effective_tissue_params["l"]**2)

        L = self.effective_tissue_params["L"]
        Lz = self.effective_tissue_params["Lz"]

        self.effective_tissue_params["kappa_V"] = self.tissue_params["kappa_V"]*L**4 * Lz**2
        self.effective_tissue_params["kappa_A"] = self.tissue_params["kappa_A"]*L**4
        self.effective_tissue_params["kappa_P"] = self.tissue_params["kappa_P"]*L**2
        self.effective_tissue_params["V0"] = self.tissue_params["V0"]/(L**2*Lz)

        self.effective_tissue_params["A0"] = self.tissue_params["A0"]/L**2
        self.effective_tissue_params["P0"] = self.tissue_params["P0"]/L


    def initialize_L(self):
        l_opt = get_l(self.mesh.mesh_props["A_apical"].mean(),self.mesh.mesh_props["P_apical"].mean(),self.tissue_params["L0"],self.tissue_params["A0"].mean(),
              self.tissue_params["kappa_A"].mean(),
              self.tissue_params["T_external"],
              self.tissue_params["kappa_P"].mean(),
              self.tissue_params["P0"].mean())
        if np.abs(l_opt.imag)>1e-15:
            l_opt = -1
        else:
            l_opt = l_opt.real

        if l_opt > 0:

            def f(t,_l):
                l = _l[0]
                return - _dE_dl(l,self.mesh.mesh_props, self.tissue_params)

            sol_l = solve_ivp(f,[0,1000],[l_opt])
            l_opt = sol_l.y[-1,-1]

        self.effective_tissue_params["l"] = l_opt
        self.update_effective_tissue_params()
        print(self.effective_tissue_params["L"],"L_init")

    def update_l(self,dt):
        dedl = dE_dl(self.effective_tissue_params["l"], self.mesh.mesh_props, self.tissue_params,int(self.mesh.mesh_props["n_c"]))
        dedl = jnp.clip(dedl,-self.tissue_options["max_l_grad"],self.tissue_options["max_l_grad"])
        self.effective_tissue_params["l"] += - dt*self.effective_tissue_params["mu_l"]*dedl

        if self.tissue_params["L0"]*self.effective_tissue_params["l"] < self.tissue_options["L_min"]:
            self.effective_tissue_params["l"] = self.tissue_options["L_min"]/self.tissue_params["L0"]
        self.update_effective_tissue_params()


    def update_x(self,X,dt):
        ##Apply the scaling factor to mu. Assuming that mu_x = 1 without a loss of generality.
        mu = jnp.array([1/self.effective_tissue_params["l"]**2,1/self.effective_tissue_params["l"]**2,self.effective_tissue_params["l"]**4])/self.tissue_params["L0"]**2

        F = - mu * self.jac(X,self.mesh.mesh_props,int(self.mesh.mesh_props["n_c"]),self.effective_tissue_params)
        X += dt*F
        Xz = X[...,2]
        X = jnp.column_stack([jnp.mod(X[...,:2],1),Xz])

        return X

    def update(self,X,dt):
        """Some care may need to be given regarding the ordering here"""
        self.mesh.update_mesh_props(X)
        self.update_l(dt)
        X = self.update_x(X,dt)
        X = self.mesh.perform_T1s(X)
        return X, self.effective_tissue_params["L"]


@partial(jit, static_argnums=(2,))
def energy(X,mesh_props,nc,tissue_params):
    mesh_props = get_geometry(X,mesh_props,nc)

    A_mask = (mesh_props["A_apical"]<tissue_params["A0"])
    E = tissue_params["kappa_V"]*(mesh_props["V"]-tissue_params["V0"])**2 \
        + tissue_params["kappa_P"]*(mesh_props["P_apical"]-tissue_params["P0"])**2 \
        + (tissue_params["kappa_A"]*(mesh_props["A_apical"]-tissue_params["A0"])**2) * A_mask \
        + (tissue_params["kappa_A"] * (mesh_props["A_apical"] - tissue_params["A0"])) * (~A_mask) \
        + tissue_params["T_external"]*mesh_props["A_apical"]

    sf = jnp.array([1, 1, 1 / tissue_params["l"] ** 3])
    X_R = mesh_props["X_R"] * sf
    Xp1_1 = mesh_props["Xp1_R"] * sf

    cos_phi_t = (X_R * Xp1_1).sum(axis=-1) / (
            jnp.linalg.norm(X_R, axis=-1) * jnp.linalg.norm(Xp1_1, axis=-1))
    phi_t = jnp.arccos(cos_phi_t)
    phi = assemble_scalar(phi_t, mesh_props["tri"], int(nc))
    E_i_phi = tissue_params["F_bend"] * (phi - jnp.pi * 2)
    E_phi = E_i_phi.sum()
    return E.sum()+E_phi.sum()


@partial(jit, static_argnums=(1,))
def get_phi(mesh_props, nc,l):
    """
    Sum of angles of the basal triangles, centred on the projected 2D centre of mass
    Where distances are scaled by the l scale factor.
    """
    sf = jnp.array([1,1, 1/l**3])
    X_R = mesh_props["X_R"]*sf
    Xp1_1 = mesh_props["Xp1_R"]*sf

    cos_phi_t = (X_R*Xp1_1).sum(axis=-1) / (
            jnp.linalg.norm(X_R, axis=-1) * jnp.linalg.norm(Xp1_1, axis=-1))
    phi_t = jnp.arccos(cos_phi_t)
    phi = assemble_scalar(phi_t, mesh_props["tri"], int(nc))
    return phi

@partial(jit, static_argnums=(2,))
def energy_phi_l(l,mesh_props,nc,tissue_params):
    phi = get_phi(mesh_props, nc,l)
    E_i_phi = tissue_params["F_bend"]*(phi-jnp.pi*2)
    return E_i_phi.sum()

@partial(jit, static_argnums=(2,))
def dE_dL_phi(l,mesh_props,nc,tissue_params):
    return jacrev(energy_phi_l)(l,mesh_props,nc,tissue_params)


@jit
def _dE_dl(l,mesh_props,tissue_params):
    """
    Note that global tissue size transformations are volume preserving so volume doesn't factor in
    excluding phi component; for initialisation where phi = 0
    """
    A0 = tissue_params["A0"]
    ka = tissue_params["kappa_A"]
    Tex = tissue_params["T_external"]
    a = mesh_props["A_apical"]
    kp = tissue_params["kappa_P"]
    P0 = tissue_params["P0"]
    p = mesh_props["P_apical"]
    L0 = tissue_params["L0"]
    L = L0*l
    dE_dli = (4*a*ka*l*L0**2*(-A0 + a*l**2*L0**2))*(a*L**2 < A0) + (2*a*ka*l*L0**2)*(a*L**2 >= A0) + 2*kp*L0*p*(l*L0*p - P0) + 2*a*l*L0**2*Tex
    return dE_dli.sum()

@jit
def _dE_dl_jac(l,mesh_props,tissue_params):
    return jacrev(_dE_dl)(l,mesh_props)


@partial(jit, static_argnums=(3,))
def dE_dl(l,mesh_props,tissue_params,nc):
    return _dE_dl(l,mesh_props,tissue_params) + dE_dL_phi(l,mesh_props,nc,tissue_params)


def get_l(a,p,L0,A0,ka,Tex,kp,P0):
    Sqrt = csqrt
    l1 = (2 * 2 ** 0.3333333333333333 * (2 * a * A0 * ka * L0 - kp * L0 * p ** 2 - a * L0 * Tex)) / (
                864 * a ** 4 * ka ** 2 * kp * L0 ** 6 * p * P0 + Sqrt(
            746496 * a ** 8 * ka ** 4 * kp ** 2 * L0 ** 12 * p ** 2 * P0 ** 2 - 55296 * a ** 6 * ka ** 3 * L0 ** 9 * (
                        2 * a * A0 * ka * L0 - kp * L0 * p ** 2 - a * L0 * Tex) ** 3)) ** 0.3333333333333333 + (
                864 * a ** 4 * ka ** 2 * kp * L0 ** 6 * p * P0 + Sqrt(
            746496 * a ** 8 * ka ** 4 * kp ** 2 * L0 ** 12 * p ** 2 * P0 ** 2 - 55296 * a ** 6 * ka ** 3 * L0 ** 9 * (
                        2 * a * A0 * ka * L0 - kp * L0 * p ** 2 - a * L0 * Tex) ** 3)) ** 0.3333333333333333 / (
                12. * 2 ** 0.3333333333333333 * a ** 2 * ka * L0 ** 3)
    l2 = -((2**0.3333333333333333*(1 + Sqrt(-3))*(2*a*A0*ka*L0 - kp*L0*p**2 - a*L0*Tex))/(864*a**4*ka**2*kp*L0**6*p*P0 + Sqrt(746496*a**8*ka**4*kp**2*L0**12*p**2*P0**2 - 55296*a**6*ka**3*L0**9*(2*a*A0*ka*L0 - kp*L0*p**2 - a*L0*Tex)**3))**0.3333333333333333) - ((1 - Sqrt(-3))*(864*a**4*ka**2*kp*L0**6*p*P0 + Sqrt(746496*a**8*ka**4*kp**2*L0**12*p**2*P0**2 - 55296*a**6*ka**3*L0**9*(2*a*A0*ka*L0 - kp*L0*p**2 - a*L0*Tex)**3))**0.3333333333333333)/(24.*2**0.3333333333333333*a**2*ka*L0**3)
    l3 = -((2**0.3333333333333333*(1 - Sqrt(-3))*(2*a*A0*ka*L0 - kp*L0*p**2 - a*L0*Tex))/(864*a**4*ka**2*kp*L0**6*p*P0 + Sqrt(746496*a**8*ka**4*kp**2*L0**12*p**2*P0**2 - 55296*a**6*ka**3*L0**9*(2*a*A0*ka*L0 - kp*L0*p**2 - a*L0*Tex)**3))**0.3333333333333333) - ((1 + Sqrt(-3))*(864*a**4*ka**2*kp*L0**6*p*P0 + Sqrt(746496*a**8*ka**4*kp**2*L0**12*p**2*P0**2 - 55296*a**6*ka**3*L0**9*(2*a*A0*ka*L0 - kp*L0*p**2 - a*L0*Tex)**3))**0.3333333333333333)/(24.*2**0.3333333333333333*a**2*ka*L0**3)

    l_vals = np.array([l1,l2,l3])
    l_vals_real = l_vals.real
    is_real = np.abs(l_vals.imag)<1e-14
    all_real = is_real.sum(axis=0)==3
    l_low = (~all_real)*l1 + all_real*(l_vals_real.max(axis=0))

    l_high = (kp*p*P0)/(L0*(a*ka + kp*p**2 + a*Tex))
    l = l_low*(a*(l_high*L0)**2<A0) + l_high*(a*(l_high*L0)**2>=A0)
    return l
