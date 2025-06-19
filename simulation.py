import sys
sys.dont_write_bytecode = True
import os
SCRIPT_DIR = "../"
sys.path.append(os.path.dirname(SCRIPT_DIR))
import numpy as np
import plotly.graph_objects as go
from tqdm import tqdm
import jax
import matplotlib.pyplot as plt
from vertex_stretch_3d_periodic_rigid.mesh import Mesh, get_geometry,assemble_scalar
from vertex_stretch_3d_periodic_rigid.tissue import Tissue
from matplotlib.patches import Polygon

jax.config.update("jax_enable_x64", True)
jax.config.update('jax_platform_name', 'cpu')
import matplotlib

matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42
plt.rc('font', family='Helvetica Neue')


class Simulation:
    def __init__(self, simulation_options=None, tissue_options=None, true_mesh_options=None):
        assert true_mesh_options is not None
        assert tissue_options is not None
        assert simulation_options is not None
        self.simulation_options = simulation_options
        self.t = Tissue(tissue_options, true_mesh_options)
        self.sim_params = {}
        self.sim_out = {}
        self.initialize()

    def initialize(self):
        self.generate_t_span()
        self.initialise_sim_out()

    def generate_t_span(self):
        self.sim_params["t_span"] = np.arange(0, self.simulation_options["tfin"], self.simulation_options["dt"])
        self.sim_params["dt"] = self.simulation_options["dt"]
        self.sim_params["t_span_save"] = self.sim_params["t_span"][::self.simulation_options["t_skip"]]
        self.sim_params["dt_save"] = self.simulation_options["dt"] * self.simulation_options["t_skip"]

    def initialise_sim_out(self):
        self.generate_t_span()
        for key, val in self.sim_params.items():
            self.sim_out[key] = val
        for key, val in self.t.tissue_options.items():
            self.sim_out[key] = val
        for key, val in self.t.true_mesh_options.items():
            self.sim_out[key] = val
        self.sim_out["X_save"] = np.zeros((len(self.sim_params["t_span_save"]), len(self.t.mesh.mesh_props["X"]), 3))
        self.sim_out["tri_save"] = np.zeros((len(self.sim_params["t_span_save"]), len(self.t.mesh.mesh_props["X"]), 3),
                                            dtype=int)
        self.sim_out["neigh_save"] = np.zeros(
            (len(self.sim_params["t_span_save"]), len(self.t.mesh.mesh_props["X"]), 3), dtype=int)
        self.sim_out["k2s_save"] = np.zeros((len(self.sim_params["t_span_save"]), len(self.t.mesh.mesh_props["X"]), 3),
                                            dtype=int)
        self.sim_out["R_save"] = np.zeros((len(self.sim_params["t_span_save"]), int(self.t.mesh.mesh_props["n_c"]), 3))
        self.sim_out["L_save"] = np.ones((len(self.sim_params["t_span_save"])))*-1

    def simulate(self):
        X = self.t.mesh.mesh_props["X"]
        L = self.t.effective_tissue_params["L"]
        k = 0
        if (L > self.t.tissue_options["L_min"]):
            if L < self.t.tissue_options["L_max"]:
                for i, tm in enumerate(tqdm(self.sim_params["t_span"])):
                    X, L = self.t.update(X, self.sim_params["dt"])
                    if (i + 1) % self.simulation_options["t_skip"] == 0 :
                        self.sim_out["X_save"][k] = X.copy()
                        self.sim_out["tri_save"][k] = self.t.mesh.mesh_props["tri"].copy()
                        self.sim_out["neigh_save"][k] = self.t.mesh.mesh_props["neigh"].copy()
                        self.sim_out["k2s_save"][k] = self.t.mesh.mesh_props["k2s"].copy()
                        self.sim_out["R_save"][k] = self.t.mesh.mesh_props["R"].copy()
                        self.sim_out["L_save"][k] = L
                        k += 1
            else:
                print("L too large, skipping")
                self.sim_out["L_save"] = 1e3
        else:
            print("L unstable, skipping")


def get_vtx_by_cell(mesh_props):
    mesh_props = get_geometry(mesh_props["X"], mesh_props, int(mesh_props["n_c"]))
    cell_dict = {}  ##get all vertices by cell, sorted in CCW order
    vtx_position_dict = {}
    renormalised_X = (mesh_props["X_R"] + mesh_props["tR"])
    tri = np.array(mesh_props["tri"])
    for i in range(int(mesh_props["n_c"])):
        idx_i, idx_j = np.nonzero(tri == i)
        thetas = mesh_props["theta_i"][(idx_i, idx_j)]
        idx_sorted = idx_i[np.argsort(thetas)], idx_j[np.argsort(thetas)]
        cell_dict[i] = idx_sorted[0]
        vtx_position_dict[i] = renormalised_X[idx_sorted]
    return cell_dict, vtx_position_dict, renormalised_X


def _plot_2d(ax, renormalised_X, vtx_position_dict, colours, **args):
    ps = []
    for key, val in vtx_position_dict.items():
        p = Polygon(val[..., :2], facecolor=colours[key], **args)
        ps.append(p)

    for p in ps:
        ax.add_patch(p)
    ax.set(xlim=(renormalised_X[:, 0].min(), renormalised_X[:, 0].max()),
           ylim=(renormalised_X[:, 1].min(), renormalised_X[:, 1].max()), aspect=1)


def _plot_3d(vtx_position_dict, R,color_dict,aspectmode="data",zdirec=-1):
    fig = go.Figure()
    for key, _vtxs in vtx_position_dict.items():
        n_vtx = len(_vtxs)
        vtxs = np.zeros((n_vtx * 2 + 2, 3))
        vtxs[:n_vtx] = _vtxs
        vtxs[n_vtx:2 * n_vtx] = _vtxs
        vtxs[n_vtx:2 * n_vtx][:, -1] = 0.
        vtxs[2 * n_vtx:] = R[key]
        vtxs[2 * n_vtx + 1][-1] = 0.

        ##triangles
        tri_basal = np.column_stack(
            [np.arange(0, n_vtx, dtype=int), np.roll(np.arange(0, n_vtx, dtype=int), -1), np.repeat(2 * n_vtx, n_vtx)])
        tri_apical = np.column_stack(
            [np.arange(n_vtx, 2 * n_vtx, dtype=int), np.roll(np.arange(n_vtx, 2 * n_vtx, dtype=int), -1),
             np.repeat(2 * n_vtx + 1, n_vtx)])
        tri_lateral_1 = np.column_stack([np.arange(0, n_vtx, dtype=int), np.roll(np.arange(0, n_vtx, dtype=int), -1),
                                         np.roll(np.arange(n_vtx, 2 * n_vtx, dtype=int), -1)])
        tri_lateral_2 = np.column_stack(
            [np.roll(np.arange(n_vtx, 2 * n_vtx, dtype=int), -1), np.arange(n_vtx, 2 * n_vtx, dtype=int),
             np.arange(0, n_vtx, dtype=int)])

        tri = np.row_stack([tri_basal, tri_apical, tri_lateral_1, tri_lateral_2])

        fig.add_trace(go.Mesh3d(x=vtxs[:, 0], y=vtxs[:, 1], z=-zdirec*vtxs[:, 2],
                                i=tri[:, 0], j=tri[:, 1], k=tri[:, 2],color=color_dict[key]))
        fig.add_trace(go.Scatter3d(
            x=_vtxs[:,0], y=_vtxs[:,1], z=-zdirec*_vtxs[:,2],
            marker=dict(
                size=0,
                color="white",
            ),
            line=dict(
                color='white',
                width=8
            )
            ))
        fig.add_trace(go.Scatter3d(
            x=_vtxs[:,0], y=_vtxs[:,1], z=np.zeros_like(_vtxs[:,0]),
            marker=dict(
                size=0,
                color="white",
            ),
            line=dict(
                color='white',
                width=8
            )
            ))
    fig.update_scenes(aspectmode=aspectmode)
    return fig

def plot_3d(mesh_props,color_dict,aspectmode="data",zdirec=-1):
    cell_dict, vtx_position_dict, renormalised_X = get_vtx_by_cell(mesh_props)
    return _plot_3d(vtx_position_dict, mesh_props["R"],color_dict,aspectmode=aspectmode,zdirec=zdirec)
