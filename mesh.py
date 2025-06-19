import numpy as np
import jax.numpy as jnp
from jax import jit
from functools import partial
import jax
from scipy import sparse
import triangle as tr
jax.config.update('jax_platform_name', 'cpu')
jax.config.update("jax_enable_x64", True)
jax.config.update('jax_platform_name', 'cpu')

class Mesh:
    """
    Let x be the 2D vertex position
    let z be the basal z distance of that vertex
    Let X be the 3D vector (x,z)
    Let r be the centre of mass of each cell, in 2D
    Let r_z be the z value of the centre of mass
    let R be the centre of mass of each cell in 3D: R = (r, r_z).
    """

    def __init__(self, mesh_options=None):
        assert mesh_options is not None
        self.mesh_options = mesh_options
        self.mesh_props = {}
        self.initialize()

    def initialize(self):
        self.generate_hexagonal()
        self.voronoi_triangulate()
        self.initialize_basal()
        self.initialize_displacements()
        self.initialize_mesh_props()
        self.initialize_T1s()

    def generate_hexagonal(self):
        """
        Generate a hexagonal grid such that each cell has on average an area A_init
        (supposing cells follow a voronoi about cell centres r)
        """
        r = hexagonal_lattice(int(self.mesh_options["L"]), int(np.ceil(self.mesh_options["L"])),
                              noise=self.mesh_options["init_noise"], A=self.mesh_options["A_init"],seed=self.mesh_options["seed"])
        r += 1e-3
        r = r[np.argsort(r.max(axis=1))[:int(self.mesh_options["L"] ** 2 / self.mesh_options["A_init"])]].astype(
            np.float32)
        r += self.mesh_options["L"]/4
        r = np.mod(r,self.mesh_options["L"])

        self.mesh_props["L"] = self.mesh_options["L"]
        self.mesh_props["r"] = r

    def voronoi_triangulate(self):
        """
        Initialize vertices as the voronoi triangulation of cell centres
        """
        y, dictionary = generate_triangulation_mask(self.mesh_props["r"], self.mesh_props["L"], self.mesh_props["L"])
        t = tr.triangulate({"vertices": y})
        tri = t["triangles"]
        self.mesh_props["n_c"] = self.mesh_props["r"].shape[0]
        tri = tri[(tri != -1).all(axis=1)]
        one_in = (tri < self.mesh_props["n_c"]).any(axis=1)
        new_tri = tri[one_in]
        n_tri = dictionary[new_tri]
        n_tri = remove_repeats(n_tri, self.mesh_props["n_c"])
        self.mesh_props["tri"] = n_tri
        self.mesh_props["n_v"] = self.mesh_props["tri"].shape[0]
        self.mesh_props["neigh"] = get_neighbours(self.mesh_props["tri"])
        self.mesh_props["t_r"] = self.mesh_props["r"][self.mesh_props["tri"]]
        self.mesh_props["x"] = jnp.array(circumcenter(self.mesh_props["t_r"], self.mesh_props["L"]))
        self.mesh_props["k2s"] = get_k2(self.mesh_props["tri"], self.mesh_props["neigh"])

    def initialize_basal(self):
        """
        Generate 3D coordinates of the basal vertices
        Project basally (+z direction)
        z_init = V_init/A_init
        """
        z = self.mesh_options["V_init"] / self.mesh_options["A_init"]
        self.mesh_props["X"] = jnp.column_stack((self.mesh_props["x"], jnp.ones(self.mesh_props["n_v"]) * z))
        self.mesh_props["R"] = jnp.column_stack((self.mesh_props["r"], jnp.ones(self.mesh_props["n_c"]) * z))
        self.mesh_props["tR"] = self.mesh_props["R"][self.mesh_props["tri"]]

    def initialize_displacements(self):
        self.mesh_props["_X"] = jnp.expand_dims(self.mesh_props["X"], axis=1)
        self.mesh_props["x"] = self.mesh_props["X"][..., :2]
        self.mesh_props["neigh_p1"] = jnp.roll(self.mesh_props["neigh"], -1, axis=1)
        self.mesh_props["Xp1"] = self.mesh_props["X"][self.mesh_props["neigh_p1"]]
        self.mesh_props["X_R"] = periodic_displacement(self.mesh_props["_X"], self.mesh_props["tR"],
                                                       self.mesh_props["L"])
        self.mesh_props["Xp1_R"] = periodic_displacement(self.mesh_props["Xp1"], self.mesh_props["tR"],
                                                         self.mesh_props["L"])

    def initialize_mesh_props(self):
        self.mesh_props = get_geometry(self.mesh_props["X"], self.mesh_props, int(self.mesh_props["n_c"]))

    def update_mesh_props(self, X):
        self.mesh_props = get_geometry(X, self.mesh_props, int(self.mesh_props["n_c"]))

    def perform_T1s(self, X):
        X, self.mesh_props = check_T1s(X, self.mesh_props, int(self.mesh_props["n_c"]), eps=self.mesh_options["eps"],
                                       mult=self.mesh_options["l_mult"])
        return X

    def initialize_T1s(self):
        X = self.perform_T1s(self.mesh_props["X"])
        self.mesh_props = get_geometry(X, self.mesh_props, int(self.mesh_props["n_c"]))

    def load_mesh_props(self,skeleton_mesh_props):
        self.mesh_props["R"] = skeleton_mesh_props["R"]
        self.mesh_props["r"] = skeleton_mesh_props["R"][...,:2]
        self.mesh_props["tri"] = skeleton_mesh_props["tri"]
        self.mesh_props["X"] = skeleton_mesh_props["X"]
        self.mesh_props["neigh"] = skeleton_mesh_props["neigh"]
        self.mesh_props["k2s"] = skeleton_mesh_props["k2s"]
        self.mesh_props["tR"] = self.mesh_props["R"][self.mesh_props["tri"]]
        self.initialize_mesh_props()

    def get_adjacency(self):
        edges = np.row_stack(
            [np.column_stack([self.mesh_props["tri"][:, i], self.mesh_props["tri"][:, (i + 1) % 3]]) for i in range(3)])
        self.mesh_props["adj"] = sparse.coo_matrix((np.ones(len(edges), dtype=int), (edges[:, 0], edges[:, 1])),
                                shape=(int(self.mesh_props["n_c"]), int(self.mesh_props["n_c"])))


def generate_triangulation_mask(x, L, max_d):
    ys = jnp.zeros((0, 2), dtype=jnp.float32)
    dictionary = jnp.zeros((0), dtype=jnp.int32)
    for i in [0, -1, 1]:
        for j in [0, -1, 1]:
            y = (x + np.array((i, j)) * L).astype(jnp.float32)
            if j == 0:
                if i == 0:
                    mask = jnp.ones_like(x[:, 0], dtype=jnp.bool_)
                else:
                    val = L * (1 - i) / 2
                    mask = jnp.abs(x[:, 0] - val) < max_d
            elif i == 0:
                val = L * (1 - j) / 2
                mask = jnp.abs(x[:, 1] - val) < max_d
            else:
                val_x = L * (1 - i) / 2
                val_y = L * (1 - j) / 2
                mask = jnp.sqrt((x[:, 0] - val_x) ** 2 + (x[:, 1] - val_y) ** 2) < max_d
            ys = jnp.row_stack((ys, y[mask]))
            dictionary = jnp.concatenate((dictionary, jnp.nonzero(mask)[0].astype(jnp.int32)))
    return ys, dictionary


def order_tris(tri):
    """
    For each triangle (i.e. row in **tri**), order cell ids in ascending order
    :param tri: Triangulation (n_v x 3) np.int32 array
    :return: the ordered triangulation
    """
    nv = tri.shape[0]
    for i in range(nv):
        Min = np.argmin(tri[i])
        tri[i] = tri[i, Min], tri[i, np.mod(Min + 1, 3)], tri[i, np.mod(Min + 2, 3)]
    return tri


def remove_repeats(tri, n_c):
    """
    For a given triangulation (nv x 3), remove repeated entries (i.e. rows)
    The triangulation is first re-ordered, such that the first cell id referenced is the smallest. Achieved via
    the function order_tris. (This preserves the internal order -- i.e. CCW)
    Then remove repeated rows via lexsort.
    NB: order of vertices changes via the conventions of lexsort
    Inspired by...
    https://stackoverflow.com/questions/31097247/remove-duplicate-rows-of-a-numpy-array
    :param tri: (nv x 3) matrix, the triangulation
    :return: triangulation minus the repeated entries (nv* x 3) (where nv* is the new # vertices).
    """
    tri = order_tris(np.mod(tri, n_c))
    sorted_tri = tri[np.lexsort(tri.T), :]
    row_mask = np.append([True], np.any(np.diff(sorted_tri, axis=0), 1))
    return sorted_tri[row_mask]


def get_neighbours(tri):
    """
    Given a triangulation, find the neighbouring triangles of each triangle.

    By convention, the column i in the output -- neigh -- corresponds to the triangle that is opposite the cell i in that triangle.

    Can supply neigh, meaning the algorithm only fills in gaps (-1 entries)

    :param tri: Triangulation (n_v x 3) np.int32 array
    :param neigh: neighbourhood matrix to update {Optional}
    :return: (n_v x 3) np.int32 array, storing the three neighbouring triangles. Values correspond to the row numbers of tri
    """
    n_v = tri.shape[0]
    neigh = np.ones_like(tri, dtype=np.int32) * -1
    tri_compare = np.concatenate((tri.T, tri.T)).T.reshape((-1, 3, 2))
    for j in range(n_v):
        tri_sample_flip = np.flip(tri[j])
        tri_i = np.concatenate((tri_sample_flip, tri_sample_flip)).reshape(3, 2)
        for k in range(3):
            if neigh[j, k] == -1:
                neighb, l = np.nonzero((tri_compare[:, :, 0] == tri_i[k, 0]) * (tri_compare[:, :, 1] == tri_i[k, 1]))
                neighb, l = neighb[0], l[0]
                neigh[j, k] = neighb
                neigh[neighb, np.mod(2 - l, 3)] = j
    return neigh.astype(np.int32)


def get_k2(tri, neigh):
    """
    To determine whether a given neighbouring pair of triangles needs to be re-triangulated, one considers the sum of
    the pair angles of the triangles associated with the cell centroids that are **not** themselves associated with the
    adjoining edge. I.e. these are the **opposite** angles.

    Given one cell centroid/angle in a given triangulation, k2 defines the column index of the cell centroid/angle in the **opposite** triangle

    :param tri: Triangulation (n_v x 3) np.int32 array
    :param neigh: Neighbourhood matrix (n_v x 3) np.int32 array
    :return:
    """
    three = np.array([0, 1, 2])
    nv = tri.shape[0]
    k2s = np.empty((nv, 3), dtype=np.int32)
    for i in range(nv):
        for k in range(3):
            neighbour = neigh[i, k]
            k2 = ((neigh[neighbour] == i) * three).sum()
            k2s[i, k] = k2
    return k2s


@partial(jit, static_argnums=(1,))
def get_rz(mesh_props, nc):
    """
    X is the 3D coordinate of vertices
    r is the 2D centre of mass

    :param X:
    :param r:
    :return:
    """

    mesh_props["theta_i"] = jnp.arctan2(mesh_props["X_R"][..., 1], mesh_props["X_R"][..., 0])
    mesh_props["theta_ip1"] = jnp.arctan2(mesh_props["Xp1_R"][..., 1], mesh_props["Xp1_R"][..., 0])

    mesh_props["t_r_z"] = (1 / (4 * np.pi)) * (mesh_props["_X"][..., 2] + mesh_props["Xp1"][..., 2]) * (
            (mesh_props["theta_ip1"] - mesh_props["theta_i"] + np.pi) % (jnp.pi * 2) - jnp.pi)

    mesh_props["r_z"] = assemble_scalar(mesh_props["t_r_z"], mesh_props["tri"],
                                        nc)
    mesh_props["r_z"] = jnp.clip(mesh_props["r_z"],1e-5,jnp.inf)
    return mesh_props


@partial(jit, static_argnums=(1,))
def get_A_apical(mesh_props, nc):
    mesh_props["tri_A_apical"] = 0.5 * jnp.cross(mesh_props["X_R"][..., :2], mesh_props["Xp1_R"][..., :2], axis=-1)
    mesh_props["A_apical"] = assemble_scalar(mesh_props["tri_A_apical"], mesh_props["tri"], nc)
    mesh_props["tA_apical"] = mesh_props["A_apical"][mesh_props["tri"]]
    return mesh_props


@partial(jit, static_argnums=(1,))
def get_2d_COM(mesh_props, nc):
    t_r = (mesh_props["X_R"] + mesh_props["Xp1_R"]) * jnp.expand_dims(
        jnp.cross(mesh_props["X_R"][..., :2], mesh_props["Xp1_R"][..., :2], axis=-1), 2)
    r_old = mesh_props["R"][..., :2]
    r_new = r_old + jnp.column_stack(
        [assemble_scalar(t_r[..., 0], mesh_props["tri"], nc),
         assemble_scalar(t_r[..., 1], mesh_props["tri"], nc)]) / jnp.expand_dims(6 * mesh_props["A_apical"] + 1e-17,
                                                                                 axis=-1)

    mesh_props["r"] = jnp.mod(r_new, mesh_props["L"])
    return mesh_props


@partial(jit, static_argnums=(1,))
def get_R(mesh_props, nc):
    mesh_props = get_2d_COM(mesh_props, nc)
    mesh_props = get_rz(mesh_props, nc)
    mesh_props["R"] = jnp.column_stack((mesh_props["r"], mesh_props["r_z"]))
    mesh_props["tR"] = mesh_props["R"][mesh_props["tri"]]
    return mesh_props



@partial(jit, static_argnums=(1,))
def get_A_basal(mesh_props, nc):
    mesh_props["tri_A_vec"] = 0.5 * jnp.cross(mesh_props["X_R"], mesh_props["Xp1_R"], axis=-1)
    mesh_props["tri_A_basal"] = jnp.linalg.norm(mesh_props["tri_A_vec"], axis=-1)
    mesh_props["A_basal"] = assemble_scalar(mesh_props["tri_A_basal"], mesh_props["tri"], nc)
    mesh_props["tA_basal"] = mesh_props["A_basal"][mesh_props["tri"]]

    return mesh_props

@partial(jit, static_argnums=(1,))
def get_V(mesh_props, nc):
    """
    Volume generated from a set of triangular 'segments', each made of three tetrahedrons
    """
    mesh_props["tri_V"] = (1 / 3) * (mesh_props["tri_A_vec"][..., 2] * mesh_props["Xp1"][..., 2]
                                     + mesh_props["tri_A_basal"] * (mesh_props["tR"][..., 2])
                                     + 0.5 * jnp.cross(mesh_props["X_R"][..., :2], mesh_props["Xp1_R"][..., :2]) *
                                     mesh_props["_X"][..., 2])  ##This has involved some proof
    mesh_props["V"] = assemble_scalar(mesh_props["tri_V"], mesh_props["tri"], nc)
    mesh_props["tV"] = mesh_props["V"][mesh_props["tri"]]
    return mesh_props


@partial(jit, static_argnums=(1,))
def get_A_lateral(mesh_props, nc):
    mesh_props["tA_lateral"] = 0.5 * mesh_props["lt_apical"] * (mesh_props["_X"][..., 2] + mesh_props["Xp1"][..., 2])
    mesh_props["A_lateral"] = assemble_scalar(mesh_props["tA_lateral"], mesh_props["tri"], nc)
    return mesh_props


@jit
def periodic_displacement(x1, x2, L):
    return jnp.mod(x1 - x2 + L / 2, L) - L / 2


@jit
def load_X(X, mesh_props, nc):
    mesh_props["X"] = X.reshape(-1, 3)
    mesh_props["_X"] = jnp.expand_dims(mesh_props["X"], axis=1)
    mesh_props["x"] = mesh_props["X"][..., :2]
    mesh_props["neigh_p1"] = jnp.roll(mesh_props["neigh"], -1, axis=1)
    mesh_props["Xp1"] = mesh_props["X"][mesh_props["neigh_p1"]]

    ##Note that this calculation is repeated twice, before and after updating Rs.
    mesh_props["X_R"] = periodic_displacement(mesh_props["_X"], mesh_props["tR"], mesh_props["L"])
    mesh_props["Xp1_R"] = periodic_displacement(mesh_props["Xp1"], mesh_props["tR"], mesh_props["L"])

    mesh_props["disp_t_basal"] = mesh_props["X_R"] - mesh_props["Xp1_R"]
    mesh_props["disp_t_apical"] = mesh_props["disp_t_basal"][..., :2]
    mesh_props["lt_apical"] = jnp.linalg.norm(mesh_props["disp_t_apical"], axis=-1)
    mesh_props["lt_basal"] = jnp.linalg.norm(mesh_props["disp_t_basal"], axis=-1)
    return mesh_props


@jit
def get_quartet(tri, neigh, k2s, tri_0i, tri_0j):
    a, b, d = jnp.roll(tri[tri_0i], -tri_0j)
    tri_1i, tri_1j = neigh[tri_0i, tri_0j], k2s[tri_0i, tri_0j]
    c = tri[tri_1i, tri_1j]
    tri0_da = (tri_0j + 1) % 3
    da_i = neigh[tri_0i, tri0_da]
    da_j = k2s[tri_0i, tri0_da]
    da = tri[da_i, da_j]

    tri0_ab = (tri_0j - 1) % 3
    ab_i = neigh[tri_0i, tri0_ab]
    ab_j = k2s[tri_0i, tri0_ab]
    ab = tri[ab_i, ab_j]

    tri1_cd = (tri_1j - 1) % 3
    cd_i = neigh[tri_1i, tri1_cd]
    cd_j = k2s[tri_1i, tri1_cd]
    cd = tri[cd_i, cd_j]

    tri1_bc = (tri_1j + 1) % 3
    bc_i = neigh[tri_1i, tri1_bc]
    bc_j = k2s[tri_1i, tri1_bc]
    bc = tri[bc_i, bc_j]

    return jnp.array(
        (tri_0i, tri_0j, tri_1i, tri_1j, a, b, c, d, da, ab, bc, cd, da_i, ab_i, bc_i, cd_i, da_j, ab_j, bc_j, cd_j))


@jit
def update_mesh(mesh_props, tri_0):
    """
    Update tri, neigh and k2. Inspect the equiangulation code for some inspo.
    :return:
    """
    tri_0i, tri_0j = tri_0
    tri, neigh, k2s = mesh_props["tri"], mesh_props["neigh"], mesh_props["k2s"]
    tri_1i, tri_1j = neigh[tri_0i, tri_0j], k2s[tri_0i, tri_0j]
    tri0_da = (tri_0j + 1) % 3
    da_i = neigh[tri_0i, tri0_da]
    da_j = k2s[tri_0i, tri0_da]

    tri1_bc = (tri_1j + 1) % 3
    bc_i = neigh[tri_1i, tri1_bc]
    bc_j = k2s[tri_1i, tri1_bc]

    neigh_new = neigh.copy()
    k2s_new = k2s.copy()

    tri_new = tri.copy()
    tri_new = tri_new.at[tri_0i, (tri_0j - 1) % 3].set(tri[tri_1i, tri_1j])
    tri_new = tri_new.at[tri_1i, (tri_1j - 1) % 3].set(tri[tri_0i, tri_0j])

    neigh_new = neigh_new.at[tri_0i, tri_0j].set(neigh[tri_1i, (tri_1j + 1) % 3])
    neigh_new = neigh_new.at[tri_0i, (tri_0j + 1) % 3].set(neigh[bc_i, bc_j])
    neigh_new = neigh_new.at[tri_0i, (tri_0j + 2) % 3].set(neigh[tri_0i, (tri_0j + 2) % 3])
    neigh_new = neigh_new.at[tri_1i, tri_1j].set(neigh[tri_0i, (tri_0j + 1) % 3])
    neigh_new = neigh_new.at[tri_1i, (tri_1j + 1) % 3].set(neigh[da_i, da_j])
    neigh_new = neigh_new.at[tri_1i, (tri_1j + 2) % 3].set(neigh[tri_1i, (tri_1j + 2) % 3])

    k2s_new = k2s_new.at[tri_0i, tri_0j].set(k2s[tri_1i, (tri_1j + 1) % 3])
    k2s_new = k2s_new.at[tri_0i, (tri_0j + 1) % 3].set(k2s[bc_i, bc_j])
    k2s_new = k2s_new.at[tri_0i, (tri_0j + 2) % 3].set(k2s[tri_0i, (tri_0j + 2) % 3])
    k2s_new = k2s_new.at[tri_1i, tri_1j].set(k2s[tri_0i, (tri_0j + 1) % 3])
    k2s_new = k2s_new.at[tri_1i, (tri_1j + 1) % 3].set(k2s[da_i, da_j])
    k2s_new = k2s_new.at[tri_1i, (tri_1j + 2) % 3].set(k2s[tri_1i, (tri_1j + 2) % 3])

    neigh_new = neigh_new.at[bc_i, bc_j].set(tri_0i)
    neigh_new = neigh_new.at[da_i, da_j].set(tri_1i)

    k2s_new = k2s_new.at[bc_i, bc_j].set(tri_0j)
    k2s_new = k2s_new.at[da_i, da_j].set(tri_1j)

    mesh_props["tri"], mesh_props["neigh"], mesh_props["k2s"] = tri_new, neigh_new, k2s_new
    return mesh_props, jnp.array((tri_1i, tri_1j))


@jit
def tri_update(val, quartet_info):
    val_new = val.copy()
    tri_0i, tri_0j, tri_1i, tri_1j, a, b, c, d, da, ab, bc, cd, da_i, ab_i, bc_i, cd_i, da_j, ab_j, bc_j, cd_j = quartet_info
    val_new[tri_0i, (tri_0j - 1) % 3] = val[tri_1i, tri_1j]
    val_new[tri_1i, (tri_1j - 1) % 3] = val[tri_0i, tri_0j]
    return val_new


@jit
def push_vertices_post_T1(X, mesh_props, tri_0, tri_1, l_new):
    X_0 = X[tri_0[0]]
    X_1 = X[tri_1[0]]
    dX = periodic_displacement(X_0, X_1, mesh_props["L"])
    mid_point = jnp.mod(X_1 + dX / 2, mesh_props["L"])
    l_dX = jnp.linalg.norm(dX[:2])
    T1_plane = jnp.array((-dX[1], dX[0], 0.)) / (l_dX+1e-17)
    X_0_new = jnp.mod(mid_point + T1_plane * l_new,
                      mesh_props["L"])
    X_1_new = jnp.mod(mid_point - T1_plane * l_new, mesh_props["L"])
    X_new = X.copy()
    X_new = X_new.at[tri_0[0]].set(X_0_new)
    X_new = X_new.at[tri_1[0]].set(X_1_new)
    return X_new


def check_T1s(_X, mesh_props, nc, eps=0.002, mult=1.02):
    """
    Swapping when l < epsilon
    epsilon = L*eps
    """

    X = _X.copy()
    Eps = eps * mesh_props["L"]
    l_new = Eps * mult
    mesh_props = load_X(X, mesh_props, nc)
    flip_mask = jnp.roll(mesh_props["lt_apical"] < Eps, 1, axis=1)
    if flip_mask.sum() > 0:
        continue_loop = True
        while continue_loop:
            q = jnp.argmax(flip_mask)
            tri_0 = q // 3, q % 3
            mesh_props, tri_1 = update_mesh(mesh_props, tri_0)
            X = push_vertices_post_T1(X, mesh_props, tri_0, tri_1, l_new)
            mesh_props = load_X(X, mesh_props, nc)
            flip_mask = jnp.roll(mesh_props["lt_apical"] < 0.002 * mesh_props["L"], 1, axis=1)
            if flip_mask.sum() == 0:
                continue_loop = False
    return X, mesh_props


@partial(jit, static_argnums=(1,))
def _get_geometry(mesh_props, nc):
    """
    Note that for all of the cell-based properties,there is a final 'float' row which mops up the 'edge' cases of the triangulation

    """
    ##Performed on the previous centre of mass
    # (reference-frame invariant calculations, provided displacement from previous iteration isn't too much to mess up
    # periodic calculations)

    mesh_props["P_apical"] = assemble_scalar(mesh_props["lt_apical"], mesh_props["tri"], nc)

    mesh_props = get_A_apical(mesh_props, nc)
    mesh_props = get_R(mesh_props, nc)

    mesh_props["X_R"] = periodic_displacement(mesh_props["_X"], mesh_props["tR"], mesh_props["L"])
    mesh_props["Xp1_R"] = periodic_displacement(mesh_props["Xp1"], mesh_props["tR"], mesh_props["L"])

    mesh_props = get_A_basal(mesh_props, nc)
    mesh_props = get_V(mesh_props, nc)
    mesh_props = get_A_lateral(mesh_props, nc)

    return mesh_props


@partial(jit, static_argnums=(2,))
def get_geometry(X, mesh_props, nc):
    mesh_props = load_X(X, mesh_props, nc)
    return _get_geometry(mesh_props, nc)


@partial(jit, static_argnums=(2,))
def assemble_scalar(tval, tri, nc):
    val = jnp.bincount(tri.ravel() + 1, weights=tval.ravel(), length=nc + 1)
    val = val[1:]
    return val


def hexagonal_lattice(_rows=3, _cols=3, noise=0.005, A=None,seed=1):
    """
    Assemble a hexagonal lattice
    :param rows: Number of rows in lattice
    :param cols: Number of columns in lattice
    :param noise: Noise added to cell locs (Gaussian SD)
    :return: points (nc x 2) cell coordinates.
    """
    if A is None:
        A = 1.
    _A = np.max(A)
    rows = int(np.round(_rows / np.sqrt(_A)))
    cols = int(np.round(_cols / np.sqrt(_A)))
    points = []
    np.random.seed(seed)
    for row in range(rows * 2):
        for col in range(cols):
            x = (col + (0.5 * (row % 2))) * np.sqrt(3)
            y = row * 0.5
            x += np.random.normal(0, noise)
            y += np.random.normal(0, noise)
            points.append((x, y))
    points = np.asarray(points)
    if A is not None:
        points = points * np.sqrt(2 * np.sqrt(3) / 3) * np.sqrt(_A)
    points = jnp.array(points)

    return points


def circumcenter(C, L):
    """
    Find the circumcentre (i.e. vertex position) of each triangle in the triangulation.

    :param C: Cell centroids for each triangle in triangulation (n_c x 3 x 2) np.float32 array
    :param L: Domain size (np.float32)
    :return: Circumcentres/vertex-positions (n_v x 2) np.float32 array
    """
    ri, rj, rk = C.transpose(1, 2, 0)
    r_mean = (ri + rj + rk) / 3
    disp = r_mean - L / 2
    ri, rj, rk = np.mod(ri - disp, L), np.mod(rj - disp, L), np.mod(rk - disp, L)
    ax, ay = ri
    bx, by = rj
    cx, cy = rk
    d = 2 * (ax * (by - cy) + bx * (cy - ay) + cx * (ay - by))
    ux = ((ax * ax + ay * ay) * (by - cy) + (bx * bx + by * by) * (cy - ay) + (cx * cx + cy * cy) * (
            ay - by)) / d
    uy = ((ax * ax + ay * ay) * (cx - bx) + (bx * bx + by * by) * (ax - cx) + (cx * cx + cy * cy) * (
            bx - ax)) / d
    vs = np.empty((ax.size, 2), dtype=np.float32)
    vs[:, 0], vs[:, 1] = ux, uy
    vs = np.mod(vs + disp.T, L).astype(np.float32)
    return vs
