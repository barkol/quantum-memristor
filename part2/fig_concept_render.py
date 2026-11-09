r"""Blender render of the three panels of the concept figure.
Usage: blender -b -P fig_concept_render.py [-- samples]
Writes figures/data/fig_concept_{a,b,c}.png and fig_concept_anchors.json with the
projected positions of the labelled objects; labels are added by fig_redesign.py."""
import json, sys
from pathlib import Path
import numpy as np
import bpy
from bpy_extras.object_utils import world_to_camera_view
from mathutils import Vector

HERE = Path(bpy.data.filepath).parent if bpy.data.filepath else None
HERE = Path(sys.argv[sys.argv.index("-P")+1]).resolve().parent
DATA = HERE.parent / "figures" / "data"
d = np.load(DATA/"fig_concept.npz")
X, Y, Z, RGB = d["X"], d["Y"], d["Z"], d["rgb"]
W1, W2 = d["w1"], d["w2"]
argv = sys.argv[sys.argv.index("--")+1:] if "--" in sys.argv else []
SAMPLES = int(argv[0]) if argv else 96
RES = (1500, 1120)
PLATE_Z = 2.75
PLATE_SCALE = 1.0


def lin(c):
    c = np.asarray(c, float)
    return np.where(c <= 0.04045, c/12.92, ((c+0.055)/1.055)**2.4)


FLAT = [False]


def surf(p):
    if FLAT[0]:
        return 0.75
    i = np.abs(X[0] - p[0]).argmin(); j = np.abs(Y[:, 0] - p[1]).argmin()
    return float(Z[j, i])


def mat(name, col, rough=0.3, alpha=1.0, emis=0.0):
    m = bpy.data.materials.new(name); m.use_nodes = True
    b = m.node_tree.nodes["Principled BSDF"]
    b.inputs["Base Color"].default_value = (*lin(col), 1)
    b.inputs["Roughness"].default_value = rough
    b.inputs["Alpha"].default_value = alpha
    if emis:
        b.inputs["Emission Color"].default_value = (*lin(col), 1)
        b.inputs["Emission Strength"].default_value = emis
    m.blend_method = "BLEND" if alpha < 1 else "OPAQUE"
    return m


def landscape():
    ny, nx = Z.shape
    z_ = np.full(Z.size, 0.75) if FLAT[0] else Z.ravel()
    v = np.stack([X.ravel(), Y.ravel(), z_], 1)
    idx = np.arange(nx*ny).reshape(ny, nx)
    q = np.stack([idx[:-1, :-1], idx[:-1, 1:], idx[1:, 1:], idx[1:, :-1]], -1).reshape(-1, 4)
    me = bpy.data.meshes.new("land")
    me.from_pydata(v.tolist(), [], q.tolist()); me.update()
    me.polygons.foreach_set("use_smooth", [True]*len(me.polygons)); me.update()
    ob = bpy.data.objects.new("land", me); bpy.context.collection.objects.link(ob)
    m = bpy.data.materials.new("land"); m.use_nodes = True
    img = bpy.data.images.new("landcol", nx, ny, alpha=True, float_buffer=True)
    img.colorspace_settings.name = "Non-Color"
    col = np.full((nx*ny, 3), 0.85) if FLAT[0] else RGB.reshape(-1, 3)
    img.pixels.foreach_set(np.c_[lin(col), np.ones(nx*ny)].ravel().astype(np.float32))
    img.update()
    n = m.node_tree.nodes; tc = n.new("ShaderNodeTexCoord"); tx = n.new("ShaderNodeTexImage")
    tx.image = img; tx.extension = "EXTEND"
    b = n["Principled BSDF"]; b.inputs["Roughness"].default_value = 0.55
    m.node_tree.links.new(tc.outputs["Generated"], tx.inputs["Vector"])
    m.node_tree.links.new(tx.outputs["Color"], b.inputs["Base Color"])
    me.materials.append(m)
    return ob


def sphere(p, r, m):
    bpy.ops.mesh.primitive_uv_sphere_add(segments=48, ring_count=32, radius=r, location=p)
    o = bpy.context.object; bpy.ops.object.shade_smooth(); o.data.materials.append(m)
    return o


def tube(pts, r, m):
    c = bpy.data.curves.new("t", "CURVE"); c.dimensions = "3D"
    c.bevel_depth = r; c.bevel_resolution = 6; c.use_fill_caps = True
    s = c.splines.new("POLY"); s.points.add(len(pts)-1)
    for k, p in enumerate(pts):
        s.points[k].co = (*p, 1)
    o = bpy.data.objects.new("t", c); bpy.context.collection.objects.link(o)
    c.materials.append(m)
    return o


def cone(p, direction, r, h, m):
    bpy.ops.mesh.primitive_cone_add(vertices=32, radius1=r, depth=h, location=p)
    o = bpy.context.object; bpy.ops.object.shade_smooth()
    o.rotation_mode = "QUATERNION"
    o.rotation_quaternion = Vector(direction).to_track_quat("Z", "Y")
    o.data.materials.append(m)
    return o


def descent(start, end, m, lift=0.13, r=0.034):
    """Arrow along the surface, stopping short of the sphere in the well."""
    s = np.linspace(0.14, 0.50, 40)
    pts = []
    for t in s:
        q = np.asarray(start)*(1-t) + np.asarray(end)*t
        pts.append((q[0], q[1], surf(q)+lift))
    tube(pts, r, m)
    a, b = np.array(pts[-4]), np.array(pts[-1])
    cone(tuple(b + 0.06*(b-a)/np.linalg.norm(b-a)), tuple(b-a), r*2.6, 0.16, m)


def arc(p, q, h, m, r=0.024, keep=None):
    t = np.linspace(0, 1, 80)
    pts = [(p[0]*(1-u)+q[0]*u, p[1]*(1-u)+q[1]*u, p[2]*(1-u)+q[2]*u + 4*h*u*(1-u)) for u in t]
    if keep is None:
        tube(pts, r, m)
    else:
        for a, b in keep:
            tube([x for x, u in zip(pts, t) if a <= u <= b], r, m)
    return pts[len(pts)//2]


def plate(m_plate, m_line):
    hx, hy = 1.15, 0.85
    bpy.ops.mesh.primitive_cube_add(location=(0, 0, PLATE_Z))
    o = bpy.context.object; o.scale = (hx, hy, 0.012); o.data.materials.append(m_plate)
    for gx in np.linspace(-hx, hx, 9):
        tube([(gx, -hy, PLATE_Z+0.014), (gx, hy, PLATE_Z+0.014)], 0.005, m_line)
    for gy in np.linspace(-hy, hy, 7):
        tube([(-hx, gy, PLATE_Z+0.014), (hx, gy, PLATE_Z+0.014)], 0.005, m_line)


def reset():
    bpy.ops.wm.read_factory_settings(use_empty=True)
    sc = bpy.context.scene
    sc.render.engine = "CYCLES"; sc.cycles.device = "CPU"; sc.cycles.samples = SAMPLES
    sc.cycles.use_denoising = True
    sc.render.resolution_x, sc.render.resolution_y = RES; sc.render.resolution_percentage = 100
    sc.render.film_transparent = True
    sc.view_settings.view_transform = "Standard"; sc.view_settings.look = "None"
    w = bpy.data.worlds.new("w"); w.use_nodes = True
    w.node_tree.nodes["Background"].inputs[0].default_value = (1, 1, 1, 1)
    w.node_tree.nodes["Background"].inputs[1].default_value = 0.72
    sc.world = w
    bpy.ops.object.light_add(type="SUN", rotation=(np.radians(38), np.radians(-22), np.radians(10)))
    bpy.context.object.data.energy = 2.6; bpy.context.object.data.angle = np.radians(18)
    bpy.ops.object.camera_add(location=(0.0, -6.7, 6.9))
    cam = bpy.context.object
    cam.data.lens = 52; cam.data.sensor_width = 36
    target = Vector((0.0, 0.18, 1.3))
    cam.rotation_mode = "QUATERNION"
    cam.rotation_quaternion = (target - cam.location).to_track_quat("-Z", "Y")
    sc.camera = cam
    return sc, cam


def panel(name):
    sc, cam = reset()
    FLAT[0] = name != "a"
    landscape()
    warm = mat("warm", d["warm"], 0.22)
    warm_soft = mat("warm_soft", d["warm"], 0.25, alpha=0.45)
    gold = mat("gold", d["gold"], 0.22)
    grey = mat("grey", d["grey"], 0.4)
    ghost = mat("ghost", (0.75, 0.75, 0.75), 0.4, alpha=0.35)
    dark = mat("dark", (0.15, 0.15, 0.15), 0.5)
    glass = mat("glass", (0.86, 0.90, 0.95), 0.2, alpha=0.30)
    line = mat("line", (0.45, 0.48, 0.52), 0.5)
    dec = mat("dec", d["blue"], 0.3)
    r, rs = 0.235, 0.17
    z = lambda q, rad: surf(q) + rad
    A = {}

    def branch(well, e, tag):
        c = well + np.asarray(e)
        p = (c[0], c[1], z(c, rs))
        sphere(p, rs, warm_soft)
        descent(c, well, warm if name == "a" else dec)
        A[f"in{tag}"] = p
        return p

    w1 = (W1[0], W1[1], z(W1, r)); w2 = (W2[0], W2[1], z(W2, r))
    A["w1"], A["w2"] = w1, w2
    E = (0.72, -0.62)

    if name == "a":
        branch(W1, E, "1")
        sphere(w1, r, warm)
        sphere(w2, r, ghost)
    else:
        e1, e2 = (E, E) if name == "b" else ((-0.72, -0.62), E)
        p1, p2 = branch(W1, e1, "1"), branch(W2, e2, "2")
        sphere(w1, r, warm); sphere(w2, r, warm)
        plate(glass, line)
        s1 = (e1[0]*PLATE_SCALE, e1[1]*PLATE_SCALE, PLATE_Z+0.13)
        s2 = (e2[0]*PLATE_SCALE, e2[1]*PLATE_SCALE, PLATE_Z+0.13)
        sphere(s1, 0.13, gold)
        if name == "c":
            sphere(s2, 0.13, gold)
        tube([p1, s1], 0.012, dark); tube([p2, s2], 0.012, dark)
        A["s1"], A["s2"] = s1, s2
        A["plate"] = (1.15, 0.85, PLATE_Z)
        h = 0.62
        if name == "b":
            A["arc"] = arc(w1, w2, h, warm, r=0.030)
        else:
            A["arc"] = arc(w1, w2, h, grey, r=0.022, keep=[(0.0, 0.34), (0.66, 1.0)])

    sc.render.filepath = str(DATA/f"fig_concept_{name}.png")
    bpy.ops.render.render(write_still=True)
    bpy.context.view_layer.update()
    return {k: list(world_to_camera_view(sc, cam, Vector(v)))[:2] for k, v in A.items()}


anch = {n: panel(n) for n in "abc"}
(DATA/"fig_concept_anchors.json").write_text(json.dumps(anch, indent=1))
print("rendered", anch)
