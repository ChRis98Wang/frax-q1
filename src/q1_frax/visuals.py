"""MuJoCo contact diagnostics and appearance changes on in-memory render copies."""
from __future__ import annotations
import copy
from pathlib import Path

import mujoco
import numpy as np


def make_render_model(physical_model):
    model = copy.copy(physical_model)
    floor = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "floor")
    material = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_MATERIAL, "MatPlane")
    if material < 0:
        material = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_MATERIAL, "groundplane")
    if floor >= 0:
        model.geom_rgba[floor] = [1., 1., 1., 1.]
        if material >= 0:
            model.geom_matid[floor] = material
            model.mat_texrepeat[material] = [1., 1.]
            model.mat_reflectance[material] = .02
    model.vis.headlight.ambient[:] = [.3, .3, .3]
    model.vis.headlight.diffuse[:] = [.7, .7, .7]
    model.vis.headlight.specular[:] = [.25, .25, .25]
    return model


def physical_contacts(model, data):
    """Deepest penetrating contact for each actual geom pair, split by kind."""
    grouped = {"self": {}, "floor": {}, "environment": {}}
    for contact in data.contact:
        if contact.dist >= 0:
            continue
        ids = tuple(sorted((int(contact.geom1), int(contact.geom2))))
        bodies = [int(model.geom_bodyid[g]) for g in ids]
        if bodies[0] == bodies[1]:
            continue
        kind = "self" if all(bodies) else "environment"
        if not all(bodies) and any(model.geom_type[g] == mujoco.mjtGeom.mjGEOM_PLANE for g, b in zip(ids, bodies) if b == 0):
            kind = "floor"
        row = {"geom_ids": list(ids), "body_ids": bodies,
               "geoms": [model.geom(g).name or f"geom_{g}" for g in ids],
               "links": [model.body(b).name or ("world" if b == 0 else f"body_{b}") for b in bodies],
               "penetration_mm": float(-contact.dist*1000), "position": contact.pos.tolist()}
        if ids not in grouped[kind] or row["penetration_mm"] > grouped[kind][ids]["penetration_mm"]:
            grouped[kind][ids] = row
    return {kind: sorted(rows.values(), key=lambda row: row["penetration_mm"], reverse=True)
            for kind, rows in grouped.items()}


def excluded_body_pairs(model):
    return [[model.body(int(value) >> 16).name, model.body(int(value) & 65535).name]
            for value in model.exclude_signature]


def collision_summary(model, motion):
    data = mujoco.MjData(model)
    pairs = {kind: {} for kind in ("self", "floor", "environment")}
    counts = dict.fromkeys(pairs, 0)
    for frame, q in enumerate(motion.qpos):
        data.qpos[:] = q
        mujoco.mj_forward(model, data)
        for kind, contacts in physical_contacts(model, data).items():
            counts[kind] += bool(contacts)
            for contact in contacts:
                key = tuple(contact["geom_ids"])
                if key not in pairs[kind]:
                    pairs[kind][key] = {key: contact[key] for key in ("geom_ids", "body_ids", "geoms", "links")}
                    pairs[kind][key].update(frames=0, first_frame=frame, last_frame=frame, max_penetration_mm=0.)
                row = pairs[kind][key]
                row["frames"] += 1
                row["last_frame"] = frame
                row["max_penetration_mm"] = max(row["max_penetration_mm"], contact["penetration_mm"])
    return {**motion.provenance(), "contact_frames": counts,
            "pairs": {kind: sorted(rows.values(), key=lambda row: row["max_penetration_mm"], reverse=True) for kind, rows in pairs.items()},
            "excluded_body_pairs": excluded_body_pairs(model),
            "definition": "Actual MuJoCo mj_forward contacts with dist<0; self, floor-plane, and other static-environment contacts separated",
            "exclusion_note": "Only active physical collision geometry is tested. Excluded body pairs are listed explicitly; visual mesh overlap is not a contact test.",
            "original_model_or_motion_modified": False}


class CollisionColors:
    def __init__(self, model):
        self.model = model
        self.rgba = model.geom_rgba.copy()
        self.material = model.geom_matid.copy()

    def update(self, contacts, *, mark=True, show_geoms=False):
        model = self.model
        model.geom_rgba[:] = self.rgba
        model.geom_matid[:] = self.material
        if show_geoms:
            physical, visual = model.geom_group == 4, model.geom_group == 1
            model.geom_matid[physical] = -1
            model.geom_rgba[physical] = [.15, .7, .85, .55]
            model.geom_rgba[visual, 3] = .17
        if mark:
            for row in reversed(contacts["self"]):
                hit = np.isin(model.geom_bodyid, row["body_ids"])
                model.geom_matid[hit] = -1
                model.geom_rgba[hit] = [1., .12, .05, 1.] if row["penetration_mm"] >= 2 else [1., .53, .05, 1.]


def contact_lines(contacts, limit=3):
    rows = contacts["self"]
    lines = ([f"SELF: {len(rows)} penetrating geom pair(s)"] if rows else ["SELF: no penetrating physical contacts"])
    for row in rows[:limit]:
        lines.append(f"{row['links'][0]} <-> {row['links'][1]}: {row['penetration_mm']:.3f} mm")
        lines.append(f"  {row['geoms'][0]} / {row['geoms'][1]}")
    if len(rows) > limit:
        lines.append(f"... {len(rows)-limit} more geom pairs (see JSON report)")
    floor = contacts["floor"]
    maximum = max((row["penetration_mm"] for row in floor), default=0.)
    lines.append(f"FLOOR: {len(floor)} penetrating pair(s), maximum {maximum:.3f} mm (separate from self)")
    if contacts["environment"]:
        lines.append(f"OTHER ENVIRONMENT: {len(contacts['environment'])} pair(s)")
    return lines


def font(size, bold=False):
    from PIL import ImageFont
    name = "DejaVuSans-Bold.ttf" if bold else "DejaVuSans.ttf"
    for path in (name, str(Path("/usr/share/fonts/truetype/dejavu")/name), "Arial.ttf"):
        try:
            return ImageFont.truetype(path, size)
        except OSError:
            pass
    try:
        return ImageFont.load_default(size=size)
    except TypeError:
        return ImageFont.load_default()


def annotate(rgb, motion, frame, contacts, *, speed=1.):
    from PIL import Image, ImageDraw
    image = Image.fromarray(rgb).convert("RGBA")
    overlay = Image.new("RGBA", image.size)
    draw = ImageDraw.Draw(overlay)
    width, height = image.size
    scale = width/1280
    def rect(x0, y0, x1, y1):
        draw.rounded_rectangle(tuple(round(v*scale) for v in (x0, y0, x1, y1)), radius=round(12*scale), fill=(12, 22, 33, 238))
    def text(x, y, value, size=18, color=(220, 233, 243), bold=False):
        draw.text((round(x*scale), round(y*scale)), value, font=font(max(10, round(size*scale)), bold), fill=color)
    H = height/scale
    status, reasons = motion.status(frame)
    rect(20, 20, 1260, 150)
    text(40, 32, "Q1 / FRAX / LAFAN REFERENCE", 28, bold=True)
    text(40, 71, "KINEMATIC REPLAY - NOT A POLICY  |  "+status, 20,
         (122, 225, 191) if motion.kind == "sealed" and motion.dependency_identity_verified else (255, 183, 96), True)
    text(40, 110, f"{motion.path.stem}  |  frame {frame}/{len(motion.qpos)-1}  |  {frame/motion.fps:.3f} s  |  {motion.fps:g} Hz  |  {speed:g}x")
    rect(20, H-225, 1260, H-20)
    lines = contact_lines(contacts, limit=2)
    for j, line in enumerate(lines[:6]):
        text(40, H-211+j*25, line, 17 if line.startswith("  ") else 19,
             (255, 147, 114) if contacts["self"] and j < len(lines)-1 else (218, 232, 242), j == 0)
    note = "Rejected: "+", ".join(reasons) if reasons else "Physical geom contacts only; excluded body pairs are listed in the JSON report."
    if not motion.hash_verified:
        note = "RAW HASH UNVERIFIED. "+note
    if len(note) > 130:
        note = note[:127]+"..."
    text(40, H-48, note, 14, (177, 201, 218))
    return Image.alpha_composite(image, overlay).convert("RGB")
