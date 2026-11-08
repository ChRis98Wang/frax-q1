#!/usr/bin/env python3
"""Render clean, real LAFAN retargeted motion with a synchronized second view.

No poses are synthesized, smoothed, grounded, repaired or interpolated here.
Selection ranges are half-open original source-frame indices. Final output
requires matching independent QA reports and verified motion/model assets.
Use --preview only for local visual selection before final validation.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import sys


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset', type=Path, required=True)
    parser.add_argument('--source-dir', type=Path, required=True)
    parser.add_argument('--model', type=Path, required=True)
    parser.add_argument('--selection', type=Path, required=True,
                        help='JSON with segments[{name,start_frame,end_frame,quality_report,title?,azimuth?}]')
    parser.add_argument('--output-dir', type=Path, default=Path('docs/assets'))
    parser.add_argument('--artifacts', type=Path, default=Path('artifacts/lafan_dynamic/final'))
    parser.add_argument('--preview', action='store_true', help='Local visual inspection; never publish as validated output')
    args = parser.parse_args(argv)
    if args.preview and args.output_dir.resolve().is_relative_to((Path(__file__).resolve().parents[1]/'docs').resolve()):
        raise ValueError('Preview media must remain outside public docs/')
    if sys.platform.startswith('linux'):
        os.environ.setdefault('MUJOCO_GL', 'egl')
    import imageio.v2 as imageio
    import mujoco
    import numpy as np
    from PIL import Image, ImageDraw
    from q1_frax.motion_io import guard_output_paths, load_motion, verify_motion_dependencies
    from q1_frax.paths import SOURCES
    from q1_frax.render import load_model
    from q1_frax.visuals import font, make_render_model, physical_contacts

    selection = json.loads(args.selection.read_text())
    source_manifest = json.loads((args.dataset/'manifest.json').read_text())
    segments = selection['segments']
    if not segments:
        raise ValueError('At least one original source segment is required')
    width, height, fps = 1280, 720, 30
    loaded, qa_reports, source_records = [], [], []
    video = args.output_dir/'lafan_q1_dynamic.mp4'
    preview = args.output_dir/'lafan_q1_dynamic.gif'
    meta_path = args.output_dir/'lafan_demo.json'
    qa_path = args.output_dir/'lafan_quality.json'
    for row in segments:
        name = row['name']; start, end = row['start_frame'], row['end_frame']
        if not isinstance(name, str) or Path(name).name != name:
            raise ValueError('Unsafe source clip name')
        if type(start) is not int or type(end) is not int or not 0 <= start < end:
            raise ValueError('Invalid source-frame range')
        if row.get('speed', 1) != 1:
            raise ValueError('Main demo keeps original 1x speed; no hidden time warping')
        motion = load_motion(args.dataset/'npz'/(name+'.npz'), model_path=args.model)
        if motion.fps != fps or end > len(motion.qpos):
            raise ValueError('Frame range or 30 Hz source convention differs')
        if not args.preview and (not motion.hash_verified or not motion.dependency_identity_verified):
            raise ValueError('Final media requires complete reference and model dependency identity')
        guard_output_paths(motion, args.model, video, preview, meta_path, qa_path)
        bvh = args.source_dir/(name+'.bvh'); bvh_hash = digest(bvh)
        declared = next(c for c in source_manifest['clips'] if c['name'] == name)
        if declared.get('source_sha256', bvh_hash) != bvh_hash:
            raise ValueError('Original BVH hash differs from retarget provenance')
        qa = None
        if not args.preview:
            qa_file = Path(row['quality_report'])
            qa = json.loads(qa_file.read_text())
            if qa.get('source_sha256') != motion.sha256 or qa.get('model_sha256') != motion.model_sha256:
                raise ValueError('Independent QA refers to a different motion/model')
            if qa.get('source_range_half_open') != [start, end]:
                raise ValueError('Independent QA refers to a different selected range')
            if not qa.get('sampled_physical_and_continuity_passed') or not all(qa.get('gates', {}).values()):
                raise ValueError('Selected segment failed independent QA')
            if not qa.get('requested_checks_passed') or not qa.get('source_tracking_verified') or not qa.get('source_and_sphere_gates_passed'):
                raise ValueError('Final media requires fresh source tracking, sphere and visual-envelope checks')
            if not qa.get('model_dependency_identity_verified'):
                raise ValueError('Independent QA did not verify model dependencies')
            if qa.get('model_dependencies') != motion.model_dependencies:
                raise ValueError('Independent QA model dependency inventory differs')
            qa_reports.append({'name': name, 'report_sha256': digest(qa_file), 'report': qa})
        loaded.append((row, motion, motion.qpos[start:end]))
        source_records.append({'name': name, 'source_bvh_sha256': bvh_hash,
                               'retarget_npz_sha256': motion.sha256, 'start_frame': start,
                               'end_frame': end, 'fps': fps, 'speed': 1,
                               'duration_seconds': (end-start)/fps})
    physical = load_model(args.model); physical_data = mujoco.MjData(physical)
    render = make_render_model(physical); data = mujoco.MjData(render)
    render.vis.global_.offwidth, render.vis.global_.offheight = width, height
    option = mujoco.MjvOption(); option.geomgroup[4] = 0
    # Keep every collider invisible; footage is a visual result, not a contact HUD.
    option.geomgroup[3] = 0
    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.artifacts.mkdir(parents=True, exist_ok=True)
    gif_frames, rendered_contacts = [], []
    output_index = 0
    total_frames = sum(len(q) for _, _, q in loaded)
    with mujoco.Renderer(render, width=width, height=height) as renderer:
        with imageio.get_writer(video, fps=fps, codec='libx264', quality=8,
                                macro_block_size=1, ffmpeg_params=['-movflags', '+faststart']) as writer:
            for segment_number, (row, motion, poses) in enumerate(loaded):
                start = row['start_frame']; count = len(poses)
                # Fixed camera height preserves takeoff/landing height. Only XY follows.
                top = 0.
                for q in poses:
                    physical_data.qpos[:] = q; mujoco.mj_forward(physical, physical_data)
                    top = max(top, float(physical_data.xpos[1:, 2].max()))
                center_z = top*.49
                distance = max(1.48, top*1.70)
                w, x, y, z = poses[0, 3:7]
                heading = math.degrees(math.atan2(2*(w*z+x*y), 1-2*(y*y+z*z)))
                azimuth = heading+float(row.get('azimuth', 135.))
                capture = {0, count//4, count//2, 3*count//4, count-1}
                for local, q in enumerate(poses):
                    physical_data.qpos[:] = q; mujoco.mj_forward(physical, physical_data)
                    contacts = physical_contacts(physical, physical_data)
                    maximum = {k: max((c['penetration_mm'] for c in v), default=0.) for k, v in contacts.items()}
                    if not args.preview and (maximum['self'] > 0. or maximum['floor'] > 0.):
                        raise ValueError('Unexpected collision while rendering unchanged validated poses')
                    rendered_contacts.append(maximum)
                    data.qpos[:] = q; data.qvel[:] = 0.; data.time = (start+local)/fps
                    mujoco.mj_forward(render, data)
                    camera = mujoco.MjvCamera(); camera.distance = distance
                    camera.elevation = -13.; camera.azimuth = azimuth
                    camera.lookat[:] = [q[0], q[1], center_z]
                    renderer.update_scene(data, camera=camera, scene_option=option)
                    frame = Image.fromarray(renderer.render().copy()).convert('RGB')
                    camera.azimuth = azimuth+90.
                    renderer.update_scene(data, camera=camera, scene_option=option)
                    inset = Image.fromarray(renderer.render().copy()).resize((352, 198), Image.Resampling.LANCZOS)
                    # Small synchronized alternate angle; the main image stays unobstructed.
                    frame.paste(inset, (width-370, 20))
                    draw = ImageDraw.Draw(frame)
                    draw.rounded_rectangle((width-372, 18, width-16, 220), radius=5, outline=(181, 203, 216), width=2)
                    draw.text((width-360, 27), 'SECOND VIEW', font=font(12, True), fill=(236, 243, 248), stroke_width=1, stroke_fill=(20, 30, 40))
                    draw.text((24, 24), 'Q1 / LAFAN', font=font(25, True), fill=(239, 245, 249), stroke_width=1, stroke_fill=(35, 48, 61))
                    label = f"{row.get('title', row['name'])}  /  {row['name']}  /  1x"
                    draw.rounded_rectangle((18, height-77, 740, height-15), radius=9, fill=(14, 25, 36))
                    draw.text((31, height-69), label, font=font(18, True), fill=(229, 238, 245))
                    caption = 'LAFAN retargeted kinematic reference / not a trained policy'
                    if args.preview: caption += ' / preview'
                    draw.text((31, height-42), caption, font=font(13), fill=(178, 204, 220))
                    writer.append_data(np.asarray(frame))
                    if output_index % 2 == 0:
                        gif_frames.append(frame.resize((640, 360), Image.Resampling.LANCZOS))
                    if local in capture:
                        frame.save(args.artifacts/f"{segment_number:02d}_{row['name']}_{start+local:05d}.png")
                    output_index += 1
    duration = total_frames/fps
    palette = [f.quantize(colors=96, method=Image.Quantize.FASTOCTREE) for f in gif_frames]
    # Exact full-clip playback duration despite GIF's centisecond time grid.
    durations = [10*(round((i+1)*duration*100/len(palette))-round(i*duration*100/len(palette))) for i in range(len(palette))]
    palette[0].save(preview, save_all=True, append_images=palette[1:], duration=durations,
                    loop=0, optimize=True, disposal=1)
    with Image.open(preview) as gif:
        encoded_gif_frames = gif.n_frames
        encoded_gif_duration_ms = 0
        for index in range(gif.n_frames):
            gif.seek(index)
            encoded_gif_duration_ms += gif.info['duration']
    if abs(encoded_gif_duration_ms/1000.-duration) > .011:
        raise ValueError('GIF encoded duration differs from source playback')
    if video.stat().st_size > 20_000_000 or preview.stat().st_size > 12_000_000:
        raise ValueError('Published media exceeds size budget')
    for _, motion, _ in loaded:
        verify_motion_dependencies(motion, args.model)
        if digest(motion.path) != motion.sha256:
            raise ValueError('Motion changed during rendering')
    if digest(args.model) != loaded[0][1].model_sha256:
        raise ValueError('Model changed during rendering')
    qa_public = {'schema_version': 1, 'preview': args.preview, 'segments': qa_reports,
                 'source_verified': not args.preview and all(r['report'].get('source_tracking_verified') for r in qa_reports),
                 'render_frame_count': output_index,
                 'maximum_rendered_penetration_mm': {k: max(c[k] for c in rendered_contacts) for k in ('self', 'floor')},
                 'pose_edits': False, 'interpolation_or_repeated_padding': False}
    qa_path.write_text(json.dumps(qa_public, indent=2, allow_nan=False)+'\n')
    meta = {'schema_version': 1, 'origin': 'lafan_retargeted', 'uses_lafan': True,
            'preview': args.preview, 'generator': 'tools/make_lafan_demo.py', 'generator_sha256': digest(__file__),
            'source_verified': qa_public['source_verified'],
            'files': {p.name: {'sha256': digest(p), 'size_bytes': p.stat().st_size} for p in (video, preview)},
            'source_clips': source_records, 'fps': fps, 'frames': output_index,
            'width': width, 'height': height, 'duration_seconds': duration, 'audio': False,
            'gif': {'nominal_fps': 15, 'frames': encoded_gif_frames, 'width': 640, 'height': 360,
                    'duration_seconds': encoded_gif_duration_ms/1000.},
            'source_quaternion': 'wxyz', 'pose_processing': 'None; original archived qpos, exact source ordering, 1x speed, no repeated padding',
            'model_sha256': loaded[0][1].model_sha256, 'model_dependencies': loaded[0][1].model_dependencies,
            'quality_report': qa_path.name, 'quality_report_sha256': digest(qa_path),
            'retarget_provenance': {k: source_manifest.get(k) for k in ('commits', 'mapping_sha256', 'collision_sha256', 'retarget_parameters', 'retarget_adapter_sha256')},
            'source_credits': {'LAFAN1': {'repository': SOURCES['lafan1_source']['url'], 'revision': SOURCES['lafan1_source']['commit'], 'license': 'CC-BY-NC-ND-4.0'},
                               'Q1': {'repository': SOURCES['primebot-q1-urdf']['url'], 'revision': SOURCES['primebot-q1-urdf']['commit']}},
            'not_a_trained_policy': True, 'not_a_dynamics_feasibility_certificate': True}
    serialized = json.dumps(meta, indent=2, allow_nan=False)+'\n'
    if '/home/' in serialized or '/tmp/' in serialized:
        raise ValueError('Public provenance contains a private path')
    meta_path.write_text(serialized)
    reader = imageio.get_reader(video)
    assert reader.count_frames() == total_frames
    assert abs(reader.get_meta_data()['duration']-duration) <= 1/fps
    reader.close()
    print(json.dumps({'video': str(video), 'gif': str(preview), 'frames': total_frames,
                      'duration_seconds': duration, 'sizes_bytes': meta['files']}, indent=2))


if __name__ == '__main__':
    main()
