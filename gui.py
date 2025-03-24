import shutil
import time
import streamlit as st
import plotly.graph_objects as go
import numpy as np
import os
import sys
import torch
import json
from argparse import Namespace

# Import functions from your project (adjust paths as needed)
from data_loaders.humanml.utils import paramUtil
from data_loaders.humanml.utils.plot_script import plot_3d_motion
from sample.generate import (
    construct_template_variables,
    load_dataset,
    save_multiple_samples,
)
from utils.fixseed import fixseed
from utils.parser_util import generate_args
from utils.model_util import create_model_and_diffusion, load_model_wo_clip
from utils import dist_util
from model.cfg_sampler import ClassifierFreeSampleModel
from data_loaders.get_data import get_dataset_loader
from data_loaders.humanml.scripts.motion_process import recover_from_ric
from data_loaders.tensors import collate
from os.path import join as pjoin

from utils.text_control_example import collate_all

torch.classes.__path__ = []

# --- Utility definitions ---
JOINTS = {
    "Pelvis (0)": 0,
    "Left Foot (10)": 10,
    "Right Foot (11)": 11,
    "Head (15)": 15,
    "Left Wrist (20)": 20,
    "Right Wrist (21)": 21,
}
TOTAL_JOINTS = 22

# Base colors for each joint (hex codes)
base_colors = {
    "Pelvis (0)": "#0000FF",  # blue
    "Left Foot (10)": "#008000",  # green
    "Right Foot (11)": "#FFA500",  # orange
    "Head (15)": "#800080",  # purple
    "Left Wrist (20)": "#A52A2A",  # brown
    "Right Wrist (21)": "#FFC0CB",  # pink
}

# Paths to normalization files – adjust if needed
RAW_MEAN_PATH = "./dataset/humanml_spatial_norm/Mean_raw.npy"
RAW_STD_PATH = "./dataset/humanml_spatial_norm/Std_raw.npy"

MODEL_PATH = "./save/omnicontrol_ckpt/model_humanml3d.pt"


def create_gradient_color(base_color, factor):
    """Lighten base_color by factor (0 = dark, 1 = white)."""
    base_color = base_color.lstrip("#")
    r = int(base_color[0:2], 16)
    g = int(base_color[2:4], 16)
    b = int(base_color[4:6], 16)
    r = int(r + (255 - r) * factor)
    g = int(g + (255 - g) * factor)
    b = int(b + (255 - b) * factor)
    return f"rgb({r},{g},{b})"


def create_advanced_3d_plot(current_x, current_y, current_z, constraints, total_frames):
    fig = go.Figure()
    # Plot the current unsaved point as a black marker.
    fig.add_trace(
        go.Scatter3d(
            x=[current_x],
            y=[current_y],
            z=[current_z],
            mode="markers",
            marker=dict(size=10, color="black"),
            name="Current Point",
        )
    )
    # Plot stored constraints with gradient colors.
    if constraints:
        groups = {}
        for frame, x, y, z, joint_id, joint_name in constraints:
            groups.setdefault(joint_name, []).append((frame, x, y, z))
        for joint, pts in groups.items():
            pts.sort(key=lambda item: item[0])
            base_color = base_colors.get(joint, "#000000")
            xs, ys, zs, colors = [], [], [], []
            for frame, x, y, z in pts:
                factor = min(max(frame / total_frames, 0), 1)
                colors.append(create_gradient_color(base_color, factor))
                xs.append(x)
                ys.append(y)
                zs.append(z)
            fig.add_trace(
                go.Scatter3d(
                    x=xs,
                    y=ys,
                    z=zs,
                    mode="markers",
                    marker=dict(size=8, color=colors),
                    name=f"Constraint: {joint}",
                )
            )
    fig.update_layout(
        scene=dict(
            xaxis=dict(range=[-5, 5], title="X"),
            yaxis=dict(range=[-5, 5], title="Y"),
            zaxis=dict(range=[-5, 5], title="Z"),
        ),
        title="3D Visualization of Constraints",
        height=500,
    )
    return fig


def generate_hint_array(total_frames, constraints):
    points_list = []
    for frame, x, y, z, joint_id, _ in constraints:
        if frame < total_frames:
            points_list.append(
                {"frame": frame, "joint_id": joint_id, "coords": (x, y, z)}
            )
    return points_list


def preprocess_hint(points_list, n_frames=196):
    """
    Convert a list of GUI-defined control points to the OmniControl hint format.

    Args:
        points_list: List of dictionaries, each containing:
                     - 'frame': the frame number
                     - 'joint_id': the joint index
                     - 'coords': the (x, y, z) global coordinates
        n_frames: Total number of frames in the sequence (default: 196)
        raw_mean: Mean values for normalization, numpy array of shape (22, 3)
        raw_std: Standard deviation values for normalization, numpy array of shape (22, 3)

    Returns:
        control_full: Normalized control hints in the format expected by OmniControl
    """

    raw_mean = np.load(RAW_MEAN_PATH)
    raw_std = np.load(RAW_STD_PATH)

    # Group points by joint_id for processing
    points_by_joint = {}
    for point in points_list:
        joint_id = point["joint_id"]
        if joint_id not in points_by_joint:
            points_by_joint[joint_id] = []
        points_by_joint[joint_id].append(
            {"frame": point["frame"], "coords": point["coords"]}
        )

    # Create control arrays for each joint
    joint_controls = {}
    for joint_id, points in points_by_joint.items():
        # Initialize empty control array for this joint
        control = np.zeros((n_frames, 3))

        # Fill in the specified points
        for point in points:
            frame = point["frame"]
            coords = point["coords"]
            control[frame] = coords

        joint_controls[joint_id] = control

    # Prepare the full control tensor (batch_size=1, n_frames, 22 joints, 3 coords)
    control_full = np.zeros((1, n_frames, 22, 3)).astype(np.float32)

    # Fill in controls for each joint and normalize
    for joint_id, control in joint_controls.items():
        # Create mask for non-zero entries (where constraints are specified)
        mask = (control.sum(-1) != 0)[..., np.newaxis]

        # Normalize the control using provided statistics
        control_normalized = (
            control - raw_mean.reshape(22, 1, 3)[joint_id]
        ) / raw_std.reshape(22, 1, 3)[joint_id]

        # Apply mask to keep only specified points
        control_normalized = control_normalized * mask

        # Insert into the full control tensor
        control_full[0, :, joint_id, :] = control_normalized

    # Reshape to match expected format (batch_size, n_frames, 22*3)
    control_full = control_full.reshape((1, n_frames, -1))

    return control_full


def visualize_motion(sample):
    total_frames, n_joints, _ = sample.shape
    fig = go.Figure()
    for j in range(n_joints):
        traj = sample[:, j, :]
        fig.add_trace(
            go.Scatter3d(
                x=traj[:, 0],
                y=traj[:, 1],
                z=traj[:, 2],
                mode="lines+markers",
                name=f"Joint {j}",
            )
        )
    fig.update_layout(
        scene=dict(
            xaxis=dict(range=[-5, 5], title="X"),
            yaxis=dict(range=[-5, 5], title="Y"),
            zaxis=dict(range=[-5, 5], title="Z"),
        ),
        title="Generated Motion Sample Trajectories",
        height=600,
    )
    return fig


def prepare_args(text_prompt_ui, num_reps, cond_mode_ui, sampler_ui, steps):
    # Override CLI defaults by temporarily setting sys.argv.
    old_argv = sys.argv.copy()
    sys.argv = ["gui.py", "--model_path", MODEL_PATH]
    sys.argv.append("--text_prompt")
    sys.argv.append(text_prompt_ui)
    sys.argv.append("--num_repetitions")
    sys.argv.append(str(num_reps))
    sys.argv.append("--output_dir")
    sys.argv.append("demo")
    sys.argv.append("--cond_mode")
    sys.argv.append(cond_mode_ui)

    if sampler_ui == "DDIM":
        sys.argv.append("--use_ddim")
        sys.argv.append("True")
        sys.argv.append("--timestep_respacing")
        sys.argv.append(f"ddim{steps}")
    elif sampler_ui == "DPM Order 2":
        sys.argv.append("--use_dpm")
        sys.argv.append("True")
        sys.argv.append("--dpm_order")
        sys.argv.append("2")
        sys.argv.append("--dpm_steps")
        sys.argv.append(str(steps))
    elif sampler_ui == "DPM Order 3":
        sys.argv.append("--use_dpm")
        sys.argv.append("True")
        sys.argv.append("--dpm_order")
        sys.argv.append("3")
        sys.argv.append("--dpm_steps")
        sys.argv.append(str(steps))

    args = generate_args()
    sys.argv = old_argv
    return args


def run_generation(args, processed_hint):
    fixseed(args.seed)
    out_path = args.output_dir
    max_frames = 196 if args.dataset in ["kit", "humanml"] else 60
    fps = 20
    n_frames = 196
    is_using_data = not any([args.text_prompt])
    dist_util.setup_dist(args.device)
    if out_path == "demo":
        out_path = os.path.join(
            os.path.dirname("./save/demo"), f"samples_seed{args.seed}"
        )
    else:
        raise ValueError("Invalid output directory for demo.")

    hints = None
    if args.text_prompt != "":
        if args.text_prompt == "predefined":
            texts, hints = collate_all(n_frames, args.dataset)
            args.num_samples = len(texts)
            if args.cond_mode == "only_spatial":
                texts = ["" for i in texts]
            elif args.cond_mode == "only_text":
                hints = None
        else:
            texts = [args.text_prompt]
            args.num_samples = 1
            print(processed_hint.shape)
            hints = np.concatenate([processed_hint], axis=0)
            print(hints.shape)

    assert (
        args.num_samples <= args.batch_size
    ), f"Please either increase batch_size({args.batch_size}) or reduce num_samples({args.num_samples})"
    args.batch_size = args.num_samples

    st.write("Loading dataset...")
    data = load_dataset(args, max_frames, n_frames)
    total_num_samples = args.num_samples * args.num_repetitions

    st.write("Creating model and diffusion...")
    model, diffusion = create_model_and_diffusion(args, data)
    st.write("Loading model checkpoint from", args.model_path)
    state_dict = torch.load(args.model_path, map_location="cpu")
    load_model_wo_clip(model, state_dict)

    if args.guidance_param != 1:
        model = ClassifierFreeSampleModel(model)
    model.to(dist_util.dev())
    model.eval()

    if is_using_data:
        iterator = iter(data)
        _, model_kwargs = next(iterator)
    else:
        collate_args = [
            {"inp": torch.zeros(n_frames), "tokens": None, "lengths": n_frames}
        ] * args.num_samples
        collate_args = [dict(a, text=txt) for a, txt in zip(collate_args, texts)]
        if hints is not None:
            collate_args = [dict(a, hint=hint) for a, hint in zip(collate_args, hints)]
        _, model_kwargs = collate(collate_args)

    for k, v in model_kwargs["y"].items():
        if torch.is_tensor(v):
            model_kwargs["y"][k] = v.to(dist_util.dev())

    all_motions = []
    all_lengths = []
    all_text = []
    all_hint = []
    all_hint_for_vis = []

    # mean inference times for a sample in a batch in each repetition
    rep_infer_times = []

    for rep_i in range(args.num_repetitions):
        print(f"### Sampling [repetitions #{rep_i}]")

        # add CFG scale to batch
        if args.guidance_param != 1:
            model_kwargs["y"]["scale"] = (
                torch.ones(args.batch_size, device=dist_util.dev())
                * args.guidance_param
            )

        assert not (args.use_ddim and args.use_dpm), "Choose one of the two"

        if args.use_ddim:
            print(f"Using DDIM with timestep respacing: {args.timestep_respacing}")
            sample_fn = diffusion.ddim_sample_loop
        elif args.use_dpm:
            print(f"Using DPM Solver with order: {args.dpm_order}")
            sample_fn = diffusion.dpm_solver_sample_loop
        else:
            print("Using default sampling method")
            sample_fn = diffusion.p_sample_loop

        if args.use_ddim:
            out_path += f"_{args.timestep_respacing}"

        batch_infer_start = time.time()
        sample = sample_fn(
            model,
            (args.batch_size, model.njoints, model.nfeats, max_frames),
            clip_denoised=False,
            model_kwargs=model_kwargs,
            skip_timesteps=0,
            init_image=None,
            progress=True,
            dump_steps=None,
            noise=None,
            const_noise=False,
            steps=args.dpm_steps,
            order=args.dpm_order,
        )
        batch_infer_time = time.time() - batch_infer_start
        st.write(f"Batch inference time: {batch_infer_time:.3f} seconds")
        rep_infer_times = [batch_infer_time / args.batch_size]

        sample = sample[:, :263]  # for HumanML3D, D = 263
        if model.data_rep == "hml_vec":
            n_joints = 22 if sample.shape[0] == 263 else 21
            sample = data.dataset.t2m_dataset.inv_transform(
                sample.cpu().permute(0, 2, 3, 1)
            ).float()
            sample = recover_from_ric(sample, n_joints)
            sample = sample.view(-1, *sample.shape[2:]).permute(0, 2, 3, 1)

        rot2xyz_pose_rep = (
            "xyz" if model.data_rep in ["xyz", "hml_vec"] else model.data_rep
        )
        rot2xyz_mask = (
            None
            if rot2xyz_pose_rep == "xyz"
            else model_kwargs["y"]["mask"].reshape(args.batch_size, n_frames).bool()
        )
        sample = model.rot2xyz(
            x=sample,
            mask=rot2xyz_mask,
            pose_rep=rot2xyz_pose_rep,
            glob=True,
            translation=True,
            jointstype="smpl",
            vertstrans=True,
            betas=None,
            beta=0,
            glob_rot=None,
            get_rotations_back=False,
        )

        if args.unconstrained:
            all_text += ["unconstrained"] * args.num_samples
        else:
            text_key = "text" if "text" in model_kwargs["y"] else "action_text"
            all_text += model_kwargs["y"][text_key]

            if "hint" in model_kwargs["y"]:
                hint = model_kwargs["y"]["hint"]
                # denormalize hint
                if args.dataset == "humanml":
                    spatial_norm_path = "./dataset/humanml_spatial_norm"
                elif args.dataset == "kit":
                    spatial_norm_path = "./dataset/kit_spatial_norm"
                else:
                    raise NotImplementedError("unknown dataset")
                raw_mean = torch.from_numpy(
                    np.load(pjoin(spatial_norm_path, "Mean_raw.npy"))
                ).cuda()
                raw_std = torch.from_numpy(
                    np.load(pjoin(spatial_norm_path, "Std_raw.npy"))
                ).cuda()
                mask = hint.view(hint.shape[0], hint.shape[1], n_joints, 3).sum(-1) != 0
                hint = hint * raw_std + raw_mean
                hint = hint.view(
                    hint.shape[0], hint.shape[1], n_joints, 3
                ) * mask.unsqueeze(-1)
                hint = hint.view(hint.shape[0], hint.shape[1], -1)
                # ---
                all_hint.append(hint.data.cpu().numpy())
                hint = hint.view(hint.shape[0], hint.shape[1], n_joints, 3)
                all_hint_for_vis.append(hint.data.cpu().numpy())

        all_motions.append(sample.cpu().numpy())
        all_lengths.append(model_kwargs["y"]["lengths"].cpu().numpy())

        print(f"created {len(all_motions) * args.batch_size} samples")

    print(f"Average inference time per sample: {np.mean(rep_infer_times):.3f} seconds")

    all_motions = np.concatenate(all_motions, axis=0)
    all_motions = all_motions[:total_num_samples]  # [bs, njoints, 6, seqlen]
    all_text = all_text[:total_num_samples]
    all_lengths = np.concatenate(all_lengths, axis=0)[:total_num_samples]
    if "hint" in model_kwargs["y"]:
        all_hint = np.concatenate(all_hint, axis=0)[:total_num_samples]
        all_hint_for_vis = np.concatenate(all_hint_for_vis, axis=0)[:total_num_samples]

    if len(all_hint) != 0:
        from utils.simple_eval import simple_eval

        results = simple_eval(all_motions, all_hint, n_joints)
        print(results)

    if os.path.exists(out_path):
        shutil.rmtree(out_path)
    os.makedirs(out_path)

    npy_path = os.path.join(out_path, "results.npy")
    print(f"saving results file to [{npy_path}]")
    np.save(
        npy_path,
        {
            "motion": all_motions,
            "text": all_text,
            "lengths": all_lengths,
            "hint": all_hint_for_vis,
            "num_samples": args.num_samples,
            "num_repetitions": args.num_repetitions,
        },
    )
    with open(npy_path.replace(".npy", ".txt"), "w") as fw:
        fw.write("\n".join(all_text))
    with open(npy_path.replace(".npy", "_len.txt"), "w") as fw:
        fw.write("\n".join([str(l) for l in all_lengths]))

    print(f"saving visualizations to [{out_path}]...")
    skeleton = (
        paramUtil.kit_kinematic_chain
        if args.dataset == "kit"
        else paramUtil.t2m_kinematic_chain
    )

    sample_files = []
    num_samples_in_out_file = 7

    (
        sample_print_template,
        row_print_template,
        all_print_template,
        sample_file_template,
        row_file_template,
        all_file_template,
    ) = construct_template_variables(args.unconstrained)

    for sample_i in range(args.num_samples):
        rep_files = []
        for rep_i in range(args.num_repetitions):
            caption = all_text[rep_i * args.batch_size + sample_i]
            length = all_lengths[rep_i * args.batch_size + sample_i]
            motion = all_motions[rep_i * args.batch_size + sample_i].transpose(2, 0, 1)[
                :length
            ]
            if "hint" in model_kwargs["y"]:
                hint = all_hint_for_vis[rep_i * args.batch_size + sample_i]
            else:
                hint = None
            save_file = sample_file_template.format(sample_i, rep_i)
            print(sample_print_template.format(caption, sample_i, rep_i, save_file))
            animation_save_path = os.path.join(out_path, save_file)
            plot_3d_motion(
                animation_save_path,
                skeleton,
                motion,
                dataset=args.dataset,
                title=caption,
                fps=fps,
                hint=hint,
            )
            # Credit for visualization: https://github.com/EricGuo5513/text-to-motion
            rep_files.append(animation_save_path)

        sample_files = save_multiple_samples(
            args,
            out_path,
            row_print_template,
            all_print_template,
            row_file_template,
            all_file_template,
            caption,
            num_samples_in_out_file,
            rep_files,
            sample_files,
            sample_i,
        )

    abs_path = os.path.abspath(out_path)
    print(f"[Done] Results are at [{abs_path}]")


# --- Streamlit GUI ---

st.title("OmniControl Motion Generator")

st.markdown("### Generation Parameters")
text_prompt_ui = st.text_input("Text Prompt", value="predefined")
col1, col2 = st.columns(2)
with col1:
    cond_mode_ui = st.selectbox(
        "Conditioning Mode", options=["both_text_spatial", "only_spatial", "only_text"]
    )
with col2:
    num_reps = st.number_input(
        "Number of Samples", min_value=1, max_value=5, value=3, step=1
    )
col3, col4 = st.columns(2)
with col3:
    sampler_ui = st.selectbox(
        "Sampler", options=["DDPM", "DDIM", "DPM Order 2", "DPM Order 3"]
    )
with col4:
    steps_ui = st.number_input("Steps", min_value=1, value=20, step=1)


st.markdown("### Spatial Constraints")
st.markdown("Adjust the sliders and add constraints:")
col1, col2 = st.columns(2)
with col1:
    constraint_frame = st.number_input("Constraint Frame", min_value=0, value=0, step=1)
with col2:
    joint_selection = st.selectbox("Select Joint", list(JOINTS.keys()))
# Place x/y/z sliders in a single row.
col1, col2, col3 = st.columns(3)
with col1:
    x_coord = st.slider("X", -5.0, 5.0, 0.0, step=0.1)
with col2:
    y_coord = st.slider("Y", -5.0, 5.0, 0.0, step=0.1)
with col3:
    z_coord = st.slider("Z", -5.0, 5.0, 0.0, step=0.1)

if "constraints" not in st.session_state:
    st.session_state.constraints = []

if st.button("Add Constraint"):
    try:
        frame = int(constraint_frame)
    except:
        st.error("Invalid frame number.")
    if frame < 0:
        st.error("Frame must be non-negative.")
    else:
        exists = any(c[0] == frame for c in st.session_state.constraints)
        if exists:
            st.error(f"A constraint for frame {frame} already exists.")
        else:
            joint_id = JOINTS[joint_selection]
            new_const = (frame, x_coord, y_coord, z_coord, joint_id, joint_selection)
            st.session_state.constraints.append(new_const)
            st.success(
                f"Added: Frame {frame}, Joint {joint_selection}: ({x_coord}, {y_coord}, {z_coord})"
            )

total_frames = 196
raw_hint = generate_hint_array(total_frames, st.session_state.constraints)
viz_fig = create_advanced_3d_plot(
    x_coord, y_coord, z_coord, st.session_state.constraints, total_frames
)

st.plotly_chart(viz_fig, use_container_width=True)

st.subheader("Stored Constraints:")
if st.session_state.constraints:
    for idx, c in enumerate(st.session_state.constraints):
        constraint_str = (
            f"Frame {c[0]}, Joint {c[5]}: ({c[1]:.2f}, {c[2]:.2f}, {c[3]:.2f})"
        )
        col1, col2 = st.columns([0.8, 0.2])
        with col1:
            st.write(constraint_str)
        with col2:
            if st.button("Remove", key=f"remove_{idx}"):
                st.session_state.constraints.pop(idx)
                st.experimental_rerun()
else:
    st.write("No constraints yet.")

st.markdown("### Generate Motion Sample")
if st.button("Generate Motion"):
    processed_hint = preprocess_hint(raw_hint)
    st.write("Processed hint array shape:", processed_hint.shape)
    args = prepare_args(text_prompt_ui, num_reps, cond_mode_ui, sampler_ui, steps_ui)

    if args is not None:
        st.write("Starting generation (this may take a while)...")
        sample_motion = run_generation(args, processed_hint)
        st.success("Motion sample generated!")
        # motion_fig = visualize_motion(sample_motion)
        # st.plotly_chart(motion_fig, use_container_width=True)
