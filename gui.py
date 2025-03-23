import streamlit as st
import plotly.graph_objects as go
import numpy as np
import os
import torch

# Import functions from your project (adjust paths as needed)
from utils.fixseed import fixseed
from utils.parser_util import generate_args
from utils.model_util import create_model_and_diffusion, load_model_wo_clip
from utils import dist_util
from model.cfg_sampler import ClassifierFreeSampleModel
from data_loaders.get_data import get_dataset_loader
from data_loaders.humanml.scripts.motion_process import recover_from_ric
from data_loaders.tensors import collate
from os.path import join as pjoin

# --- Utility functions for the GUI ---
JOINTS = {
    "Pelvis (0)": 0,
    "Left Foot (10)": 10,
    "Right Foot (11)": 11,
    "Head (15)": 15,
    "Left Wrist (20)": 20,
    "Right Wrist (21)": 21,
}
TOTAL_JOINTS = 22

# Paths to normalization files – adjust if needed
RAW_MEAN_PATH = "./dataset/humanml_spatial_norm/Mean_raw.npy"
RAW_STD_PATH = "./dataset/humanml_spatial_norm/Std_raw.npy"


def create_3d_constraints_plot(current_hint, constraints):
    fig = go.Figure()
    # For visualization, show the current hint for frame 0
    frame_vis = 0
    for j in range(TOTAL_JOINTS):
        x = current_hint[frame_vis, j, 0]
        y = current_hint[frame_vis, j, 1]
        z = current_hint[frame_vis, j, 2]
        fig.add_trace(
            go.Scatter3d(
                x=[x],
                y=[y],
                z=[z],
                mode="markers",
                marker=dict(size=8, color="red"),
                name=f"Current Hint Joint {j}",
            )
        )
    if constraints:
        groups = {}
        for frame, x, y, z, joint_id, joint_name in constraints:
            groups.setdefault(joint_name, []).append((x, y, z))
        markers = ["circle", "square", "diamond", "cross", "x", "triangle-up"]
        colors = ["blue", "green", "orange", "purple", "brown", "pink"]
        for i, (joint, pts) in enumerate(groups.items()):
            xs = [pt[0] for pt in pts]
            ys = [pt[1] for pt in pts]
            zs = [pt[2] for pt in pts]
            fig.add_trace(
                go.Scatter3d(
                    x=xs,
                    y=ys,
                    z=zs,
                    mode="markers",
                    marker=dict(
                        size=6,
                        symbol=markers[i % len(markers)],
                        color=colors[i % len(colors)],
                    ),
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
    hint = np.zeros((total_frames, TOTAL_JOINTS, 3), dtype=np.float32)
    for frame, x, y, z, joint_id, _ in constraints:
        if frame < total_frames:
            hint[frame, joint_id, :] = np.array([x, y, z], dtype=np.float32)
    return hint


def preprocess_hint(hint):
    """
    Normalize hint values using raw_mean and raw_std.
    For each joint, if a constraint exists (nonzero), apply:
      normalized = (hint - raw_mean) / raw_std.
    Then flatten to shape (total_frames, TOTAL_JOINTS*3).
    """
    if not os.path.exists(RAW_MEAN_PATH) or not os.path.exists(RAW_STD_PATH):
        st.error("Normalization files not found!")
        return hint
    raw_mean = np.load(RAW_MEAN_PATH)
    raw_std = np.load(RAW_STD_PATH)
    if raw_mean.ndim == 1:
        raw_mean = raw_mean.reshape(TOTAL_JOINTS, 3)
    if raw_std.ndim == 1:
        raw_std = raw_std.reshape(TOTAL_JOINTS, 3)
    norm_hint = hint.copy()
    total_frames = norm_hint.shape[0]
    for j in range(TOTAL_JOINTS):
        mask = np.any(hint[:, j, :] != 0, axis=1, keepdims=True)
        norm_hint[:, j, :] = ((hint[:, j, :] - raw_mean[j]) / raw_std[j]) * mask
    norm_hint_flat = norm_hint.reshape(total_frames, -1)
    return norm_hint_flat


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


# --- Generation Pipeline (based on generate.py) ---
def load_dataset(args, max_frames, n_frames):
    data = get_dataset_loader(
        name=args.dataset,
        batch_size=args.batch_size,
        num_frames=n_frames,
        split="test",
        hml_mode="train",
    )
    if args.dataset in ["kit", "humanml"]:
        data.dataset.t2m_dataset.fixed_length = n_frames
    return data


def run_generation(args, processed_hint):
    # Mimic generate.py: set seed, load model/diffusion, prepare hints, sample, and postprocess.
    fixseed(args.seed)
    max_frames = 196 if args.dataset in ["kit", "humanml"] else 60
    # For simplicity, force n_frames to total_frames provided by UI
    n_frames = int(args.motion_length * (12.5 if args.dataset == "kit" else 20))
    n_frames = int(total_frames)  # override with UI total_frames
    dist_util.setup_dist(args.device)

    # Set output directory (unused here)
    if args.output_dir == "":
        args.output_dir = "./generated_samples"

    # We force using our hint, so texts/hints from collate_all are ignored.
    args.num_samples = 1
    args.batch_size = 1

    data = load_dataset(args, max_frames, n_frames)
    model, diffusion = create_model_and_diffusion(args, data)
    st.write("Loading model checkpoint from", args.model_path)
    state_dict = torch.load(args.model_path, map_location="cpu")
    load_model_wo_clip(model, state_dict)
    if args.guidance_param != 1:
        model = ClassifierFreeSampleModel(model)
    model.to(dist_util.dev())
    model.eval()

    # Prepare dummy collate_args (we use our text prompt and processed hint)
    collate_args = [{"inp": torch.zeros(n_frames), "tokens": None, "lengths": n_frames}]
    collate_args = [dict(arg, text=args.text_prompt) for arg in collate_args]
    # Convert processed_hint (shape (n_frames, TOTAL_JOINTS*3)) to tensor and add batch dimension.
    hint_tensor = torch.tensor(processed_hint).unsqueeze(0).float().to(dist_util.dev())
    collate_args = [dict(arg, hint=hint_tensor)]
    _, model_kwargs = collate(collate_args)
    for k, v in model_kwargs["y"].items():
        if torch.is_tensor(v):
            model_kwargs["y"][k] = v.to(dist_util.dev())

    # Choose sampling function (using default p_sample_loop here)
    sample_fn = diffusion.p_sample_loop

    st.write("Sampling motion...")
    sample = sample_fn(
        model,
        (args.batch_size, model.njoints, model.nfeats, max_frames),
        clip_denoised=False,
        model_kwargs=model_kwargs,
        skip_timesteps=0,
        init_image=None,
        progress=False,
        dump_steps=None,
        noise=None,
        const_noise=False,
    )
    # Assume sample shape is (1, njoints, nfeats, max_frames); select first sample
    sample = sample[0]
    sample = sample[:, :263]  # for HumanML3D, D = 263
    # Recover XYZ positions
    n_joints = 22 if sample.shape[0] == 263 else 21
    sample = data.dataset.t2m_dataset.inv_transform(
        sample.cpu().permute(1, 2, 0)
    ).float()
    sample = recover_from_ric(sample, n_joints)
    sample = sample.view(-1, *sample.shape[2:]).permute(0, 2, 3, 1)
    rot2xyz_pose_rep = "xyz" if model.data_rep in ["xyz", "hml_vec"] else model.data_rep
    if rot2xyz_pose_rep != "xyz":
        rot2xyz_mask = (
            model_kwargs["y"]["mask"].reshape(args.batch_size, n_frames).bool()
        )
    else:
        rot2xyz_mask = None
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
    return sample.cpu().numpy()


# --- Streamlit GUI ---
st.title("OmniControl Motion Generator")

st.markdown("### Generation Parameters")
text_prompt_ui = st.text_input("Text Prompt", value="predefined")
total_frames = st.number_input("Total Frames", min_value=1, value=120, step=1)

st.markdown("### Spatial Constraints")
st.markdown("Adjust the sliders and add constraints:")
constraint_frame = st.number_input("Constraint Frame", min_value=0, value=0, step=1)
joint_selection = st.selectbox("Select Joint", list(JOINTS.keys()))
x_coord = st.slider("X Coordinate", -5.0, 5.0, 0.0, step=0.1)
y_coord = st.slider("Y Coordinate", -5.0, 5.0, 0.0, step=0.1)
z_coord = st.slider("Z Coordinate", -5.0, 5.0, 0.0, step=0.1)

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
        joint_id = JOINTS[joint_selection]
        new_const = (frame, x_coord, y_coord, z_coord, joint_id, joint_selection)
        st.session_state.constraints.append(new_const)
        st.success(
            f"Added: Frame {frame}, Joint {joint_selection}: ({x_coord}, {y_coord}, {z_coord})"
        )

st.subheader("Stored Constraints:")
if st.session_state.constraints:
    for c in st.session_state.constraints:
        st.write(f"Frame {c[0]}, Joint {c[5]}: ({c[1]:.2f}, {c[2]:.2f}, {c[3]:.2f})")
else:
    st.write("No constraints yet.")

# Live visualization of constraints.
st.markdown("### Live 3D Visualization of Constraints")
raw_hint = generate_hint_array(int(total_frames), st.session_state.constraints)
viz_fig = create_3d_constraints_plot(raw_hint, st.session_state.constraints)
st.plotly_chart(viz_fig, use_container_width=True)

st.markdown("### Generate Motion Sample")
# Button to generate motion using the generation pipeline.
if st.button("Generate Motion"):
    # Preprocess the raw hint array using normalization.
    processed_hint = preprocess_hint(raw_hint)
    # Set up args using generate_args, then override necessary fields.
    args = generate_args()
    args.text_prompt = text_prompt_ui  # we set to "predefined" to use spatial hints
    args.dataset = "humanml"  # adjust if needed
    # Provide the model checkpoint path (update as needed)
    args.model_path = st.text_input("Model Path", value="./model_checkpoint/model.pt")
    args.seed = 42
    args.guidance_param = 1.0
    args.decompose_prompt = False
    args.output_dir = "./generated_samples"
    # Set motion_length so that n_frames equals total_frames. For HumanML3D, fps=20.
    args.motion_length = total_frames / 20.0
    args.batch_size = 1
    st.write("Starting generation (this may take a while)...")
    sample_motion = run_generation(args, processed_hint)
    st.success("Motion sample generated!")
    motion_fig = visualize_motion(sample_motion)
    st.plotly_chart(motion_fig, use_container_width=True)
