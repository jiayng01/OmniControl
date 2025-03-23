import streamlit as st
import plotly.graph_objects as go
import numpy as np

# Define joints dictionary and total joints constant.
JOINTS = {
    "Pelvis (0)": 0,
    "Left Foot (10)": 10,
    "Right Foot (11)": 11,
    "Head (15)": 15,
    "Left Wrist (20)": 20,
    "Right Wrist (21)": 21,
}
TOTAL_JOINTS = 22

st.title("OmniControl 3D Constraint Picker")

# Inputs: text prompt, total frames, constraint frame, joint selection.
text_prompt = st.text_input("Text Prompt")
total_frames = st.number_input("Total Frames", min_value=1, value=120, step=1)
constraint_frame = st.number_input("Constraint Frame", min_value=0, value=0, step=1)
joint_selection = st.selectbox("Select Joint", list(JOINTS.keys()))

st.markdown("### Adjust the point coordinates (starting at the origin)")

# Sliders for x, y, z.
x_val = st.slider("X Coordinate", -5.0, 5.0, 0.0, step=0.1)
y_val = st.slider("Y Coordinate", -5.0, 5.0, 0.0, step=0.1)
z_val = st.slider("Z Coordinate", -5.0, 5.0, 0.0, step=0.1)


def create_3d_plot(current_x, current_y, current_z, constraints):
    """Create a Plotly 3D scatter plot that always shows the current point (from sliders)
    and any stored constraints."""
    fig = go.Figure()
    # Plot the current point.
    fig.add_trace(
        go.Scatter3d(
            x=[current_x],
            y=[current_y],
            z=[current_z],
            mode="markers",
            marker=dict(size=10, color="red"),
            name="Current Point",
        )
    )
    # Plot stored constraints (if any).
    if constraints:
        # Group constraints by joint.
        joint_groups = {}
        for frame, x, y, z, joint_id, joint_name in constraints:
            joint_groups.setdefault(joint_name, []).append((x, y, z))
        markers = ["circle", "square", "diamond", "cross", "x", "triangle-up"]
        colors = ["blue", "green", "orange", "purple", "brown", "pink"]
        for i, (joint, pts) in enumerate(joint_groups.items()):
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
                        size=8,
                        symbol=markers[i % len(markers)],
                        color=colors[i % len(colors)],
                    ),
                    name=f"Constraints: {joint}",
                )
            )
    fig.update_layout(
        scene=dict(
            xaxis=dict(range=[-5, 5]),
            yaxis=dict(range=[-5, 5]),
            zaxis=dict(range=[-5, 5]),
        ),
        title="3D Constraints Visualization",
        height=500,
    )
    return fig


# Use Streamlit session_state to store constraints.
if "constraints" not in st.session_state:
    st.session_state.constraints = []

# Create and display the 3D plot that updates as sliders move.
plot_fig = create_3d_plot(x_val, y_val, z_val, st.session_state.constraints)
st.plotly_chart(plot_fig, use_container_width=True)

# Button to add the current point as a constraint.
if st.button("Add Constraint"):
    try:
        frame = int(constraint_frame)
    except:
        st.error("Invalid frame number.")
    if frame < 0:
        st.error("Frame must be non-negative.")
    else:
        joint_id = JOINTS[joint_selection]
        new_constraint = (frame, x_val, y_val, z_val, joint_id, joint_selection)
        st.session_state.constraints.append(new_constraint)
        st.success(
            f"Added constraint: Frame {frame}, Joint {joint_selection}: ({x_val}, {y_val}, {z_val})"
        )

st.subheader("Constraints:")
if st.session_state.constraints:
    for c in st.session_state.constraints:
        st.write(f"Frame {c[0]}, Joint {c[5]}: ({c[1]:.2f}, {c[2]:.2f}, {c[3]:.2f})")
else:
    st.write("No constraints added yet.")

# Button to generate the hint array.
if st.button("Generate Hint Array"):
    try:
        total = int(total_frames)
    except:
        st.error("Total frames must be an integer.")
    hint = np.zeros((total, TOTAL_JOINTS, 3), dtype=np.float32)
    for frame, x, y, z, joint_id, _ in st.session_state.constraints:
        if frame < total:
            hint[frame, joint_id, :] = np.array([x, y, z], dtype=np.float32)
    st.text_area("Hint Array", value=str(hint), height=200)

# Button to visualize constraints (the plot above already updates continuously).
if st.button("Refresh 3D Visualization"):
    viz_fig = create_3d_plot(x_val, y_val, z_val, st.session_state.constraints)
    st.plotly_chart(viz_fig, use_container_width=True)
