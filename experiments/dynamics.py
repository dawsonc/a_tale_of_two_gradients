import jax.numpy as jnp
import os

from autograd_manipulation.sim import (
    box_single_finger_simulate,
    box_two_finger_simulate,
)
from autograd_manipulation.plotting import (
    plot_box_trajectory,
    plot_box_finger_trajectory,
    plot_box_fingers_trajectory,
)


def throw_box_right():
    # Throw the box to the right
    # abbreviate bs = box_state, fs = finger_state
    bs_initial = jnp.array([0.0, 1.0, 1.0, 4.0, 0.0, 0.0])
    fs_initial = jnp.array([0.0, 0.0, 0.0, 0.0])
    finger_k = 10
    N_steps = 1000
    fs_desired_trace = jnp.zeros((N_steps, 2))

    # Simulate
    bs_trace, _ = box_single_finger_simulate(
        bs_initial, fs_initial, fs_desired_trace, finger_k, N_steps
    )

    # Make the plots
    fig = plot_box_trajectory(bs_trace)

    # Save the plots
    current_file_dir = os.path.dirname(os.path.abspath(__file__))
    save_dir = current_file_dir + "/results/dynamics"
    os.makedirs(save_dir, exist_ok=True)
    fig.savefig(save_dir + "/throw_box_right.png")


def drop_box():
    # Starting balanced on an edge, fall to the side.
    # abbreviate bs = box_state, fs = finger_state
    bs_initial = jnp.array(
        [0.5, jnp.sqrt(2) * 0.25 - 1e-2, jnp.pi / 4 - 0.1, 0.0, 0.0, 0.0]
    )
    fs_initial = jnp.array([0.0, 0.0, 0.0, 0.0])
    finger_k = 10
    N_steps = 1000
    fs_desired_trace = jnp.zeros((N_steps, 2))

    # Simulate
    bs_trace, _ = box_single_finger_simulate(
        bs_initial, fs_initial, fs_desired_trace, finger_k, N_steps
    )

    # Make the plots
    fig = plot_box_trajectory(bs_trace)

    # Save the plots
    current_file_dir = os.path.dirname(os.path.abspath(__file__))
    save_dir = current_file_dir + "/results/dynamics"
    os.makedirs(save_dir, exist_ok=True)
    fig.savefig(save_dir + "/drop_box.png")


def push_box():
    # Start with the box on the ground and push it
    # abbreviate bs = box_state, fs = finger_state
    bs_initial = jnp.array([-1.0, 0.25, 0.0, 0.0, 0.0, 0.0])
    fs_initial = jnp.array([-1.3, 0.25, 0.0, 0.0])
    finger_k = 10
    N_steps = 1000
    fs_desired_trace = jnp.zeros((N_steps, 2))
    fs_desired_trace = fs_desired_trace.at[:, 1].set(0.25)

    # Simulate
    bs_trace, fs_trace = box_single_finger_simulate(
        bs_initial, fs_initial, fs_desired_trace, finger_k, N_steps
    )

    # Make the plots
    fig = plot_box_finger_trajectory(bs_trace, fs_trace, fs_desired_trace)

    # Save the plots
    current_file_dir = os.path.dirname(os.path.abspath(__file__))
    save_dir = current_file_dir + "/results/dynamics"
    os.makedirs(save_dir, exist_ok=True)
    fig.savefig(save_dir + "/push_box.png")


def flip_box():
    # Start with the box on the ground and flip it
    # abbreviate bs = box_state, fs = finger_state
    bs_initial = jnp.array([-1.0, 0.25, 0.0, 0.0, 0.0, 0.0])
    fs_initial = jnp.array([-1.25, 0.4, 0.0, 0.0])
    finger_k = 10
    N_steps = 600
    fs_desired_trace = jnp.zeros((N_steps, 2))
    fs_desired_trace = fs_desired_trace.at[:, 1].set(1.2)
    fs_desired_trace = fs_desired_trace.at[:, 0].set(0.3)

    # Simulate
    bs_trace, fs_trace = box_single_finger_simulate(
        bs_initial, fs_initial, fs_desired_trace, finger_k, N_steps
    )

    # Make the plots
    fig = plot_box_finger_trajectory(bs_trace, fs_trace, fs_desired_trace)

    # Save the plots
    current_file_dir = os.path.dirname(os.path.abspath(__file__))
    save_dir = current_file_dir + "/results/dynamics"
    os.makedirs(save_dir, exist_ok=True)
    fig.savefig(save_dir + "/flip_box.png")


def flip_box_two_fingers():
    # Start with the box on the ground and flip it, but use two fingers
    # abbreviate bs = box_state, fs = finger_state
    bs_initial = jnp.array([0.0, 0.25, 0.0, 0.0, 0.0, 0.0])
    fs1_initial = jnp.array([-0.25, 0.05, 0.0, 0.0])
    fs2_initial = jnp.array([0.25, 0.45, 0.0, 0.0])
    finger_k = 20
    N_steps = 700
    fs1_desired_trace = jnp.zeros((N_steps, 2))
    fs1_desired_trace = fs1_desired_trace.at[:, 1].set(0.1)
    fs1_desired_trace = fs1_desired_trace.at[:, 0].set(0.5)
    fs2_desired_trace = jnp.zeros((N_steps, 2))
    fs2_desired_trace = fs2_desired_trace.at[:, 1].set(0.4)
    fs2_desired_trace = fs2_desired_trace.at[:, 0].set(-0.5)

    # Simulate
    bs_trace, fs1_trace, fs2_trace = box_two_finger_simulate(
        bs_initial,
        fs1_initial,
        fs1_desired_trace,
        fs2_initial,
        fs2_desired_trace,
        finger_k,
        N_steps,
    )

    # Make the plots
    fig = plot_box_fingers_trajectory(
        bs_trace, fs1_trace, fs1_desired_trace, fs2_trace, fs2_desired_trace
    )

    # Save the plots
    current_file_dir = os.path.dirname(os.path.abspath(__file__))
    save_dir = current_file_dir + "/results/dynamics"
    os.makedirs(save_dir, exist_ok=True)
    fig.savefig(save_dir + "/flip_box_two_fingers.png")


if __name__ == "__main__":
    print("Throwing box...")
    throw_box_right()
    print("Done.")

    print("Pivoting box...")
    drop_box()
    print("Done.")

    print("Pushing box...")
    push_box()
    print("Done.")

    print("Flipping box...")
    flip_box()
    print("Done.")

    print("Flipping box (two fingers)...")
    flip_box_two_fingers()
    print("Done.")
