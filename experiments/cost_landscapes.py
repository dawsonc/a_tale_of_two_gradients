import jax.numpy as jnp
import os
import matplotlib.pyplot as plt

from autograd_manipulation.sim import (
    box_single_finger_simulate,
    box_two_finger_simulate,
)
from autograd_manipulation.plotting import (
    plot_cost_landscape,
)


def flip_box_one_finger_cost_landscape():
    # Set up some details for plotting
    grid_size = 200
    var_min = (-0.5, 0.0)
    var_max = (0.5, 1.0)

    # Set up some variables we'll reuse for simulations
    bs_initial = jnp.array([-1.0, 0.25, jnp.pi / 2, 0.0, 0.0, 0.0])
    fs_initial = jnp.array([-1.25, 0.4, 0.0, 0.0])
    finger_k = 10
    n_sim_steps = 600

    # Define the cost function
    def box_flip_cost(finger_command):
        """Compute the optimization cost for pushing and flipping the box

        args:
            finger_command: (x, z) desired finger location
        """
        # Set the desired finger location as a first-order hold
        fs_desired_trace = jnp.zeros((n_sim_steps, 2))
        fs_desired_trace = fs_desired_trace.at[:, 0].add(finger_command[0])
        fs_desired_trace = fs_desired_trace.at[:, 1].add(finger_command[1])

        # Simulate
        bs_trace, _ = box_single_finger_simulate(
            bs_initial, fs_initial, fs_desired_trace, finger_k, n_sim_steps
        )

        # Make a cost using just the terminal values of x and theta
        cost = bs_trace[-1, 0] ** 2 + bs_trace[-1, 2] ** 2

        return cost

    # Plot on a grid
    cost_fig = plot_cost_landscape(
        box_flip_cost, var_min, var_max, grid_size, r"$x_{des}$", r"$y_{des}$"
    )
    # Save the plots
    current_file_dir = os.path.dirname(os.path.abspath(__file__))
    save_dir = current_file_dir + "/results/optimization/flip"
    os.makedirs(save_dir, exist_ok=True)
    cost_fig.savefig(save_dir + "/cost_landscape.png")

    # Clean up
    plt.close(cost_fig)


def grasp_box_two_fingers_cost_landscape():
    # Set up some details for plotting
    grid_size = 200
    var_min = (-1.0, 0.0)
    var_max = (1.0, 2.0)

    # Set up some variables we'll reuse for simulations
    bs_initial = jnp.array([0.0, 0.25, 0.0, 0.0, 0.0, 0.0])
    fs1_initial = jnp.array([-0.3, 0.25, 0.0, 0.0])
    fs2_initial = jnp.array([0.3, 0.25, 0.0, 0.0])
    finger_k = 40
    n_sim_steps = 500

    # Define the cost function
    def box_grasp_cost(finger_commands):
        """Compute the optimization cost for pushing and flipping the box

        args:
            finger_command: (x, z) desired finger location
        """
        # Make the setpoints symmetric to allow for 2D plotting
        fs1_desired_trace = jnp.zeros((n_sim_steps, 2))
        fs1_desired_trace = fs1_desired_trace.at[:, 0].set(finger_commands[0])
        fs1_desired_trace = fs1_desired_trace.at[:, 1].set(finger_commands[1])
        fs2_desired_trace = jnp.zeros((n_sim_steps, 2))
        fs2_desired_trace = fs2_desired_trace.at[:, 0].set(-finger_commands[0])
        fs2_desired_trace = fs2_desired_trace.at[:, 1].set(finger_commands[1])

        # Simulate
        bs_trace, fs1_trace, fs2_trace = box_two_finger_simulate(
            bs_initial,
            fs1_initial,
            fs1_desired_trace,
            fs2_initial,
            fs2_desired_trace,
            finger_k,
            n_sim_steps,
        )

        # Make a cost with terminal cost telling us to lift to y = 1.0 and stop there
        target_state = jnp.array([0.0, 1.0, 0.0, 0.0, 0.0, 0.0])
        cost_weights = jnp.array([0.1, 1.0, 0.1, 0.1, 0.1, 0.1])
        cost = jnp.sum(cost_weights * (bs_trace[-1, :] - target_state) ** 2)

        return cost

    # Plot on a grid
    cost_fig = plot_cost_landscape(
        box_grasp_cost, var_min, var_max, grid_size, r"$x_{des}$", r"$y_{des}$"
    )
    # Save the plots
    current_file_dir = os.path.dirname(os.path.abspath(__file__))
    save_dir = current_file_dir + "/results/optimization/grasp"
    os.makedirs(save_dir, exist_ok=True)
    cost_fig.savefig(save_dir + "/cost_landscape.png")

    # Clean up
    plt.close(cost_fig)


if __name__ == "__main__":
    print("Plotting cost landscape: one finger flip...")
    flip_box_one_finger_cost_landscape()

    # print("Plotting cost landscape: two finger grasp...")
    # grasp_box_two_fingers_cost_landscape()
