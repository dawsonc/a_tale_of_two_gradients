import jax
import jax.numpy as jnp
import os
import matplotlib.pyplot as plt
import pickle

from autograd_manipulation.sim import (
    box_single_finger_simulate,
    box_two_finger_simulate,
)
from autograd_manipulation.plotting import (
    plot_box_finger_trajectory,
    plot_box_fingers_trajectory,
)
from autograd_manipulation.optimizers import stochastic_approx_gradient_descent


def flip_box_one_finger():
    # Set up initial_guess
    setpoint0 = jnp.array([0.0, 0.25])

    # Set up some details for a stochastic approximate-gradient-based optimization
    n_gd_steps = 10
    gd_step_size = 1e-1
    perturbation_stddev = 0.5
    random_seed = 0

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

    # Make the plotting callback
    def plotting_cb(i, decision_vars, cost_trace):
        fs_desired_trace = jnp.zeros((n_sim_steps, 2))
        fs_desired_trace = fs_desired_trace.at[:, 0].add(decision_vars[0])
        fs_desired_trace = fs_desired_trace.at[:, 1].add(decision_vars[1])
        bs_trace, fs_trace = box_single_finger_simulate(
            bs_initial, fs_initial, fs_desired_trace, finger_k, n_sim_steps
        )

        # Make the plots
        sim_fig = plot_box_finger_trajectory(bs_trace, fs_trace, fs_desired_trace)
        cost_fig, cost_ax = plt.subplots(figsize=(5, 5))
        cost_ax.plot(cost_trace, "o-")
        cost_ax.set_xlabel("# gradient descent steps")
        cost_ax.set_ylabel("Cost")
        cost_fig.tight_layout()

        # Save the plots
        current_file_dir = os.path.dirname(os.path.abspath(__file__))
        save_dir = current_file_dir + "/results/optimization/flip/zero_order"
        os.makedirs(save_dir, exist_ok=True)
        sim_fig.savefig(save_dir + f"/iter_{i:03}_sim.png")
        cost_fig.savefig(save_dir + "/cost_trace.png")

        # Clean up
        plt.close(sim_fig)
        plt.close(cost_fig)

    # Run the optimization
    setpoint, cost_trace, setpoint_trace = stochastic_approx_gradient_descent(
        box_flip_cost,
        setpoint0,
        n_gd_steps,
        gd_step_size,
        perturbation_stddev,
        plotting_cb,
        random_seed,
    )
    print(f"Obtained optimal cost {cost_trace[-1]}")

    # Save the results
    current_file_dir = os.path.dirname(os.path.abspath(__file__))
    save_dir = current_file_dir + "/results/optimization/flip/zero_order"
    os.makedirs(save_dir, exist_ok=True)
    data_packet = {
        "optimal_setpoint": setpoint,
        "cost_trace": cost_trace,
        "setpoint_trace": setpoint_trace,
    }
    with open(save_dir + "/results.txt", "wb") as f:
        pickle.dump(data_packet, f)


def grasp_box_two_fingers():
    # Set up initial_guess
    setpoints0 = jnp.array([-0.05, 0.25, 0.05, 0.25])

    # Set up some details for a gradient-descent-based optimization
    n_gd_steps = 1000
    gd_step_size = 0.5
    perturbation_stddev = 0.1
    random_seed = 0

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
        fs1_desired_trace = jnp.zeros((n_sim_steps, 2))
        fs1_desired_trace = fs1_desired_trace.at[:, 0].set(finger_commands[0])
        fs1_desired_trace = fs1_desired_trace.at[:, 1].set(finger_commands[1])
        fs2_desired_trace = jnp.zeros((n_sim_steps, 2))
        fs2_desired_trace = fs2_desired_trace.at[:, 0].set(finger_commands[2])
        fs2_desired_trace = fs2_desired_trace.at[:, 1].set(finger_commands[3])

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

    # Make the plotting callback
    def plotting_cb(i, decision_vars, cost_trace):
        print(i)
        if i % 10 != 0:
            return

        fs1_desired_trace = jnp.zeros((n_sim_steps, 2))
        fs1_desired_trace = fs1_desired_trace.at[:, 0].add(decision_vars[0])
        fs1_desired_trace = fs1_desired_trace.at[:, 1].add(decision_vars[1])
        fs2_desired_trace = jnp.zeros((n_sim_steps, 2))
        fs2_desired_trace = fs2_desired_trace.at[:, 0].add(decision_vars[2])
        fs2_desired_trace = fs2_desired_trace.at[:, 1].add(decision_vars[3])
        bs_trace, fs1_trace, fs2_trace = box_two_finger_simulate(
            bs_initial,
            fs1_initial,
            fs1_desired_trace,
            fs2_initial,
            fs2_desired_trace,
            finger_k,
            n_sim_steps,
        )

        # Make the plots
        sim_fig = plot_box_fingers_trajectory(
            bs_trace, fs1_trace, fs1_desired_trace, fs2_trace, fs2_desired_trace
        )
        cost_fig, cost_ax = plt.subplots(figsize=(5, 5))
        cost_ax.plot(cost_trace, "o-")
        cost_ax.set_xlabel("# gradient descent steps")
        cost_ax.set_ylabel("Cost")
        cost_fig.tight_layout()

        # Save the plots
        current_file_dir = os.path.dirname(os.path.abspath(__file__))
        save_dir = current_file_dir + "/results/optimization/grasp/zero_order"
        os.makedirs(save_dir, exist_ok=True)
        sim_fig.savefig(save_dir + f"/iter_{i:04}_sim.png")
        cost_fig.savefig(save_dir + "/cost_trace.png")

        # Clean up
        plt.close(sim_fig)
        plt.close(cost_fig)

    # Run the optimization
    setpoints, cost_trace, setpoints_trace = stochastic_approx_gradient_descent(
        box_grasp_cost,
        setpoints0,
        n_gd_steps,
        gd_step_size,
        perturbation_stddev,
        plotting_cb,
        random_seed,
    )
    print(f"Obtained optimal cost {cost_trace[-1]}")

    # Save the results
    current_file_dir = os.path.dirname(os.path.abspath(__file__))
    save_dir = current_file_dir + "/results/optimization/grasp/zero_order"
    os.makedirs(save_dir, exist_ok=True)
    data_packet = {
        "optimal_setpoint": setpoints,
        "cost_trace": cost_trace,
        "setpoint_trace": setpoints_trace,
    }
    with open(save_dir + "/results.txt", "wb") as f:
        pickle.dump(data_packet, f)


if __name__ == "__main__":
    # print("Optimizing: one finger flip...")
    # flip_box_one_finger()

    print("Optimizing: two finger grasp...")
    grasp_box_two_fingers()
