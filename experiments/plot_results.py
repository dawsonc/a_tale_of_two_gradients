import jax
import jax.numpy as jnp
import os
import matplotlib.pyplot as plt
import pickle
import time

from autograd_manipulation.sim import (
    box_single_finger_simulate,
    box_two_finger_simulate,
)
from autograd_manipulation.plotting import (
    plot_box_finger_trajectory,
    animate_box_finger,
    plot_box_fingers_trajectory,
    animate_box_fingers,
)


def load_flip_results():
    current_file_dir = os.path.dirname(os.path.abspath(__file__))
    flip_results_dir = current_file_dir + "/results/optimization/flip/"
    first_order_results_f = open(flip_results_dir + "first_order/results.txt", "rb")
    first_order_results = pickle.load(first_order_results_f)
    zero_order_results_f = open(flip_results_dir + "zero_order/results.txt", "rb")
    zero_order_results = pickle.load(zero_order_results_f)

    return first_order_results, zero_order_results


def plot_flip_cost_traces():
    # Load cost traces
    first_order_results, zero_order_results = load_flip_results()

    # Extract cost traces
    first_order_cost = first_order_results["cost_trace"]
    zero_order_cost = zero_order_results["cost_trace"]

    # Plot
    fig, ax = plt.subplots()
    for item in (
        [ax.title, ax.xaxis.label, ax.yaxis.label]
        + ax.get_xticklabels()
        + ax.get_yticklabels()
    ):
        item.set_fontsize(16)

    ax.plot(first_order_cost, "x-", color="black", label="Exact GD", linewidth=2)
    ax.plot(
        zero_order_cost, "x--", color="darkgrey", label="Approximate GD", linewidth=2
    )
    ax.legend(fontsize=16)
    ax.set_xlabel("# parameter updates")
    ax.set_ylabel(r"$J_{flip}$")

    # Save plot
    current_file_dir = os.path.dirname(os.path.abspath(__file__))
    flip_results_dir = current_file_dir + "/results/optimization/flip/"
    fig.savefig(flip_results_dir + "flip_cost_traces.png")


def plot_flip_behaviors():
    # Load results
    first_order_results, zero_order_results = load_flip_results()

    # Get behavior with the initial guess
    initial_setpoint = first_order_results["setpoint_trace"][0]
    # Set up some variables we'll reuse for simulations
    bs_initial = jnp.array([-1.0, 0.25, jnp.pi / 2, 0.0, 0.0, 0.0])
    fs_initial = jnp.array([-1.25, 0.4, 0.0, 0.0])
    finger_k = 10
    n_sim_steps = 600

    fs_desired_trace = jnp.zeros((n_sim_steps, 2))
    fs_desired_trace = fs_desired_trace.at[:, 0].add(initial_setpoint[0])
    fs_desired_trace = fs_desired_trace.at[:, 1].add(initial_setpoint[1])
    bs_trace, fs_trace = box_single_finger_simulate(
        bs_initial, fs_initial, fs_desired_trace, finger_k, n_sim_steps
    )

    # Make the plot
    sim_fig = plot_box_finger_trajectory(bs_trace, fs_trace, fs_desired_trace)
    ax = sim_fig.axes[0]
    ax.set_xlim([-1.5, 0.4])
    ax.set_ylim([-0.1, 1.0])

    # Save the plot
    current_file_dir = os.path.dirname(os.path.abspath(__file__))
    save_dir = current_file_dir + "/results/optimization/flip/"
    sim_fig.savefig(save_dir + "initial_sim.png")
    plt.close(sim_fig)

    # Make the animation
    anim = animate_box_finger(bs_trace, fs_trace, fs_desired_trace)
    # Save it
    anim.save(save_dir + "initial_sim.gif", writer="imagemagick")

    # Get behavior with the first-order solution
    setpoint = first_order_results["optimal_setpoint"]
    # Set up some variables we'll reuse for simulations
    bs_initial = jnp.array([-1.0, 0.25, jnp.pi / 2, 0.0, 0.0, 0.0])
    fs_initial = jnp.array([-1.25, 0.4, 0.0, 0.0])
    finger_k = 10
    n_sim_steps = 600

    fs_desired_trace = jnp.zeros((n_sim_steps, 2))
    fs_desired_trace = fs_desired_trace.at[:, 0].add(setpoint[0])
    fs_desired_trace = fs_desired_trace.at[:, 1].add(setpoint[1])
    bs_trace, fs_trace = box_single_finger_simulate(
        bs_initial, fs_initial, fs_desired_trace, finger_k, n_sim_steps
    )

    # Make the plot
    sim_fig = plot_box_finger_trajectory(bs_trace, fs_trace, fs_desired_trace)
    ax = sim_fig.axes[0]
    ax.set_xlim([-1.5, 0.4])
    ax.set_ylim([-0.1, 1.0])

    # Save the plot
    current_file_dir = os.path.dirname(os.path.abspath(__file__))
    save_dir = current_file_dir + "/results/optimization/flip/"
    sim_fig.savefig(save_dir + "first_order_optimal_sim.png")
    plt.close(sim_fig)

    # Make the animation
    anim = animate_box_finger(bs_trace, fs_trace, fs_desired_trace)
    # Save it
    anim.save(save_dir + "first_order_optimal_sim.gif", writer="imagemagick")

    # Get behavior with the zero-order solution
    setpoint = zero_order_results["optimal_setpoint"]
    # Set up some variables we'll reuse for simulations
    bs_initial = jnp.array([-1.0, 0.25, jnp.pi / 2, 0.0, 0.0, 0.0])
    fs_initial = jnp.array([-1.25, 0.4, 0.0, 0.0])
    finger_k = 10
    n_sim_steps = 600

    fs_desired_trace = jnp.zeros((n_sim_steps, 2))
    fs_desired_trace = fs_desired_trace.at[:, 0].add(setpoint[0])
    fs_desired_trace = fs_desired_trace.at[:, 1].add(setpoint[1])
    bs_trace, fs_trace = box_single_finger_simulate(
        bs_initial, fs_initial, fs_desired_trace, finger_k, n_sim_steps
    )

    # Make the plot
    sim_fig = plot_box_finger_trajectory(bs_trace, fs_trace, fs_desired_trace)
    ax = sim_fig.axes[0]
    ax.set_xlim([-1.5, 0.4])
    ax.set_ylim([-0.1, 1.0])

    # Save the plot
    current_file_dir = os.path.dirname(os.path.abspath(__file__))
    save_dir = current_file_dir + "/results/optimization/flip/"
    sim_fig.savefig(save_dir + "zero_order_optimal_sim.png")
    plt.close(sim_fig)

    # Make the animation
    anim = animate_box_finger(bs_trace, fs_trace, fs_desired_trace)
    # Save it
    anim.save(save_dir + "zero_order_optimal_sim.gif", writer="imagemagick")


def load_grasp_results():
    current_file_dir = os.path.dirname(os.path.abspath(__file__))
    grasp_results_dir = current_file_dir + "/results/optimization/grasp/"
    first_order_results_f = open(grasp_results_dir + "first_order/results.txt", "rb")
    first_order_results = pickle.load(first_order_results_f)
    zero_order_results_f = open(grasp_results_dir + "zero_order/results.txt", "rb")
    zero_order_results = pickle.load(zero_order_results_f)

    return first_order_results, zero_order_results


def plot_grasp_cost_traces():
    # Load cost traces
    first_order_results, zero_order_results = load_grasp_results()

    # Extract cost traces
    first_order_cost = first_order_results["cost_trace"]
    zero_order_cost = zero_order_results["cost_trace"]

    # Plot
    fig, ax = plt.subplots()
    for item in (
        [ax.title, ax.xaxis.label, ax.yaxis.label]
        + ax.get_xticklabels()
        + ax.get_yticklabels()
    ):
        item.set_fontsize(16)

    ax.plot(first_order_cost, "x-", color="black", label="Exact GD", linewidth=2)
    ax.plot(
        zero_order_cost, "x--", color="darkgrey", label="Approximate GD", linewidth=2
    )
    ax.legend(fontsize=16)
    ax.set_xlabel("# parameter updates")
    ax.set_ylabel(r"$J_{grasp}$")

    # Save plot
    current_file_dir = os.path.dirname(os.path.abspath(__file__))
    grasp_results_dir = current_file_dir + "/results/optimization/grasp/"
    fig.savefig(grasp_results_dir + "grasp_cost_traces.png")
    plt.close(fig)

    # For clarity, also plot side by side
    fig, axs = plt.subplots(1, 2)
    for ax in axs:
        for item in (
            [ax.title, ax.xaxis.label, ax.yaxis.label]
            + ax.get_xticklabels()
            + ax.get_yticklabels()
        ):
            item.set_fontsize(30)

    axs[0].plot(first_order_cost, "x-", color="black", label="Exact GD", linewidth=2)
    axs[0].set_xlabel("# parameter updates")
    axs[0].set_ylabel(r"$J_{flip}$")
    axs[0].legend(["Exact GD"], fontsize=16)
    axs[1].plot(
        zero_order_cost, "x--", color="darkgrey", label="Approximate GD", linewidth=2
    )
    axs[1].set_xlabel("# parameter updates")
    axs[1].legend(["Approximate GD"], fontsize=16)
    plt.show()


def plot_grasp_behaviors():
    # Load results
    first_order_results, zero_order_results = load_grasp_results()

    # Get behavior with the initial guess
    setpoint = first_order_results["setpoint_trace"][0]
    # Set up some variables we'll reuse for simulations
    bs_initial = jnp.array([0.0, 0.25, 0.0, 0.0, 0.0, 0.0])
    fs1_initial = jnp.array([-0.3, 0.25, 0.0, 0.0])
    fs2_initial = jnp.array([0.3, 0.25, 0.0, 0.0])
    finger_k = 40
    n_sim_steps = 500

    fs1_desired_trace = jnp.zeros((n_sim_steps, 2))
    fs1_desired_trace = fs1_desired_trace.at[:, 0].add(setpoint[0])
    fs1_desired_trace = fs1_desired_trace.at[:, 1].add(setpoint[1])
    fs2_desired_trace = jnp.zeros((n_sim_steps, 2))
    fs2_desired_trace = fs2_desired_trace.at[:, 0].add(setpoint[2])
    fs2_desired_trace = fs2_desired_trace.at[:, 1].add(setpoint[3])
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
    ax = sim_fig.axes[0]
    ax.set_xlim([-0.6, 0.6])
    ax.set_ylim([-0.1, 1.2])

    # Save the plot
    current_file_dir = os.path.dirname(os.path.abspath(__file__))
    save_dir = current_file_dir + "/results/optimization/grasp/"
    sim_fig.savefig(save_dir + "initial_sim.png")
    plt.close(sim_fig)

    # Make the animation
    anim = animate_box_fingers(
        bs_trace, fs1_trace, fs1_desired_trace, fs2_trace, fs2_desired_trace
    )
    # Save it
    anim.save(save_dir + "initial_sim.gif", writer="imagemagick")

    # Get behavior with the initial guess
    setpoint = first_order_results["optimal_setpoint"]
    # Set up some variables we'll reuse for simulations
    bs_initial = jnp.array([0.0, 0.25, 0.0, 0.0, 0.0, 0.0])
    fs1_initial = jnp.array([-0.3, 0.25, 0.0, 0.0])
    fs2_initial = jnp.array([0.3, 0.25, 0.0, 0.0])
    finger_k = 40
    n_sim_steps = 500

    fs1_desired_trace = jnp.zeros((n_sim_steps, 2))
    fs1_desired_trace = fs1_desired_trace.at[:, 0].add(setpoint[0])
    fs1_desired_trace = fs1_desired_trace.at[:, 1].add(setpoint[1])
    fs2_desired_trace = jnp.zeros((n_sim_steps, 2))
    fs2_desired_trace = fs2_desired_trace.at[:, 0].add(setpoint[2])
    fs2_desired_trace = fs2_desired_trace.at[:, 1].add(setpoint[3])
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
    ax = sim_fig.axes[0]
    ax.set_xlim([-0.6, 0.6])
    ax.set_ylim([-0.1, 1.2])

    # Save the plot
    current_file_dir = os.path.dirname(os.path.abspath(__file__))
    save_dir = current_file_dir + "/results/optimization/grasp/"
    sim_fig.savefig(save_dir + "first_order_optimal_sim.png")
    plt.close(sim_fig)

    # Make the animation
    anim = animate_box_fingers(
        bs_trace, fs1_trace, fs1_desired_trace, fs2_trace, fs2_desired_trace
    )
    # Save it
    anim.save(save_dir + "first_order_optimal_sim.gif", writer="imagemagick")

    # Get behavior with the approximate gradient
    setpoint = zero_order_results["optimal_setpoint"]
    # Set up some variables we'll reuse for simulations
    bs_initial = jnp.array([0.0, 0.25, 0.0, 0.0, 0.0, 0.0])
    fs1_initial = jnp.array([-0.3, 0.25, 0.0, 0.0])
    fs2_initial = jnp.array([0.3, 0.25, 0.0, 0.0])
    finger_k = 40
    n_sim_steps = 500

    fs1_desired_trace = jnp.zeros((n_sim_steps, 2))
    fs1_desired_trace = fs1_desired_trace.at[:, 0].add(setpoint[0])
    fs1_desired_trace = fs1_desired_trace.at[:, 1].add(setpoint[1])
    fs2_desired_trace = jnp.zeros((n_sim_steps, 2))
    fs2_desired_trace = fs2_desired_trace.at[:, 0].add(setpoint[2])
    fs2_desired_trace = fs2_desired_trace.at[:, 1].add(setpoint[3])
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
    ax = sim_fig.axes[0]
    ax.set_xlim([-0.6, 0.6])
    ax.set_ylim([-0.1, 1.2])

    # Save the plot
    current_file_dir = os.path.dirname(os.path.abspath(__file__))
    save_dir = current_file_dir + "/results/optimization/grasp/"
    sim_fig.savefig(save_dir + "zero_order_optimal_sim.png")
    plt.close(sim_fig)

    # Make the animation
    anim = animate_box_fingers(
        bs_trace, fs1_trace, fs1_desired_trace, fs2_trace, fs2_desired_trace
    )
    # Save it
    anim.save(save_dir + "zero_order_optimal_sim.gif", writer="imagemagick")


if __name__ == "__main__":
    # plot_flip_cost_traces()
    # plot_grasp_cost_traces()

    # plot_flip_behaviors()
    plot_grasp_behaviors()
