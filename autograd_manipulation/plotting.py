"""Plotting utilities"""
import jax.numpy as jnp
import jax
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.transforms as transforms


def make_box_patches(box_state, alpha, box_side_length, ax):
    """Adds patches for visualizing the box to the given axes

    args:
        box_state: (x, z, theta, vx, vz, thetadot)
        alpha: float transparency
        box_side_length: float side length of box
        ax: matplotlib axes
    returns:
        a list of properly transformed and colored patches for the box
    """
    box_xz = box_state[:2]
    box_theta = box_state[2]
    xform = transforms.Affine2D()
    xform = xform.rotate_around(
        box_side_length / 2.0, box_side_length / 2.0, theta=box_theta
    )
    xform = xform.translate(*(box_xz - box_side_length / 2.0))
    xform = xform + ax.transData
    box = patches.Rectangle(
        (0, 0),
        box_side_length,
        box_side_length,
        linewidth=2,
        transform=xform,
        edgecolor=plt.get_cmap("Blues")(0.1 + alpha),
        fill=False,
    )
    ax.add_patch(box)

    # Add an arrow pointing up
    xform = transforms.Affine2D()
    xform = xform.rotate_around(
        box_side_length / 2.0, box_side_length / 2.0, theta=box_theta
    )
    xform = xform.translate(*(box_xz - box_side_length / 2.0))
    xform = xform + ax.transData
    arrow = patches.Arrow(
        0,
        0,
        0,
        box_side_length / 8,
        width=box_side_length / 20,
        linewidth=2,
        transform=xform,
        edgecolor=plt.get_cmap("Blues")(0.1 + alpha),
        fill=True,
    )
    ax.add_patch(arrow)


def make_finger_patches(finger_state, alpha, ax):
    """Adds patches for visualizing the finger to the given axes

    args:
        box_state: (x, z, theta, vx, vz, thetadot)
        alpha: float transparency
        box_side_length: float side length of box
        ax: matplotlib axes
    returns:
        a list of properly transformed and colored patches for the box
    """
    finger_xz = finger_state[:2]
    xform = transforms.Affine2D()
    xform = xform.translate(*finger_xz)
    xform = xform + ax.transData
    finger = patches.Circle(
        (0, 0),
        0.005,
        linewidth=2,
        transform=xform,
        edgecolor=plt.get_cmap("Oranges")(0.1 + alpha),
        facecolor=plt.get_cmap("Oranges")(0.1 + alpha),
        fill=True,
    )
    ax.add_patch(finger)


def plot_box_trajectory(bs_trace):
    """Plot the trajectory of a box. Creates and returns a figure (but does not show it)

    args:
        bs_trace: n_sim_steps x 6 jnp.array of box states over time
    """
    # Make the plots
    fig, ax = plt.subplots(figsize=(5, 5))
    # Plot the center of mass trajectory
    ax.plot(bs_trace[:, 0], bs_trace[:, 1], label="box")

    # Plot the box over time
    n_steps_to_show = 20
    n_steps = bs_trace.shape[0]
    i_to_show = jnp.linspace(0, n_steps, n_steps_to_show, dtype=int)
    alphas = jnp.linspace(0.0, 1.0, n_steps)
    box_side_length = 0.5
    for i in i_to_show:
        make_box_patches(bs_trace[i], alphas[i].item(), box_side_length, ax)

    # Label etc.
    ax.set_xlabel("x")
    ax.set_ylabel("z")
    ax.set_aspect("equal")
    ax.legend()

    return fig


def plot_box_finger_trajectory(bs_trace, fs_trace, fs_desired_trace):
    """Plot the trajectory of a box and a finger.

    Creates and returns a figure (but does not show it)

    args:
        bs_trace: n_sim_steps x 6 jnp.array of box states over time
        fs_trace: n_sim_steps x 4 jnp.array of finger states over time
        fs_desired_trace: n_sim_steps x 2 jnp.array of finger setpoints over time
    """
    # Make the plots
    fig, ax = plt.subplots(figsize=(5, 5))
    # Plot the center of mass trajectory
    ax.plot(bs_trace[:, 0], bs_trace[:, 1], label="box")
    ax.plot(fs_trace[:, 0], fs_trace[:, 1], label="finger")
    ax.plot(fs_desired_trace[:, 0], fs_desired_trace[:, 1], "-o", label="setpoint")

    # Plot the box over time
    n_steps_to_show = 20
    n_steps = bs_trace.shape[0]
    i_to_show = jnp.linspace(0, n_steps, n_steps_to_show, dtype=int)
    alphas = jnp.linspace(0.0, 1.0, n_steps)
    box_side_length = 0.5
    for i in i_to_show:
        make_box_patches(bs_trace[i], alphas[i].item(), box_side_length, ax)
        make_finger_patches(fs_trace[i], alphas[i].item(), ax)

    # Label etc.
    ax.set_xlabel("x")
    ax.set_ylabel("z")
    ax.set_aspect("equal")
    ax.legend()

    return fig


def plot_box_fingers_trajectory(
    bs_trace, fs1_trace, fs1_desired_trace, fs2_trace, fs2_desired_trace
):
    """Plot the trajectory of a box and two fingers.

    Creates and returns a figure (but does not show it)

    args:
        bs_trace: n_sim_steps x 6 jnp.array of box states over time
        fs1_trace: n_sim_steps x 4 jnp.array of finger 1 states over time
        fs1_desired_trace: n_sim_steps x 2 jnp.array of finger 1 setpoints over time
        fs1_trace: n_sim_steps x 4 jnp.array of finger 2 states over time
        fs1_desired_trace: n_sim_steps x 2 jnp.array of finger 2 setpoints over time
    returns:
        a figure
    """
    # Make the plots
    fig, ax = plt.subplots(figsize=(5, 5))
    # Plot the center of mass trajectory
    ax.plot(bs_trace[:, 0], bs_trace[:, 1], label="box")
    ax.plot(fs1_trace[:, 0], fs1_trace[:, 1], color="orange", label="finger")
    ax.plot(fs2_trace[:, 0], fs2_trace[:, 1], color="orange")
    ax.plot(
        fs1_desired_trace[:, 0],
        fs1_desired_trace[:, 1],
        "-o",
        color="green",
        label="setpoint",
    )
    ax.plot(fs2_desired_trace[:, 0], fs2_desired_trace[:, 1], "-o", color="green")

    # Plot the box over time
    n_steps_to_show = 20
    n_steps = bs_trace.shape[0]
    i_to_show = jnp.linspace(0, n_steps, n_steps_to_show, dtype=int)
    alphas = jnp.linspace(0.0, 1.0, n_steps)
    box_side_length = 0.5
    for i in i_to_show:
        make_box_patches(bs_trace[i], alphas[i].item(), box_side_length, ax)
        make_finger_patches(fs1_trace[i], alphas[i].item(), ax)
        make_finger_patches(fs2_trace[i], alphas[i].item(), ax)

    # Label etc.
    ax.set_xlabel("x")
    ax.set_ylabel("z")
    ax.set_aspect("equal")
    ax.legend()

    return fig


def plot_cost_landscape(f, var_min, var_max, grid_size, xlabel, ylabel):
    """Plot f on a grid. Creates and returns a figure but does not show it.

    args:
        f: the function to plot
        var_min: a tuple (x_min, y_min)
        var_max: a tuple (x_max, y_max)
        grid_size: the size of the grid
        xlabel: label for x axis
        ylabel: label for y axis
    returns:
        a figure
    """
    # Create the grid
    x_min, y_min = var_min
    x_max, y_max = var_max
    x_range = jnp.linspace(x_min, x_max, grid_size)
    y_range = jnp.linspace(y_min, y_max, grid_size)

    X, Y = jnp.meshgrid(x_range, y_range)
    XY = jnp.stack((X, Y)).reshape(2, grid_size ** 2).T

    # Vectorize the cost function and compute it
    cost = jax.vmap(f, in_axes=0)(XY).reshape(grid_size, grid_size)

    # Make the plot
    fig, ax = plt.subplots()
    contours = ax.contourf(X, Y, cost, levels=10)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    fig.colorbar(contours)

    return fig
