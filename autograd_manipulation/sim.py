"""Automatically-differentiable manipulation simulation engine using JAX"""
import jax.numpy as jnp
import jax


@jax.jit
def box_finger_signed_distance(box_pose, finger_pose, box_size):
    """Compute the signed distance from the box to the finger

    args:
        box_pose: current (x, z, theta) state of the box
        finger_pose: current (x, z) state of the finger
        box_size: side length of box
    returns:
        float signed distance
    """
    # Credit to this stackoverflow answer for the inspiration for this code:
    # stackoverflow.com/questions/30545052/calculate-signed-distance-between-point-and-
    # rectangle

    # First transform the finger (x, z) into the box frame
    p_WF = finger_pose
    p_WB = box_pose[:2]
    theta_B = box_pose[2]

    p_BF_W = p_WF - p_WB
    # Rotate p_BF_W by -theta about the z axis to get position in box frame
    R_WB = jnp.array(
        [[jnp.cos(theta_B), -jnp.sin(theta_B)], [jnp.sin(theta_B), jnp.cos(theta_B)]]
    )
    R_BW = R_WB.T
    p_BF = R_BW @ p_BF_W

    # Now get the signed distance
    x_dist = jnp.maximum(-(p_BF[0] + box_size / 2.0), p_BF[0] - box_size / 2.0)
    z_dist = jnp.maximum(-(p_BF[1] + box_size / 2.0), p_BF[1] - box_size / 2.0)

    # phi = signed distance.
    phi = jnp.minimum(0.0, jnp.maximum(x_dist, z_dist))
    phi = phi + jnp.linalg.norm(
        jnp.maximum(jnp.array([1e-3, 1e-3]), jnp.array([x_dist, z_dist]))
    )

    return phi


@jax.jit
def rotation_matrix(theta):
    """Return the 2D rotation matrix for angle theta"""
    return jnp.array(
        [[jnp.cos(theta), -jnp.sin(theta)], [jnp.sin(theta), jnp.cos(theta)]]
    )


@jax.jit
def calc_finger_ground_force(finger_state, mu_d, c, psi_s, contact_k, contact_d):
    """Compute the contact force between a finger and the ground.

    args:
        finger_state: current (x, z, theta, vx, vz, omega) state of the box
        mu_d: coefficient of friction between box and ground while slipping
        c: coefficient of tangential velocity in determining sticking friction
        psi_s: tangential velocity where slipping begins
        contact_k: spring constant of contact
        contact_d: damping constant of contact
    returns:
        contact force in x and z
    """
    # Get the position and velocity of the finger in the world frame
    p_WF = finger_state[:2]
    v_WF = finger_state[2:]

    # Get penetration into ground
    phi_finger_ground = jnp.minimum(jnp.zeros(1), p_WF[1])

    # Get the contact forces. Approximate ground force as a damped spring, as in
    # the simplified friction model from eq 21 and 22 in
    # https://arxiv.org/pdf/2109.05143.pdf, but with damping.
    normal_velocity = v_WF[1]
    normal_force = -contact_k * phi_finger_ground
    normal_force = normal_force - contact_d * normal_velocity * (phi_finger_ground < 0)
    tangential_velocity = v_WF[0]
    sticking_mask = jnp.abs(tangential_velocity) <= psi_s
    slipping_mask = jnp.logical_not(sticking_mask) * jnp.sign(tangential_velocity)
    mu = sticking_mask * c * tangential_velocity + slipping_mask * mu_d
    tangent_force = -mu * normal_force

    contact_force = jnp.array([tangent_force, normal_force]).reshape(2)

    return contact_force


@jax.jit
def calc_box_ground_wrench(box_state, box_size, mu_d, c, psi_s, contact_k, contact_d):
    """Compute the contact wrench between the box and the ground.

    args:
        box_state: current (x, z, theta, vx, vz, omega) state of the box
        box_size: float indicating the side length of the box
        mu_d: coefficient of friction between box and ground while slipping
        c: coefficient of tangential velocity in determining sticking friction
        psi_s: tangential velocity where slipping begins
        contact_k: spring constant of contact
        contact_d: damping constant of contact
    returns:
        contact wrench in x, z, and theta.
    """
    # Start by finding any box corner points that intersect the ground at z = 0
    half_size = box_size / 2.0
    p_BC = jnp.array(
        [
            [-half_size, half_size],  # top left
            [half_size, half_size],  # top right
            [-half_size, -half_size],  # bottom left
            [half_size, -half_size],  # bottom right
        ]
    )  # corner points in box frame
    # Transform into world frame
    R_WB = rotation_matrix(box_state[2])
    p_BC_W = (R_WB @ p_BC.T).T
    p_WC = p_BC_W + jnp.tile(box_state[:2], [4, 1])

    # Also find the velocities of each corner point
    r = jnp.sqrt(2) * half_size
    v_BC = (
        box_state[5]
        * r
        * jnp.array(
            [
                [-1, -1],  # top left
                [-1, 1],  # top right
                [1, -1],  # bottom left
                [1, 1],  # bottom right
            ]
        )
    )  # corner point velocities in box frame
    # Transform to world frame
    v_WC = (R_WB @ v_BC.T).T + jnp.tile(box_state[3:5], [4, 1])

    # Find any that have negative z: min(0, signed distance)
    phi_corner_ground = jnp.minimum(jnp.zeros(4), p_WC[:, 1])

    # For each corner, sum up the forces and torques due to penetration with the ground
    contact_wrench_on_box = jnp.zeros(3)
    for i in range(4):
        # Get the friction force. Approximate ground force as a damped spring, as in
        # the simplified friction model from eq 21 and 22 in
        # https://arxiv.org/pdf/2109.05143.pdf, but with damping.
        normal_velocity = v_WC[i, 1]
        normal_force = -contact_k * phi_corner_ground[i]
        normal_force = normal_force - contact_d * normal_velocity * (
            phi_corner_ground[i] < 0
        )
        tangential_velocity = v_WC[i, 0]
        sticking_mask = jnp.abs(tangential_velocity) <= psi_s
        slipping_mask = jnp.logical_not(sticking_mask) * jnp.sign(tangential_velocity)
        mu = sticking_mask * c * tangential_velocity + slipping_mask * mu_d
        tangent_force = -mu * normal_force

        contact_force = jnp.array([tangent_force, normal_force])

        # Add the friction force to the box
        contact_wrench_on_box = contact_wrench_on_box.at[:2].add(contact_force)

        # Also add the torque from this interaction
        contact_wrench_on_box = contact_wrench_on_box.at[2].add(
            jnp.cross(p_BC_W[i, :], contact_force)
        )

    return contact_wrench_on_box


@jax.jit
def calc_box_finger_wrench(
    box_state, finger_state, box_size, mu_d, c, psi_s, contact_k, contact_d
):
    """Compute the contact wrench between the box and the ground.

    args:
        box_state: current (x, z, theta, vx, vz, omega) state of the box
        finger_state: current (x, z, vx, vz) state of the finger
        box_size: float indicating the side length of the box
        mu_d: coefficient of friction between box and ground while slipping
        c: coefficient of tangential velocity in determining sticking friction
        psi_s: tangential velocity where slipping begins
        contact_k: spring constant of contact
        contact_d: damping constant of contact
    returns:
        Tuple of
            - contact wrench on box in x, z, and theta.
            - contact force on finger in x and z.
    """
    # Contact point is just the finger point in the box frame
    p_WF = finger_state[:2]
    p_WB = box_state[:2]
    p_BF_W = p_WF - p_WB
    R_WB = rotation_matrix(box_state[2])
    p_BF = R_WB.T @ p_BF_W

    # Get velocity of the finger in box frame
    v_WF = finger_state[2:]
    v_WB = box_state[3:5]
    v_BF_W = v_WF - v_WB
    v_BF = R_WB.T @ v_BF_W

    # Get velocity of contact point in box frame
    v_Bcontact = box_state[5] * jnp.array([[0, -1], [1, 0]]) @ p_BF

    # Get velocity of finger relative to contact pt in box frame
    v_contactF_B = v_BF - v_Bcontact

    # Get the normal vector of the contact in the box frame
    right_or_up = p_BF[1] > -p_BF[0]
    left_or_up = p_BF[1] > p_BF[0]
    normal_right = jnp.logical_and(right_or_up, jnp.logical_not(left_or_up))
    normal_up = jnp.logical_and(right_or_up, left_or_up)
    normal_left = jnp.logical_and(jnp.logical_not(right_or_up), left_or_up)
    normal_down = jnp.logical_and(
        jnp.logical_not(right_or_up), jnp.logical_not(left_or_up)
    )
    normal = normal_right * jnp.array([1.0, 0.0])
    normal += normal_left * jnp.array([-1.0, 0.0])
    normal += normal_up * jnp.array([0.0, 1.0])
    normal += normal_down * jnp.array([0.0, -1.0])

    # Get the tangent vector, which is orthogonal to the normal vector
    # and points in the same direction as the relative velocity
    tangential_velocity = (
        v_contactF_B - v_contactF_B.dot(normal) * normal
    )  # relative velocity in tangent direction
    normal_velocity = v_contactF_B.dot(normal)  # scalar, along the normal vector
    tangent = tangential_velocity / (jnp.linalg.norm(tangential_velocity + 1e-3) + 1e-3)

    # Get signed distance
    phi_finger_box = box_finger_signed_distance(
        box_state[:3], finger_state[:2], box_size
    )
    # Clip to only consider negative values
    phi_finger_box = jnp.minimum(0, phi_finger_box)

    # Use the same simplified friction model as used for ground contact
    normal_force = -contact_k * phi_finger_box  # scalar, in normal direction
    normal_force = normal_force - contact_d * normal_velocity * (phi_finger_box < 0)
    sticking_mask = jnp.linalg.norm(tangential_velocity + 1e-3) <= psi_s
    slipping_mask = jnp.logical_not(sticking_mask)
    mu = sticking_mask * c * tangential_velocity + slipping_mask * mu_d * tangent
    tangent_force = -mu * normal_force  # vector!

    # Sum up the contact forces in the box frame
    contact_force_B = normal_force * normal + tangent_force
    # transform into the world frame
    contact_force_W = R_WB @ contact_force_B

    # Add the contact force to the box and finger
    box_wrench = jnp.zeros(3)
    box_wrench = box_wrench.at[:2].add(-contact_force_W)
    box_wrench = box_wrench.at[2].add(jnp.cross(p_BF_W, -contact_force_W))

    finger_forces = contact_force_W

    return box_wrench, finger_forces


@jax.jit
def box_single_finger_step(
    box_state,
    finger_state,
    finger_state_desired,
    finger_control_stiffness,
):
    """Compute a single discrete-time update for box manipulation with one finger, using
    the penalty method for contact modelling with a simplified Coulomb friction model

    args:
        box_state: current (x, z, theta, vx, vz, omega) state of the box
        finger_state: current (x, z, vx, vz) state of the finger
        finger_state_desired: desired (x_d, z_d) state of the finger
        finger_control_stiffness: the parameter for the finger stiffness control
    returns:
        new_box_state, new_finger_state
    """
    ######################################
    # define parameters of the simulation
    ######################################
    # Box properties
    box_mass_kg = 1.0
    box_side_m = 0.5
    box_inertia = 1 / 6 * box_mass_kg * box_side_m ** 2

    # Finger properties
    finger_mass_kg = 0.1
    finger_control_damping = 2

    # Contact properties
    mu_d = 0.7
    c = 2.0
    psi_s = mu_d / c
    contact_k = 1000
    contact_d = 2 * jnp.sqrt(box_mass_kg * contact_k)  # critical damping

    # General properties
    g = 9.81
    dt = 0.001  # seconds per step

    ######################################
    # Get forces on each body
    ######################################
    finger_forces = jnp.zeros(2)
    box_forces = jnp.zeros(3)

    # Gravitational force on each body
    finger_forces = finger_forces.at[1].add(-g * finger_mass_kg)
    box_forces = box_forces.at[1].add(-g * box_mass_kg)

    # Control forces on finger
    finger_pos_error = finger_state_desired - finger_state[:2]
    finger_vel_error = -finger_state[2:]
    finger_forces = finger_forces + finger_control_stiffness * finger_pos_error
    finger_forces = finger_forces + finger_control_damping * finger_vel_error

    # Contact forces from the ground.
    box_forces += calc_box_ground_wrench(
        box_state, box_side_m, mu_d, c, psi_s, contact_k, contact_d
    )
    finger_forces += calc_finger_ground_force(
        finger_state, mu_d, c, psi_s, contact_k, contact_d
    )

    # Contact forces between box and finger
    finger_wrench_on_box, box_force_on_finger = calc_box_finger_wrench(
        box_state, finger_state, box_side_m, mu_d, c, psi_s, contact_k, contact_d
    )
    box_forces += finger_wrench_on_box
    finger_forces += box_force_on_finger

    ######################################
    # Numerically integrate
    ######################################
    # Build the derivatives matrix
    box_state_dot = jnp.zeros(6)
    finger_state_dot = jnp.zeros(4)

    # Velocities
    box_state_dot = box_state_dot.at[:3].add(box_state[3:])
    finger_state_dot = finger_state_dot.at[:2].add(finger_state[2:])

    # Forces
    box_state_dot = box_state_dot.at[3:5].add(box_forces[:2] / box_mass_kg)
    finger_state_dot = finger_state_dot.at[2:].add(finger_forces / finger_mass_kg)

    # Torques
    box_state_dot = box_state_dot.at[5].add(box_forces[2] / box_inertia)

    # Itegrate
    new_box_state = box_state + dt * box_state_dot
    new_finger_state = finger_state + dt * finger_state_dot

    return new_box_state, new_finger_state


def box_single_finger_simulate(
    box_state_initial,
    finger_state_initial,
    finger_state_desired_trace,
    finger_control_stiffness,
    N_steps,
):
    """Simulate the evolution of the box-finger system with one finger, starting at the
    given initial states and applying the specified control inputs

    args:
        box_state_initial: initial (x, z, theta, vx, vz, omega) state of the box
        finger_state_initial: initial (x, z, vx, vz) state of the finger
        finger_state_desired_trace: N_steps x 2 array of desired (x_d, z_d) state of the
                                    finger over time
        finger_control_stiffness: the parameter for the finger stiffness control
        N_steps: int specifying the number of discrete time steps to simulate
    returns:
        box_state_trace, finger_state_trace
    """
    # Create arrays to store simulation traces
    box_state_trace = jnp.zeros((N_steps, 6))
    finger_state_trace = jnp.zeros((N_steps, 4))

    # Store the initial conditions
    box_state_trace = box_state_trace.at[0, :].set(box_state_initial)
    finger_state_trace = finger_state_trace.at[0, :].set(finger_state_initial)

    # Simulate
    for i in range(1, N_steps):
        # get currents state
        current_box_state = box_state_trace[i - 1, :]
        current_finger_state = finger_state_trace[i - 1, :]
        current_finger_state_desired = finger_state_desired_trace[i - 1]

        # get next state
        next_box_state, next_finger_state = box_single_finger_step(
            current_box_state,
            current_finger_state,
            current_finger_state_desired,
            finger_control_stiffness,
        )

        # Save
        box_state_trace = box_state_trace.at[i, :].set(next_box_state)
        finger_state_trace = finger_state_trace.at[i, :].set(next_finger_state)

    # Return the simulated values
    return box_state_trace, finger_state_trace


@jax.jit
def box_two_finger_step(
    box_state,
    finger1_state,
    finger1_state_desired,
    finger2_state,
    finger2_state_desired,
    finger_control_stiffness,
):
    """Compute a single discrete-time update for box manipulation with one finger, using
    the penalty method for contact modelling with a simplified Coulomb friction model

    args:
        box_state: current (x, z, theta, vx, vz, omega) state of the box
        finger1_state: current (x, z, vx, vz) state of the first finger
        finger1_state_desired: desired (x_d, z_d) state of the first finger
        finger2_state: current (x, z, vx, vz) state of the second finger
        finger2_state_desired: desired (x_d, z_d) state of the second finger
        finger_control_stiffness: the parameter for the finger stiffness control
    returns:
        new_box_state, new_finger_state
    """
    ######################################
    # define parameters of the simulation
    ######################################
    # Box properties
    box_mass_kg = 1.0
    box_side_m = 0.5
    box_inertia = 1 / 6 * box_mass_kg * box_side_m ** 2

    # Finger properties
    finger_mass_kg = 0.1
    finger_control_damping = 2

    # Contact properties
    mu_d = 0.7
    c = 2.0
    psi_s = mu_d / c
    contact_k = 1000
    contact_d = 2 * jnp.sqrt(box_mass_kg * contact_k)  # critical damping

    # General properties
    g = 9.81
    dt = 0.001  # seconds per step

    ######################################
    # Get forces on each body
    ######################################
    finger1_forces = jnp.zeros(2)
    finger2_forces = jnp.zeros(2)
    box_forces = jnp.zeros(3)

    # Gravitational force on each body
    finger1_forces = finger1_forces.at[1].add(-g * finger_mass_kg)
    finger2_forces = finger2_forces.at[1].add(-g * finger_mass_kg)
    box_forces = box_forces.at[1].add(-g * box_mass_kg)

    # Control forces on fingers
    finger1_pos_error = finger1_state_desired - finger1_state[:2]
    finger1_vel_error = -finger1_state[2:]
    finger1_forces = finger1_forces + finger_control_stiffness * finger1_pos_error
    finger1_forces = finger1_forces + finger_control_damping * finger1_vel_error

    finger2_pos_error = finger2_state_desired - finger2_state[:2]
    finger2_vel_error = -finger2_state[2:]
    finger2_forces = finger2_forces + finger_control_stiffness * finger2_pos_error
    finger2_forces = finger2_forces + finger_control_damping * finger2_vel_error

    # Contact forces from ground.
    box_forces += calc_box_ground_wrench(
        box_state, box_side_m, mu_d, c, psi_s, contact_k, contact_d
    )
    finger1_forces += calc_finger_ground_force(
        finger1_state, mu_d, c, psi_s, contact_k, contact_d
    )
    finger2_forces += calc_finger_ground_force(
        finger2_state, mu_d, c, psi_s, contact_k, contact_d
    )

    # Contact forces between box and fingers
    finger1_wrench_on_box, box_force_on_finger1 = calc_box_finger_wrench(
        box_state, finger1_state, box_side_m, mu_d, c, psi_s, contact_k, contact_d
    )
    box_forces += finger1_wrench_on_box
    finger1_forces += box_force_on_finger1

    finger2_wrench_on_box, box_force_on_finger2 = calc_box_finger_wrench(
        box_state, finger2_state, box_side_m, mu_d, c, psi_s, contact_k, contact_d
    )
    box_forces += finger2_wrench_on_box
    finger2_forces += box_force_on_finger2

    ######################################
    # Numerically integrate
    ######################################
    # Build the derivatives matrix
    box_state_dot = jnp.zeros(6)
    finger1_state_dot = jnp.zeros(4)
    finger2_state_dot = jnp.zeros(4)

    # Velocities
    box_state_dot = box_state_dot.at[:3].add(box_state[3:])
    finger1_state_dot = finger1_state_dot.at[:2].add(finger1_state[2:])
    finger2_state_dot = finger2_state_dot.at[:2].add(finger2_state[2:])

    # Forces
    box_state_dot = box_state_dot.at[3:5].add(box_forces[:2] / box_mass_kg)
    finger1_state_dot = finger1_state_dot.at[2:].add(finger1_forces / finger_mass_kg)
    finger2_state_dot = finger2_state_dot.at[2:].add(finger2_forces / finger_mass_kg)

    # Torques
    box_state_dot = box_state_dot.at[5].add(box_forces[2] / box_inertia)

    # Itegrate
    new_box_state = box_state + dt * box_state_dot
    new_finger1_state = finger1_state + dt * finger1_state_dot
    new_finger2_state = finger2_state + dt * finger2_state_dot

    return new_box_state, new_finger1_state, new_finger2_state


def box_two_finger_simulate(
    box_state_initial,
    finger1_state_initial,
    finger1_state_desired_trace,
    finger2_state_initial,
    finger2_state_desired_trace,
    finger_control_stiffness,
    N_steps,
):
    """Simulate the evolution of the box-finger system with one finger, starting at the
    given initial states and applying the specified control inputs

    args:
        box_state_initial: initial (x, z, theta, vx, vz, omega) state of the box
        finger1_state_initial: initial (x, z, vx, vz) state of the finger
        finger1_state_desired_trace: N_steps x 2 array of desired (x_d, z_d) state of
                                     the finger over time
        finger2_state_initial: initial (x, z, vx, vz) state of the finger
        finger2_state_desired_trace: N_steps x 2 array of desired (x_d, z_d) state of
                                     the finger over time
        finger_control_stiffness: the parameter for the finger stiffness control
        N_steps: int specifying the number of discrete time steps to simulate
    returns:
        box_state_trace, finger_state_trace
    """
    # Create arrays to store simulation traces
    box_state_trace = jnp.zeros((N_steps, 6))
    finger1_state_trace = jnp.zeros((N_steps, 4))
    finger2_state_trace = jnp.zeros((N_steps, 4))

    # Store the initial conditions
    box_state_trace = box_state_trace.at[0, :].set(box_state_initial)
    finger1_state_trace = finger1_state_trace.at[0, :].set(finger1_state_initial)
    finger2_state_trace = finger2_state_trace.at[0, :].set(finger2_state_initial)

    # Simulate
    for i in range(1, N_steps):
        # get currents state
        current_box_state = box_state_trace[i - 1, :]
        current_finger1_state = finger1_state_trace[i - 1, :]
        current_finger1_state_desired = finger1_state_desired_trace[i - 1]
        current_finger2_state = finger2_state_trace[i - 1, :]
        current_finger2_state_desired = finger2_state_desired_trace[i - 1]

        # get next state
        next_box_state, next_finger1_state, next_finger2_state = box_two_finger_step(
            current_box_state,
            current_finger1_state,
            current_finger1_state_desired,
            current_finger2_state,
            current_finger2_state_desired,
            finger_control_stiffness,
        )

        # Save
        box_state_trace = box_state_trace.at[i, :].set(next_box_state)
        finger1_state_trace = finger1_state_trace.at[i, :].set(next_finger1_state)
        finger2_state_trace = finger2_state_trace.at[i, :].set(next_finger2_state)

    # Return the simulated values
    return box_state_trace, finger1_state_trace, finger2_state_trace
