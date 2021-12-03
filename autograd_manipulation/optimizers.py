"""Gradient-based and gradient-free optimization algorithms"""
import jax.numpy as jnp
import jax


def gradient_descent(f, initial_guess, n_steps, step_size, plotting_cb):
    """
    Run gradient descent to optimize the cost function f.

    args:
        f: a function taking a single jnp.array of decision variables and returning a
           tuple (cost, cost_gradient)
        initial_guess: a starting point for the decision variables
        n_steps: the number of gradient descent steps to run
        step_size: the step size parameter for gradient descent
        plotting_cb: a function taking an integer for the current iteration, a jnp.array
                     of decision variables, and a list of costs seen so far. Can plot
                     results, but should not block. No return value needed or used.
    returns:
        a tuple containing
            - the optimized decision variables, in a jnp.array
            - a list of costs at each step
            - a list of jnp.arrays of the decision variables at each step
    """
    # Set up lists to store results
    cost_trace = []
    decision_var_trace = []
    decision_vars = initial_guess.clone()

    # Do gradient descent
    for i in range(n_steps):
        # Get the cost and gradient
        cost, cost_grad = f(decision_vars)

        # Save the cost and decision vars
        cost_trace.append(cost)
        decision_var_trace.append(decision_vars)

        # Plot performance
        plotting_cb(i, decision_vars, cost_trace)

        # Update the decision variables
        decision_vars = decision_vars - step_size * cost_grad

    # Do a final evaluation and plotting
    cost, _ = f(decision_vars)
    cost_trace.append(cost)
    decision_var_trace.append(decision_vars)
    plotting_cb(n_steps, decision_vars, cost_trace)

    # Return the desired information
    return decision_vars, cost_trace, decision_var_trace


def stochastic_approx_gradient_descent(
    f, initial_guess, n_steps, step_size, perturbation_stddev, plotting_cb, random_seed
):
    """
    Run gradient descent with gradients derived using stochastic approximation
    to optimize the cost function f.

    args:
        f: a function taking a single jnp.array of decision variables and returning a
           cost
        initial_guess: a starting point for the decision variables
        n_steps: the number of gradient descent steps to run
        step_size: the step size parameter for gradient descent
        perturbation_stddev: the standard deviation of the random perturbation used
                             to approximate the gradient
        plotting_cb: a function taking an integer for the current iteration, a jnp.array
                     of decision variables, and a list of costs seen so far. Can plot
                     results, but should not block. No return value needed or used.
        random_seed: an integer
    returns:
        a tuple containing
            - the optimized decision variables, in a jnp.array
            - a list of costs at each step
            - a list of jnp.arrays of the decision variables at each step
    """
    # Set up lists to store results
    cost_trace = []
    decision_var_trace = []
    decision_vars = initial_guess.clone()

    # Set up the PRNG
    prng_key = jax.random.PRNGKey(random_seed)

    # Do approximate gradient descent
    for i in range(n_steps):
        # Get the cost at the current decision variables
        current_cost = f(decision_vars)

        # Save the cost and decision vars
        cost_trace.append(current_cost)
        decision_var_trace.append(decision_vars)

        # Plot performance
        plotting_cb(i, decision_vars, cost_trace)

        # Perturb the decision variables in a random direction and get the cost
        prng_key, subkey = jax.random.split(prng_key)
        w = perturbation_stddev * jax.random.normal(subkey, shape=decision_vars.shape)
        perturbed_cost = f(decision_vars + w)

        # Update the decision variables
        decision_vars = decision_vars - step_size * (perturbed_cost - current_cost) * w

    # Do a final evaluation and plotting
    cost = f(decision_vars)
    cost_trace.append(cost)
    decision_var_trace.append(decision_vars)
    plotting_cb(decision_vars)

    # Return the desired information
    return decision_vars, cost_trace, decision_var_trace
