"""
This file contains the implementation of a simple particle swarm optimizer.
The algorithm is based on the paper "Particle Swarm Optimization"
by James Kennedy and Russell Eberhart.
"""

import numpy as np
from copy import deepcopy


def particle_optimizer(objective, start_positions, start_velocities, max_iter):
    """Simple particle swarm optimization implementation.
    While the algorithm itself is unconstrained. Constrained optimization can be performed by
    including a constraint penalty a*max(0,g(x)) into the objective function.

    Args:
        objective ([type]): A objective function to be minimized.
                            The function has to be vectorized such that the objective of all
                            states can be computed at once.
        start_positions ([type]): The initial state of the particles with dimension
                                  (number_of_particle_states,number_of_particles)
        start_velocities ([type]): The initial velocity of the particles with the same
                                   dimension as the start_positions
        max_iter ([type]): the number of iterations before the system terminates

    Returns:
        [type]: The current global optimum state
        float: the objective function evaluated at the current global optimum state
    """
    c1 = c2 = 0.1
    w = 0.8
    X = deepcopy(start_positions)
    V = deepcopy(start_velocities)

    pbest = deepcopy(start_positions)
    pbest_obj = objective(X)
    gbest = pbest[:, pbest_obj.argmin()]
    gbest_obj = pbest_obj.min()

    for i in range(max_iter):
        print("iteration", i)
        r1, r2 = np.random.rand(2)
        V = w * V + c1*r1*(pbest - X) + c2*r2*(gbest.reshape(-1, 1)-X)
        X = X + V
        obj = objective(X)
        pbest[:, (pbest_obj >= obj)] = X[:, (pbest_obj >= obj)]
        pbest_obj = np.array([pbest_obj, obj]).min(axis=0)
        gbest = pbest[:, pbest_obj.argmin()]
        gbest_obj = pbest_obj.min()

    return gbest, gbest_obj


def particle_optimizer_log(objective, start_positions, start_velocities, max_iter):
    """Simple particle swarm optimization implementation. This variate retruns the history of
       the optimization process.

    Args:
        objective ([type]): A objective function to be minimized.
                            The function has to be vectorized such that the objective of all
                            states can be computed at once.
        start_positions ([type]): The initial state of the particles with dimension
                                  (number_of_particle_states,number_of_particles)
        start_velocities ([type]): The initial velocity of the particles with the same
                                   dimension as the start_positions
        max_iter ([type]): the number of iterations before the system terminates

    Returns:
        [type]: The current global optimum state
        float: the objective function evaluated at the current global optimum state
    """
    c1 = c2 = 0.1
    w = 0.8
    X = deepcopy(start_positions)
    X_log = [X]
    V = deepcopy(start_velocities)

    pbest = deepcopy(start_positions)
    pbest_obj = objective(X)
    obj_log =[objective(X)]
    gbest = pbest[:, pbest_obj.argmin()]
    gbest_obj = pbest_obj.min()

    for i in range(max_iter):
        print("iteration", i)
        r1, r2 = np.random.rand(2)
        V = w * V + c1*r1*(pbest - X) + c2*r2*(gbest.reshape(-1, 1)-X)
        X = X + V
        X_log.append(X)
        obj = objective(X)
        obj_log.append(obj)
        pbest[:, (pbest_obj >= obj)] = X[:, (pbest_obj >= obj)]
        pbest_obj = np.array([pbest_obj, obj]).min(axis=0)
        gbest = pbest[:, pbest_obj.argmin()]
        gbest_obj = pbest_obj.min()

    return X_log,obj_log


def test_particle_optimizer():
    def banana_function(x):
        return (1-x[0])**2+100*(x[1]-x[0]**2)**2

    def quadratic_function(x):
        return (x[2]-6)**2

    # Create particles
    n_particles = 20
    np.random.seed(100)
    start_position = np.random.rand(3, n_particles) * 5
    start_velocities = np.random.randn(3, n_particles) * 0.1

    simple_value, _ = particle_optimizer(
        quadratic_function, start_position, start_velocities, 5000)
    banana_value, _ = particle_optimizer(
        banana_function, start_position[:2], start_velocities[:2], 5000)

    assert simple_value[2] == 6
    assert (banana_value-np.ones(2) <= 0.001).all()


if __name__ == "__main__":
    test_particle_optimizer()
