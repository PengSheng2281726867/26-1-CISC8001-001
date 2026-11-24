"""Expose simulation geometry for the privileged waypoint diagnostic."""
from phase_simulator import observe, serve_stdio


def observe_geometry(env):
    lift = env.env
    return dict(observe(env), oracle_cube_center=lift.sim.data.body_xpos[lift.cube_body_id].copy())


if __name__ == "__main__":
    serve_stdio(observation_fields=observe_geometry)
