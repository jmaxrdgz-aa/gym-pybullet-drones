import numpy as np

from gym_pybullet_drones.control.AngleControl import AngleControl
from gym_pybullet_drones.envs.CtrlAviary import CtrlAviary
from gym_pybullet_drones.utils.enums import DroneModel


def _fly(target_rp, duration_sec=2, ctrl_freq=240):
    env = CtrlAviary(drone_model=DroneModel.RACE, initial_xyzs=np.array([[0, 0, 1.]]),
                     pyb_freq=ctrl_freq, ctrl_freq=ctrl_freq, gui=False)
    ctrl = AngleControl(drone_model=DroneModel.RACE)
    obs, _ = env.reset()
    for _ in range(duration_sec*ctrl_freq):
        rpm = ctrl.computeControlFromState(env.CTRL_TIMESTEP, obs[0], ctrl.HOVER_THROTTLE, target_rp)
        obs, *_ = env.step(rpm.reshape(1, 4))
    env.close()
    return obs[0]


def test_angle_hover():
    state = _fly(np.zeros(2))
    assert np.allclose(state[7:9], 0, atol=0.02)    # level
    assert abs(state[2] - 1.) < 0.1                 # hover throttle holds altitude


def test_angle_tracks_roll_and_pitch():
    state = _fly(np.array([0.2, -0.1]), duration_sec=1)
    assert np.allclose(state[7:9], [0.2, -0.1], atol=0.02)
    assert state[11] < 0 and state[10] < 0          # roll right -> drift -y, pitch nose down -> drift -x
