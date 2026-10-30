"""Fly the racer in real time with a USB RC transmitter, in angle mode.

Plug the radio in, pick "USB Joystick (HID)" on it, then run

    $ python rc.py

Lower the throttle to arm. Press `r` in the PyBullet window to reset (and re-arm).
Check your stick mapping first with `python -m gym_pybullet_drones.utils.RCInput`.

"""
import time
import argparse

import numpy as np
import pybullet as p

from gym_pybullet_drones.utils.enums import DroneModel, Physics
from gym_pybullet_drones.envs.CtrlAviary import CtrlAviary
from gym_pybullet_drones.control.AngleControl import AngleControl
from gym_pybullet_drones.utils.RCInput import RCInput
from gym_pybullet_drones.utils.utils import sync

DEFAULT_PHYSICS = Physics("pyb")
DEFAULT_FREQ_HZ = 240
DEFAULT_MAX_ANGLE_DEG = 45
DEFAULT_MAX_YAW_RATE_DEG = 200

def run(physics=DEFAULT_PHYSICS,
        freq_hz=DEFAULT_FREQ_HZ,
        max_angle_deg=DEFAULT_MAX_ANGLE_DEG,
        max_yaw_rate_deg=DEFAULT_MAX_YAW_RATE_DEG
        ):
    rc = RCInput()
    env = CtrlAviary(drone_model=DroneModel.RACE, physics=physics, pyb_freq=freq_hz, ctrl_freq=freq_hz, gui=True)
    ctrl = AngleControl(drone_model=DroneModel.RACE)
    obs, _ = env.reset()
    armed = False
    START = time.time()
    i = 0
    while p.isConnected(env.getPyBulletClient()):
        roll, pitch, yaw, throttle = rc.read()
        #### Same safety as a real flight controller: no thrust until the throttle has been low
        armed = armed or throttle < 0.05
        rpm = ctrl.computeControlFromState(control_timestep=env.CTRL_TIMESTEP,
                                           state=obs[0],
                                           throttle=throttle,
                                           target_rp=np.radians(max_angle_deg) * np.array([roll, pitch]),
                                           target_yaw_rate=-np.radians(max_yaw_rate_deg) * yaw  # stick right = clockwise = -z
                                           ) if armed else np.zeros(4)
        obs, *_ = env.step(rpm.reshape(1, 4))
        p.resetDebugVisualizerCamera(cameraDistance=1.5, cameraYaw=np.degrees(obs[0][9]) - 90, cameraPitch=-20,
                                     cameraTargetPosition=obs[0][0:3], physicsClientId=env.getPyBulletClient())
        if ord('r') in p.getKeyboardEvents(physicsClientId=env.getPyBulletClient()):
            obs, _ = env.reset()
            armed = False
        sync(i, START, env.CTRL_TIMESTEP)
        i += 1
    env.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Fly the racer in real time with a USB RC transmitter')
    parser.add_argument('--physics',          default=DEFAULT_PHYSICS,          type=Physics, help='Physics updates (default: PYB)', metavar='', choices=Physics)
    parser.add_argument('--freq_hz',          default=DEFAULT_FREQ_HZ,          type=int,     help='Simulation and control frequency in Hz (default: 240)', metavar='')
    parser.add_argument('--max_angle_deg',    default=DEFAULT_MAX_ANGLE_DEG,    type=float,   help='Roll/pitch at full stick in degrees (default: 45)', metavar='')
    parser.add_argument('--max_yaw_rate_deg', default=DEFAULT_MAX_YAW_RATE_DEG, type=float,   help='Yaw rate at full stick in deg/s (default: 200)', metavar='')
    ARGS = parser.parse_args()
    run(**vars(ARGS))
