import numpy as np
import pygame

from gym_pybullet_drones.utils.RCInput import RCInput


class _FakeJoystick:
    def __init__(self, axes):
        self.axes = axes

    def get_axis(self, i):
        return self.axes[i]


def test_rc_input_mapping(monkeypatch):
    monkeypatch.setattr(pygame.joystick, 'get_count', lambda: 1)
    monkeypatch.setattr(pygame.joystick, 'Joystick', lambda i: _FakeJoystick([0.5, 0.01, -1., -0.3]))
    rc = RCInput(invert=(False, False, True, False))    # AETR: ail, ele, thr, rud on axes 0..3
    roll, pitch, yaw, throttle = rc.read()
    assert roll == 0.5
    assert pitch == 0.                                  # inside the deadband
    assert np.isclose(yaw, 0.3)                         # inverted
    assert throttle == 0.                               # stick low -> 0
