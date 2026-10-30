import os

import numpy as np
os.environ.setdefault('PYGAME_HIDE_SUPPORT_PROMPT', '1')
import pygame

class RCInput:
    """USB RC transmitter (EdgeTX/OpenTX "USB Joystick" mode) read through pygame.

    Defaults match an EdgeTX radio (e.g. RadioMaster Pocket) in Classic joystick
    mode with the AETR channel order: CH1 aileron, CH2 elevator, CH3 throttle,
    CH4 rudder on joystick axes 0 to 3. Run `python -m gym_pybullet_drones.utils.RCInput`
    to print the raw axes and check `axes` and `invert` against your sticks.

    """

    ################################################################################

    def __init__(self,
                 index: int=0,
                 axes=(0, 1, 3, 2),
                 invert=(False, False, False, False),
                 deadband: float=0.02
                 ):
        """RCInput __init__ method.

        Parameters
        ----------
        index : int, optional
            The pygame joystick index of the radio.
        axes : tuple, optional
            Joystick axis indices of (roll, pitch, yaw, throttle).
        invert : tuple, optional
            Whether to flip the sign of each of (roll, pitch, yaw, throttle).
        deadband : float, optional
            Stick values below this magnitude are zeroed (not applied to throttle).

        """
        pygame.init()    # event.pump(), which refreshes the axes, needs the video subsystem too
        if pygame.joystick.get_count() <= index:
            raise RuntimeError("in RCInput.__init__(), no joystick at index {:d}, is the radio plugged in and set to USB Joystick?".format(index))
        self.JOYSTICK = pygame.joystick.Joystick(index)
        self.AXES = axes
        self.SIGNS = np.where(invert, -1., 1.)
        self.DEADBAND = deadband

    ################################################################################

    def read(self):
        """Reads the sticks.

        Returns
        -------
        ndarray
            (4,)-shaped array of floats: roll, pitch, yaw in [-1, 1] (right, forward, right
            are positive), throttle in [0, 1].

        """
        pygame.event.pump()
        sticks = self.SIGNS * np.array([self.JOYSTICK.get_axis(i) for i in self.AXES])
        sticks[:3] = np.where(np.abs(sticks[:3]) < self.DEADBAND, 0., sticks[:3])
        sticks[3] = (sticks[3] + 1) / 2
        return sticks

if __name__ == "__main__":
    #### Axis monitor: move one stick at a time to find its axis and sign
    rc = RCInput()
    print(rc.JOYSTICK.get_name(), "-", rc.JOYSTICK.get_numaxes(), "axes, Ctrl+C to quit")
    while True:
        pygame.event.pump()
        print("  ".join("{:d}:{:+.2f}".format(i, rc.JOYSTICK.get_axis(i)) for i in range(rc.JOYSTICK.get_numaxes())), end="\r")
        pygame.time.wait(50)
