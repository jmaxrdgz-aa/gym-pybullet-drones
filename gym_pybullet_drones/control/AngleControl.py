import numpy as np
import pybullet as p

from gym_pybullet_drones.control.BaseControl import BaseControl
from gym_pybullet_drones.utils.enums import DroneModel

class AngleControl(BaseControl):
    """Angle mode (self-levelling) control class for the racer drone.

    Takes RC-style inputs, throttle in [0, 1], target roll and pitch angles
    in rad, and a target yaw rate in rad/s, as in Betaflight's angle mode.
    A cascaded P angle loop feeds a P body-rate loop whose torques are
    scaled by the URDF inertia, then the motor allocation is inverted
    into RPMs through `kf` and `km`.

    """

    ################################################################################

    def __init__(self,
                 drone_model: DroneModel,
                 g: float=9.8
                 ):
        """Angle control class __init__ method.

        Parameters
        ----------
        drone_model : DroneModel
            The type of drone to control (detailed in an .urdf file in folder `assets`).
        g : float, optional
            The gravitational acceleration in m/s^2.

        """
        super().__init__(drone_model=drone_model, g=g)
        if self.DRONE_MODEL != DroneModel.RACE:
            raise ValueError("in AngleControl.__init__(), AngleControl requires DroneModel.RACE")
        self.J = np.array([self._getURDFParameter(i) for i in ['ixx', 'iyy', 'izz']])
        self.MAX_THRUST = self._getURDFParameter('thrust2weight') * self.GRAVITY
        self.MAX_RPM = np.sqrt(self.MAX_THRUST / (4*self.KF))
        self.HOVER_THROTTLE = self.GRAVITY / self.MAX_THRUST
        self.P_COEFF_ANGLE = np.array([6., 6.])         # 1/s, angle error -> roll/pitch rate setpoint
        self.P_COEFF_RATE = np.array([30., 30., 10.])   # 1/s, rate error -> angular acceleration
        #### Allocation: [thrust, x_torque, y_torque, z_torque] = A @ rpm**2
        #### Signs follow BaseAviary._dynamics() for DroneModel.RACE
        # shortcut: arm*sqrt(.5) lever as in _dynamics(); racer.urdf props sit at (.085, .0675), refine if the rate loop needs it
        L = self._getURDFParameter('arm') / np.sqrt(2)
        A = np.array([[1, 1, 1, 1],
                      [L, L, -L, -L],
                      [-L, L, L, -L],
                      [self.KM/self.KF, -self.KM/self.KF, self.KM/self.KF, -self.KM/self.KF]
                      ]) * self.KF
        self.A_INV = np.linalg.inv(A)

    ################################################################################

    def computeControlFromState(self,
                                control_timestep,
                                state,
                                throttle,
                                target_rp=np.zeros(2),
                                target_yaw_rate=0.
                                ):
        """Interface method using `computeControl`.

        It can be used to compute a control action directly from the value of key "state"
        in the `obs` returned by a call to BaseAviary.step().

        Parameters
        ----------
        control_timestep : float
            The time step at which control is computed.
        state : ndarray
            (20,)-shaped array of floats containing the current state of the drone.
        throttle : float
            Collective throttle in [0, 1], fraction of the maximum thrust.
        target_rp : ndarray, optional
            (2,1)-shaped array of floats containing the desired roll and pitch, in rad.
        target_yaw_rate : float, optional
            The desired yaw rate, in rad/s.

        """
        return self.computeControl(control_timestep=control_timestep,
                                   cur_quat=state[3:7],
                                   cur_ang_vel=state[13:16],
                                   throttle=throttle,
                                   target_rp=target_rp,
                                   target_yaw_rate=target_yaw_rate
                                   )

    ################################################################################

    def computeControl(self,
                       control_timestep,
                       cur_quat,
                       cur_ang_vel,
                       throttle,
                       target_rp=np.zeros(2),
                       target_yaw_rate=0.
                       ):
        """Computes the angle mode control action (as RPMs) for a single drone.

        Parameters
        ----------
        control_timestep : float
            The time step at which control is computed (unused, the loops are proportional).
        cur_quat : ndarray
            (4,1)-shaped array of floats containing the current orientation as a quaternion.
        cur_ang_vel : ndarray
            (3,1)-shaped array of floats containing the current angular velocity, in the world frame.
        throttle : float
            Collective throttle in [0, 1], fraction of the maximum thrust.
        target_rp : ndarray, optional
            (2,1)-shaped array of floats containing the desired roll and pitch, in rad.
        target_yaw_rate : float, optional
            The desired yaw rate, in rad/s.

        Returns
        -------
        ndarray
            (4,1)-shaped array of floats containing the RPMs to apply to each of the 4 motors.

        """
        self.control_counter += 1
        cur_rotation = np.array(p.getMatrixFromQuaternion(cur_quat)).reshape(3, 3)
        cur_rpy = np.array(p.getEulerFromQuaternion(cur_quat))
        body_rates = cur_rotation.T @ np.array(cur_ang_vel)
        #### Angle loop -> rate setpoint, rate loop -> torques #####
        target_rates = np.append(self.P_COEFF_ANGLE * (np.array(target_rp) - cur_rpy[:2]), target_yaw_rate)
        torques = self.J * self.P_COEFF_RATE * (target_rates - body_rates) \
                  + np.cross(body_rates, self.J * body_rates)
        thrust = np.clip(throttle, 0., 1.) * self.MAX_THRUST
        rpm_sq = self.A_INV @ np.append(thrust, torques)
        return np.clip(np.sqrt(np.clip(rpm_sq, 0., None)), 0., self.MAX_RPM)
