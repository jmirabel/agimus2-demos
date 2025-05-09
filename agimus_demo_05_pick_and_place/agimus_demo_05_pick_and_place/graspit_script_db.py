import csv
import dataclasses
import numpy as np
import numpy.typing as npt
import typing as T
import eigenpy
import pinocchio


@dataclasses.dataclass
class GraspDefinition:
    translation: npt.NDArray[float]
    rotation: eigenpy.Quaternion
    approach_direction: npt.NDArray[float]

    @property
    def pose_list(self):
        return self.translation.tolist() + self.rotation.coeffs().tolist()

    @property
    def se3(self):
        return pinocchio.SE3(self.rotation, self.translation)
    
    @se3.setter
    def se3(self, jMg: pinocchio.SE3):
        self.translation = jMg.translation
        self.rotation = eigenpy.Quaternion(jMg.rotation)


def load_from_csv(filename: str) -> T.List[GraspDefinition]:
    grasps = list()
    rot = eigenpy.Quaternion(np.array(
        [
            [0, 0, 1],
            [1, 0, 0],
            [0, 1, 0],
        ],
        dtype=np.float64,
    )).inverse()
    with open(filename, "r", newline="") as f:
        reader = csv.reader(f, skipinitialspace=True, delimiter=",")
        # Skip headers
        next(reader)
        for row in reader:
            assert len(row) == 10
            frow = list(map(float, row))
            q = eigenpy.Quaternion(np.array(frow[3:7])).normalized()
            grasps.append(
                GraspDefinition(
                    translation=np.array(frow[:3]),
                    rotation=q, # * rot,
                    approach_direction=np.array(frow[7:10]),
                )
            )
    return grasps

def shift_grasps(robot, gripper, graspit_link) -> pinocchio.SE3:
    joint, jMg = robot.getGripperPositionInJoint(gripper)
    jMg = pinocchio.XYZQUATToSE3(jMg)
    wMl = pinocchio.XYZQUATToSE3(robot.getLinkPosition(graspit_link))
    wMl = wMl * pinocchio.SE3(np.eye(3), np.array([0,0,0.01]))
    wMj = pinocchio.XYZQUATToSE3(robot.getJointPosition(joint))
    
    # handle position using graspit link: oMH
    # handle position using hpp link: oMh
    # jMo = jMg * hMo = jMl * HMo
    # oMh = oMH * lMj * jMg
    return wMl.inverse() * wMj * jMg
