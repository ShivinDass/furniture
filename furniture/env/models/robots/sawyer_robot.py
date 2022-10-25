import numpy as np

from ...mjcf_utils import array_to_string, xml_path_completion
from .robot import Robot


class Sawyer(Robot):
    """Sawyer is a witty single-arm robot designed by Rethink Robotics."""

    def __init__(
        self, use_torque=False, xml_path="robots/sawyer/robot.xml",
    ):
        if use_torque:
            xml_path = "robots/sawyer/robot_torque.xml"
        super().__init__(xml_path_completion(xml_path))

        self.bottom_offset = np.array([0, 0, -0.913])

        # self._init_qpos = np.array([-0.23429241 - 0.4, -1.1364233, 0.336434, 2.18, -0.16150611, 0.31906261 + 0.2, 0])
        #self._init_qpos = np.array([-0.28, -0.60, 0.00, 1.86, 0.00, 0.3, 1.57])
        #self._init_qpos = np.array([-0.21482441, -0.48111927, -0.04771009,  1.9549736 , -1.02385222,
        #0.09436399,  1.63076707])
        #self._init_qpos = np.array([-0.13471455, -0.61623393,  0.16102363,  1.93952715, -0.92682511,  0.32211509, 1.08598543])
        #self._init_qpos = np.array([0.07779019, -0.55636811, -0.03968154, 1.9320403, -1.05190032, 0.13437532, 1.52006013])
        #self._init_qpos = np.array([0.56074377, -0.20968404, -0.52782309,  1.9338516,  -1.11261246, -0.43942591, 1.72248362])
        #self._init_qpos = np.array([0.2367299,  -0.2482205,  -0.43201701,  1.95870625, -1.28029764, -0.26473828, 1.62392545])
        #self._init_qpos = np.array([0.261286,   -0.31716141, -0.29484426,  1.65314806, -2.02185309, -0.22783806, 2.37782449])
        self._init_qpos = np.array([-0.05917441, -0.49122831, -0.07468635,  2.16852352, -1.92482301, -0.19426369, 2.08737853])

        self._model_name = "sawyer"

    def set_base_xpos(self, pos):
        """
        Places the robot on position @pos.
        """
        node = self.worldbody.find("./body[@name='base']")
        node.set("pos", array_to_string(pos - self.bottom_offset))

    def set_base_xquat(self, quat):
        """
        Places the robot on position @quat.
        """
        node = self.worldbody.find("./body[@name='base']")
        node.set("quat", array_to_string(quat))

    def set_joint_damping(
        self, damping=np.array((0.1, 0.1, 0.1, 0.1, 0.1, 0.01, 0.01))
    ):
        """Set joint damping """

        body = self._base_body
        for i in range(len(self._link_body)):
            body = body.find("./body[@name='{}']".format(self._link_body[i]))
            joint = body.find("./joint[@name='{}']".format(self._joints[i]))
            joint.set("damping", array_to_string(np.array([damping[i]])))

    def set_joint_frictionloss(
        self, friction=np.array((0.1, 0.1, 0.1, 0.1, 0.1, 0.01, 0.01))
    ):
        body = self._base_body
        for i in range(len(self._link_body)):
            body = body.find("./body[@name='{}']".format(self._link_body[i]))
            joint = body.find("./joint[@name='{}']".format(self._joints[i]))
            joint.set("frictionloss", array_to_string(np.array([friction[i]])))

    @property
    def dof(self):
        return 7

    @property
    def joints(self):
        return ["right_j{}".format(x) for x in range(7)]

    @property
    def init_qpos(self):
        # return np.array([0, -1.18, 0.00, 2.18, 0.00, 0.57, 3.3161])
        # 0: base, 1: 1st elbow, 3: 2nd elbow 5: 3rd elbow
        return self._init_qpos

    @init_qpos.setter
    def init_qpos(self, init_qpos):
        self._init_qpos = init_qpos

    @property
    def contact_geoms(self):
        return ["right_l{}_collision".format(x) for x in range(2, 7)]

    # @property
    # def _base_body(self):
    #     node = self.worldbody.find("./body[@name='base']")
    #     body = node.find("./body[@name='right_arm_base_link']")
    #     return body

    @property
    def _link_body(self):
        return [
            "right_l0",
            "right_l1",
            "right_l2",
            "right_l3",
            "right_l4",
            "right_l5",
            "right_l6",
        ]

    @property
    def _joints(self):
        return [
            "right_j0",
            "right_j1",
            "right_j2",
            "right_j3",
            "right_j4",
            "right_j5",
            "right_j6",
        ]

    @property
    def contact_geoms(self):
        return [
            "pedestal_collision",
            "right_arm_base_link_collision",
            "right_l0_collision",
            "head_collision",
            "screen_collision",
            "right_l1_collision",
            "right_l1_collision",
            "right_l2_collision",
            "right_l2_collision",
            "right_l3_collision",
            "right_l3_collision",
            "right_l4_collision",
            "right_l4_collision",
            "right_l5_collision",
            "right_l5_collision",
            "right_l6_collision",
            "right_l6_collision",
            "right_l4_2_collision",
            "right_l2_2_collision",
            "right_l1_2_collision",
        ]
