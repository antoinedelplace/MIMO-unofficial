import torch
import copy
import numpy as np
import pytorch3d.transforms as pt3d

import matplotlib.pyplot as plt
from matplotlib import animation

SMPL_bones = [
    {"name": "m_avg_Pelvis", "x": 3.133900463581085e-05 , "y": -0.2356526404619217 , "z": 0.01718907244503498 },
    {"name": "m_avg_L_Hip", "x": 0.05390536040067673 , "y": -0.3188248574733734 , "z": -0.0030965283513069153 },
    {"name": "m_avg_R_Hip", "x": -0.058486707508563995 , "y": -0.3263060748577118 , "z": -0.00023084506392478943 },
    {"name": "m_avg_Spine1", "x": 0.00391986221075058 , "y": -0.11295263469219208 , "z": -0.012817669659852982 },
    {"name": "m_avg_L_Knee", "x": 0.0966343879699707 , "y": -0.692694365978241 , "z": -0.0017415557522326708 },
    {"name": "m_avg_R_Knee", "x": -0.09811581671237946 , "y": -0.6903712749481201 , "z": -0.006651695817708969 },
    {"name": "m_avg_Spine2", "x": 0.006947551853954792 , "y": 0.022616088390350342 , "z": 0.015981821343302727 },
    {"name": "m_avg_L_Ankle", "x": 0.08292833715677261 , "y": -1.09688138961792 , "z": -0.035464271903038025 },
    {"name": "m_avg_R_Ankle", "x": -0.08056940138339996 , "y": -1.0934457778930664 , "z": -0.03911367058753967 },
    {"name": "m_avg_Spine3", "x": 0.004825130105018616 , "y": 0.07664139568805695 , "z": 0.01876234821975231 },
    {"name": "m_avg_L_Foot", "x": 0.12632450461387634 , "y": -1.1585181951522827 , "z": 0.08561922609806061 },
    {"name": "m_avg_R_Foot", "x": -0.10781009495258331 , "y": -1.1574530601501465 , "z": 0.08794237673282623 },
    {"name": "m_avg_Neck", "x": -0.006190690211951733 , "y": 0.27603384852409363 , "z": -0.00810537301003933 },
    {"name": "m_avg_L_Collar", "x": 0.075787253677845 , "y": 0.18380588293075562 , "z": 0.002111745998263359 },
    {"name": "m_avg_R_Collar", "x": -0.07743699848651886 , "y": 0.18278825283050537 , "z": 0.0007707271724939346 },
    {"name": "m_avg_Head", "x": 0.003776766359806061 , "y": 0.3712044060230255 , "z": 0.04340692609548569 },
    {"name": "m_avg_L_Shoulder", "x": 0.19017186760902405 , "y": 0.23034624755382538 , "z": -0.014895318076014519 },
    {"name": "m_avg_R_Shoulder", "x": -0.18306025862693787 , "y": 0.22942060232162476 , "z": -0.005208417773246765 },
    {"name": "m_avg_L_Elbow", "x": 0.4313494563102722 , "y": 0.21278591454029083 , "z": -0.03530063480138779 },
    {"name": "m_avg_R_Elbow", "x": -0.43106773495674133 , "y": 0.21266445517539978 , "z": -0.0328189842402935 },
    {"name": "m_avg_L_Wrist", "x": 0.6809249520301819 , "y": 0.2216266393661499 , "z": -0.043727461248636246 },
    {"name": "m_avg_R_Wrist", "x": -0.6879209876060486 , "y": 0.2189309298992157 , "z": -0.038220442831516266 },
    {"name": "m_avg_L_Hand", "x": 0.7602951526641846 , "y": 0.21127495169639587 , "z": -0.05755671113729477 },
    {"name": "m_avg_R_Hand", "x": -0.7704770565032959 , "y": 0.21136067807674408 , "z": -0.04758597910404205 }
]

SMPL_hierarchy = np.array([[0, 1],
             [0, 2],
             [0, 3],
             [1, 4],
             [2, 5],
             [3, 6],
             [4, 7],
             [5, 8],
             [6, 9],
             [7, 10],
             [8, 11],
             [9, 12],
             [9, 13],
             [9, 14],
             [12, 15],
             [13, 16],
             [14, 17],
             [16, 18],
             [17, 19],
             [18, 20],
             [19, 21],
             [20, 22],
             [21, 23]])


class Object3D():
    def __init__(self, name, parent=None):
        self.name = name
        self.parent = None
        if parent is not None :
            parent.add(self)
        self.parent = parent
        self.children = set()

        # 4x4 Matrices representing the 3D local and global transforms
        self.local_matrix = torch.eye(4)
        self.global_matrix = torch.eye(4)
        self.recompute_global_matrix()
      
    def apply_to_all_children_recursively(self, function):
        function(self)
        for child in self.children:
            child.apply_to_all_children_recursively(function)

    def local_to_global(self, position):
        return (self.global_matrix @ torch.cat([position, torch.ones(1)]))[:3]

    def global_to_local(self, position):
        return (torch.inverse(self.global_matrix) @ torch.cat([position, torch.ones(1)]))[:3]

    def attach(self, child_object):
        """
        Parents child_object to this object, while keeping child_object's global TRS
        """
        self.add(child_object)

        # Make sure the new global TRS is equal to the old global TRS,
        # by changing the local TRS of the child object
        old_global_matrix = child_object.global_matrix
        child_object.local_matrix = torch.inverse(self.global_matrix) @ old_global_matrix

        child_object.recompute_global_matrix(hierarchy=True)

    def detach(self, child_object):
        """
        Unparents child_object to this object, while keeping child_object's global TRS
        """
        self.children.remove(child_object)
        child_object.parent = None

        # Make sure the new global TRS is equal to the old global TRS,
        # by changing the local TRS of the child object
        old_global_matrix = child_object.global_matrix
        child_object.local_matrix = old_global_matrix

    def add(self, child_object):
        """
        Parents child_object to this object, but child_object's local TRS stays the same
        """
        if child_object.parent is not None :
            child_object.parent.remove(child_object)
        child_object.parent = self
        self.children.add(child_object)

    def remove(self, child_object):
        self.children.remove(child_object)

    def get_all_child_objects(self):
        """
        Gets a list of all objects contained in the scene, whether direct children or deep in the hierarchy.
        """
        all_objects = set()
        for child in self.children :
            all_objects.add(child)
            all_objects = all_objects.union(child.get_all_child_objects())
        return all_objects

    def recompute_global_matrix(self, hierarchy=True):
        """
        Triggers a recomputing of the global matrix, based on self.parent.global_matrix and self.local_matrix.

        Params :
          - hierarchy (bool) : if True, recursively triggers recompute_global_matrix for each child.
        """
        if self.parent is None :
            self.global_matrix = self.local_matrix
            return

        self.global_matrix = self.parent.global_matrix @ self.local_matrix

        if hierarchy :
            for child in self.children :
                child.recompute_global_matrix(hierarchy)

    def get_global_position(self):
        return self.global_matrix[:3, 3]

    def get_local_position(self):
        return self.local_matrix[:3, 3]

    def set_local_position(self, position):
        self.local_matrix[:3, 3] = position
        self.recompute_global_matrix(hierarchy=True)

    def get_global_rotation_quaternion(self):
        """
        Returns : torch.Tensor of shape (4), xyzw quaternion
        """
        normalized_matrix = self.global_matrix[:3, :3].clone() / self.get_global_scale().repeat(3,1)
        return pt3d.matrix_to_quaternion(normalized_matrix)

    def get_global_rotation_axisangle(self):
        """
        Returns : torch.Tensor of shape (3), xyz axis angle
        """
        normalized_matrix = self.global_matrix[:3, :3].clone() / self.get_global_scale().repeat(3,1)
        return pt3d.rotation_conversions.matrix_to_axis_angle(normalized_matrix)

    def get_global_rotation_euler(self, order="xyz"):
        """
        Params :
          - order : string composed of x, y, z (for extrinsic euler angles)
                                    or X, Y, Z (for intrinsic euler angles)
        """
        normalized_matrix = self.global_matrix[:3, :3].clone() / self.get_global_scale().repeat(3,1)

        if order.lower() == order:
            return pt3d.matrix_to_euler_angles(normalized_matrix, order.upper()[::-1]).flip(0)
        elif order.upper() == order:
            return pt3d.matrix_to_euler_angles(normalized_matrix, order)
        else:
            raise Exception("Extrinsic (lowercase xyz) and intrinsic (uppercase XYZ) euler angles cannot be mixed.")

    def get_local_rotation_quaternion(self):
        """
        Returns : torch.Tensor of shape (4), xyzw quaternion
        """
        normalized_matrix = self.local_matrix[:3, :3].clone() / self.get_local_scale().repeat(3,1)
        return pt3d.matrix_to_quaternion(normalized_matrix)

    def get_local_rotation_axisangle(self):
        """
        Returns : torch.Tensor of shape (3), xyz axis angle
        """
        normalized_matrix = self.local_matrix[:3, :3].clone() / self.get_local_scale().repeat(3,1)
        return pt3d.rotation_conversions.matrix_to_axis_angle(normalized_matrix)

    def get_local_rotation_euler(self, order="xyz"):
        """
        Params :
          - order : string composed of x, y, z (for extrinsic euler angles)
                                    or X, Y, Z (for intrinsic euler angles)
        """
        normalized_matrix = self.local_matrix[:3, :3].clone() / self.get_local_scale().repeat(3,1)
        if order.lower() == order:
            return pt3d.matrix_to_euler_angles(normalized_matrix, order.upper()[::-1]).flip(0)
        elif order.upper() == order:
            return pt3d.matrix_to_euler_angles(normalized_matrix, order)
        else:
            raise Exception("Extrinsic (lowercase xyz) and intrinsic (uppercase XYZ) euler angles cannot be mixed.")

    def set_local_rotation_euler(self, rotation, order="xyz"):
        """
        Params :
          - rotation : torch.Tensor() of shape (3), Euler angles
          - order : string composed of x, y, z (for extrinsic euler angles)
                                    or X, Y, Z (for intrinsic euler angles)
        """
        if order.lower() == order:
            rotation = pt3d.euler_angles_to_matrix(rotation.flip(0), order.upper()[::-1])
        elif order.upper() == order:
            rotation = pt3d.euler_angles_to_matrix(rotation, order)
        else:
            raise Exception("Extrinsic (lowercase xyz) and intrinsic (uppercase XYZ) euler angles cannot be mixed.")

        self.set_local_rotation_matrix(rotation)

    def set_local_rotation_axisangle(self, rotation):
        rotation = pt3d.axis_angle_to_matrix(rotation)
        self.set_local_rotation_matrix(rotation)

    def set_local_rotation_quaternion(self, rotation):
        rotation = pt3d.quaternion_to_matrix(rotation)
        self.set_local_rotation_matrix(rotation)
    
    def set_local_rotation_matrix(self, rotation):
        self.local_matrix[:3, :3] = self.get_local_scale().repeat(3,1) * rotation
        self.recompute_global_matrix(hierarchy=True)

    def get_local_scale(self):
        return torch.norm(self.local_matrix[:3, :3], dim=0)

    def get_global_scale(self):
        return torch.norm(self.global_matrix[:3, :3], dim=0)

    def set_local_scale(self, scale):
        self.local_matrix[:3, :3] = self.local_matrix[:3, :3].clone() * scale.repeat(3,1)
        self.recompute_global_matrix(hierarchy=True)

    def apply_local_translation(self, translation):
        self.set_local_position(self.get_local_position() + translation)

    def apply_local_rotation_axisangle(self, axisangle):
        quat = pt3d.rotation_conversions.axis_angle_to_quaternion(axisangle)
        self.apply_local_rotation_quaternion(quat)

    def apply_local_rotation_quaternion(self, quaternion):
        final_quat = pt3d.rotation_conversions.quaternion_multiply(self.get_local_rotation_quaternion(), quaternion)
        self.set_local_rotation_quaternion(final_quat)

    def lookat(self, position, up=torch.Tensor([0, 1, 0]), forward=torch.Tensor([0, 0, 1])):
        """
        Sets the local rotation so that the object's forward axis points towards
        the position (given in global space), and so that the object's up axis
        is in the plane formed by the forward axis and the (0,1,0) world up vector.
        Triggers recompute_global_matrix for each child.

        Params :
          - position : torch.Tensor() of size (3), given in global coordinates
          - up (optional, default torch.Tensor([0, 1, 0])) : local coordinates of up axis
          - forward (optional, default torch.Tensor([0, 0, 1])) : local coordinates of forward axis
        """
        # First, normalize the up and forward vectors
        up = up / torch.linalg.norm(up)
        forward = forward / torch.linalg.norm(forward)

        # Then, we compute a (one out of many possible) rotation such that the
        # object's forward axis points to the given position
        direction = self.global_to_local(position)
        direction = direction / torch.linalg.norm(direction)
        axis = torch.cross(direction, forward)
        axis = axis / torch.linalg.norm(axis)
        angle = -torch.acos(torch.dot(direction, forward))
        self.apply_local_rotation_axisangle(axis*angle)

        # Second, we need to rotate the object around its forward axis until the
        # up axis is in the plane formed by forward and the world's up vector
        y_local = self.global_to_local(self.get_global_position() + torch.Tensor([0, 1, 0]))
        y_local = y_local / torch.linalg.norm(y_local)
        plane_normal = torch.cross(y_local, forward)
        plane_normal = plane_normal / torch.linalg.norm(plane_normal)
        up_projected = up - plane_normal * torch.dot(plane_normal, up)

        angle = 0
        if torch.linalg.norm(up_projected) > 0 :
            # In that case, the up_projected is not in the plane formed by forward and y_local
            # So we project it on the plane and compute the rotation needed
            up_projected = up_projected / torch.linalg.norm(up_projected)
            sign = 1 if torch.dot(torch.cross(up, up_projected), forward) > 0 else -1
            angle = sign * torch.acos(torch.dot(up, up_projected))
            if torch.dot(y_local, up_projected) < 0 :
                # It will end up towards 'negative y' if we forget this
                angle += np.pi
        else :
            # In that case, the up_projected is already in the plane formed by forward and y_local
            # We just have to point it to positive y and not negative y
            angle = np.pi if torch.dot(up, y_local) < 0 else 0

        self.apply_local_rotation_axisangle(forward*angle)


class Skeleton(Object3D):
    def __init__(self, name, parent=None, bones=[]):
        super().__init__(name, parent)
        self.bone_list = copy.copy(bones)       # We need to copy the list because construct_from_zero_pose modifies it
    
    def construct_from_zero_pose(self, bones_dict, bones_hierarchy):
        """
        Constructs the skeleton from the given bones dict and hierarchy.
        It is assumed that the given inputs will correspond to the zero pose
        (pose at which the local rotation of each bone is zero).
        """
        for bone_id, bone_data in enumerate(bones_dict):
            bone = Bone(bone_data["name"], self)
            self.bone_list.append(bone)
            bone.set_local_position(torch.Tensor([bone_data["x"], bone_data["y"], bone_data["z"]]))

        for bone_parent, bone_child in bones_hierarchy :
            self.bone_list[bone_parent].attach(self.bone_list[bone_child])

    def set_pose_axis_angle(self, rotations):
        for bone_id, bone in enumerate(self.bone_list):
            bone.set_local_rotation_axisangle(rotations[3*bone_id:3*(bone_id+1)])

    def get_global_position_joints(self):
        output = torch.zeros(len(self.bone_list)*3)
        for bone_id, bone in enumerate(self.bone_list):
            output[bone_id*3:(bone_id+1)*3] = bone.get_global_position()
        return output
    
    def get_local_position_joints(self):
        output = torch.zeros(len(self.bone_list)*3)
        for bone_id, bone in enumerate(self.bone_list):
            output[bone_id*3:(bone_id+1)*3] = bone.get_local_position()
        return output

    def set_lookat(self):
        """
        Change bone orientations to look at the only child if there is only one child
        """
        for bone_id, bone in enumerate(self.bone_list):
            if len(bone.children) == 1:
                child = next(iter(bone.children))
                bone.detach(child)
                bone.lookat(child.get_global_position())
                bone.attach(child)
            bone.local_tpose_matrix  = torch.clone(bone.local_matrix)
            bone.global_tpose_matrix = torch.clone(bone.global_matrix)
    
    def set_tpose(self):
        """
        Set joint rotations to match saved tpose
        """
        root = next(iter(self.children))
        root.apply_to_all_children_recursively(lambda x: x.set_tpose())
    
    def get_bone2idx(self):
        """
        Return a dictionary to map bones to their index in the bone_list list
        """
        bone2idx = dict()
        
        for bone_id, bone in enumerate(self.bone_list):
            bone2idx[bone] = bone_id
        
        return bone2idx
            
    def set_pose_from_skeleton_source(self, skeleton_source):
        """
        Retarget pose from skeleton source to this skeleton.
        WARNING : Assumes that :
        - the bones are the same (same length, same kinematic chain)
        - the bone_list is ordered in the same manner
        """
        bone2idx = self.get_bone2idx()

        root = next(iter(self.children))
        root.apply_to_all_children_recursively(lambda x: x.set_global_rotation_from_tpose(skeleton_source.bone_list[bone2idx[x]].get_global_rotation_from_tpose()))

class Bone(Object3D):
    def __init__(self, name, parent):
        super().__init__(name, parent)
        
        # 4x4 Matrices representing the 3D local transforms in T-pose
        # Tpose is by default in zero pose
        self.local_tpose_matrix  = torch.clone(self.local_matrix) 
        self.global_tpose_matrix = torch.clone(self.global_matrix)
    
    def set_tpose(self):
        self.local_matrix  = torch.clone(self.local_tpose_matrix)
        self.global_matrix = torch.clone(self.global_tpose_matrix)
    
    def get_global_rotation_from_tpose(self):
        """
        Extracts the global rotation that, when applied to the T-pose, yields the same pose as the currrent pose.
        """
        local_rot = self.local_matrix[:3, :3] / self.get_global_scale().repeat(3, 1)
        tpose_local = self.local_tpose_matrix[:3, :3] / self.get_global_scale().repeat(3, 1)
        
        local_rot_from_tpose = tpose_local.T @ local_rot
        
        glob2loc = self.global_tpose_matrix[:3, :3] / self.get_global_scale().repeat(3,1)
        
        # Apply conjugate (recall transpose is inverse in SO(3))
        global_rot = glob2loc @ local_rot_from_tpose @ glob2loc.T
        
        return global_rot
    
    def set_global_rotation_from_tpose(self, rotation):
        """
        Changes the object's local rotation by applying the given global rotation to the T-pose.
        Triggers recompute_global_matrix for each child.

        Params :
          - rotation : torch.Tensor() of size (3, 3), rotation matrix
        """
        tpose_local = self.local_tpose_matrix[:3, :3] / self.get_global_scale().repeat(3, 1)
        
        glob2loc = self.global_tpose_matrix[:3, :3] / self.get_global_scale().repeat(3, 1)
        
        # Apply conjugate (recall transpose is inverse in SO(3))
        local_to_apply = glob2loc.T @ rotation @ glob2loc
        
        self.set_local_rotation_matrix(tpose_local @ local_to_apply)


def get_chains_from_bones_hierarchy(bones_hierarchy):
    """
    return chains: list of list
    if [0, 5, 9] is in it, it means 0 is the parent of 5 which is the parent of 9
    """
    is_in_chains = [0]*len(bones_hierarchy)
    chains = []

    for i in range(0, len(bones_hierarchy)):
        if not is_in_chains[i]:
            is_in_chains[i] = 1
            chains.append(bones_hierarchy[i].tolist())
            j = 0
            while j < len(bones_hierarchy):
                if not is_in_chains[j] and bones_hierarchy[j, 0] == chains[-1][-1]:
                    is_in_chains[j] = 1
                    chains[-1].append(bones_hierarchy[j, 1])
                    j = 0
                else:
                    j += 1
    return chains

    
def points_animation_linked_3d(points,
                               chains,
                               joint_labels=None,
                               fps=24,
                               show=False,
                               save_path=None):
    """
    Inputs :
        - points: (N, 3)
        - chains: list of list - if [0, 5, 9] is in it, it means 0 is the parent of 5 which is the parent of 9
        - joint_labels: list of joint labels (list of strings)
        - fps : frame per second (default 24)
        - show: boolean
        - save_path: path to save the animation (.gif), None if no saving
    """
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    txt = fig.suptitle('num=')

    x = points[:, :, 0]
    y = points[:, :, 1]
    z = points[:, :, 2]

    def update_points(num, x, y, z, chains):
        txt.set_text('num={:d}'.format(num))

        ax.cla()
        for index, chain in enumerate(chains):
            x_chain = np.asarray([x[num][i] for i in chain])
            y_chain = np.asarray([y[num][i] for i in chain])
            z_chain = np.asarray([z[num][i] for i in chain])

            ax.plot(x_chain, y_chain, z_chain, marker=".", markersize=10)

            if joint_labels:
                for i, label in enumerate(joint_labels):
                    ax.text(x[num][i], y[num][i], z[num][i], label)

        ax.set_xlim([np.min(x), np.max(x)])
        ax.set_ylim([np.min(y), np.max(y)])
        ax.set_zlim([np.min(z), np.max(z)])

        ax.azim = -90
        ax.elev = -90

        ax.grid(False)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_zticks([])
        ax.xaxis.pane.fill = False
        ax.yaxis.pane.fill = False
        ax.zaxis.pane.fill = False
        ax.xaxis.pane.set_edgecolor('w')
        ax.yaxis.pane.set_edgecolor('w')
        ax.zaxis.pane.set_edgecolor('w')

    ani = animation.FuncAnimation(fig, update_points, frames=np.shape(points)[0], fargs=(x, y, z, chains), interval=fps, init_func=ax.cla)

    if show: plt.show()
    if save_path: ani.save(save_path, writer=animation.FFMpegWriter(fps=fps))

    plt.close(fig)