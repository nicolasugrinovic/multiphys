import glob
import os
import os.path as osp
import sys
import torch
sys.path.append(os.getcwd())
from stl import mesh
import numpy as np
from uhc.smpllib.smpl_robot import Robot
from lxml import etree
from lxml.etree import Element, SubElement
from uhc.utils.config_utils.copycat_config import Config as CC_Config
from copy import deepcopy
import shutil
from scipy.spatial.transform import Rotation as sRot
from embodiedpose.models.implicit_sdfs import BoxSDF_F, CylinderSDF_F
from embodiedpose.models.implicit_sdfs_np import BoxSDF_N, CylinderSDF_N
from embodiedpose.models.humor.utils.transforms import (compute_world2aligned_mat)
import copy
from pathlib import Path

class SceneRobotMulti(Robot):

    def load_from_skeleton(self, betas=None, gender=[0], scene_and_key=None, obj_info=None, num_agent=1):
        # NOTE: this is called 3 times before going into add_simple_scene() !!!
        super().load_from_skeleton(betas=betas, gender=gender, num_agent=num_agent)

        if 0:
            # here the robot specifications are already created
            fname = "inspect_out/xml_s/Robot_load_from_skeleton.xml"
            self.tree.write(fname, pretty_print=True)

        self.scene_sdfs = []
        self.scene_sdfs_np = []
        # scene_and_key can be None!
        # as the scene_and_key for chi3d is 's' and the file does not exist
        # add_simple_scene() does not load anything
        if not scene_and_key is None:
            # For adding scene through scene_and_key (prox)
            # self.add_scene(scene_and_key)
            # NOTE: this is called after the robot is initialized and after the Mujoco simu is created from the xml
            print("***NOT adding SCENE information***")
            # self.add_simple_scene(scene_and_key)
            # todo: add here the creation of another agent, don't do it outiside this function
            # self.tree exists here
            # self.add_agent(betas, gender)

        if not obj_info is None:
            # For adding scene through obj_info (h36m)
            self.add_obj(obj_info)

        self.scene_sdfs.append(BoxSDF_F(trans=[0, 0, -0.5], orientation=np.eye(3), side_lengths=np.array([1000, 1000, 1])))  ## Adding ground
        self.scene_sdfs_np.append(BoxSDF_N(trans=np.array([0, 0, -0.5]), orientation=np.eye(3), side_lengths=np.array([1000, 1000, 1])))  ## Adding ground
        self.create_voxel_field(span=self.cfg.get('span', 1.8))

    def query_voxel(self, root_pos, root_mat, res=8):
        world2local = compute_world2aligned_mat(root_mat[None,]).float()
        # self.grid_points.min()
        # self.grid_points.max()
        # self.grid_points is (4096, 3) and ranges from -0.9 to 0.9
        new_pts = torch.matmul(self.grid_points, world2local.float()) + root_pos.float()

        # new_pts = torch.matmul(self.grid_points, root_mat.T) + root_pos
        # new_pts = self.grid_points + root_pos

        occ = self.get_sdf_np(new_pts)
        occ = occ.reshape(self.vox_res, self.vox_res, self.vox_res)

        if res == 8:
            occ = -self.pool2d(-occ[None,]).numpy().squeeze()  # sdf is negative. Need to min pool
        else:
            occ = occ.numpy()

        return occ # (16, 16, 16)

    def create_voxel_field(self, fine_res=16, coarse_res=16, span=1.8):
        self.vox_res = fine_res

        size_range = torch.arange(0, fine_res + 1, fine_res / (fine_res - 1)) / (fine_res / span) - span / 2
        x = size_range
        grid_x, grid_y, grid_z = torch.meshgrid(x, x, x)
        self.grid_points = torch.cat([grid_x.reshape(-1, 1), grid_y.reshape(-1, 1), grid_z.reshape(-1, 1)], dim=1).float()
        self.pool2d = torch.nn.MaxPool3d(2, stride=2)

    def get_sdf(self, points, topk=1, return_grad=False):
        points = points.view(-1, 3)
        if len(self.scene_sdfs) > 0:
            dists = []
            dirs = []

            for sdf in self.scene_sdfs:
                if return_grad:
                    points = torch.autograd.Variable(points, requires_grad=True)

                    with torch.enable_grad():
                        dist = sdf(points)
                        loss = (dist - 0).sum()
                        loss.backward()
                        dir = -points.grad / torch.norm(points.grad, dim=1)[:, None]
                        dirs.append(dir)
                else:
                    dist = sdf(points)
                dists.append(dist)

            dists = torch.cat(dists, dim=1)

            if return_grad:
                dirs = torch.cat(dirs, dim=1).reshape(-1, len(self.scene_sdfs), 3)
                dirs = torch.nan_to_num(dirs)

            if len(self.scene_sdfs) < topk:

                vals, locs = torch.topk(dists, k=len(self.scene_sdfs), largest=False)
                vals = torch.cat([vals, torch.ones([vals.shape[0], topk - len(self.scene_sdfs)]) * (-1)], dim=1)
                if return_grad:
                    dirs = torch.cat([dirs[idx][loc][None,] for idx, loc in enumerate(locs)], dim=0)
                    dirs = torch.cat([dirs, torch.ones([vals.shape[0], topk - len(self.scene_sdfs), 3]) * (-1)], dim=1)

            else:
                vals, locs = torch.topk(dists, k=topk, largest=False)
                if return_grad:
                    dirs = torch.cat([dirs[idx][loc][None,] for idx, loc in enumerate(locs)], dim=0)

        else:
            vals = torch.ones([points.shape[0], topk]) * (-1)
            if return_grad:
                dirs = torch.zeros([points.shape[0], topk, 3]) * (-1)
        if return_grad:
            return vals.detach(), dirs.detach()
        else:
            return vals.detach()

    def get_sdf_np(self, points, topk=1, return_grad=False):

        points = points.view(-1, 3).numpy()
        if len(self.scene_sdfs_np) > 0:
            dists = []
            dirs = []
            with np.errstate(all='ignore'):
                for sdf in self.scene_sdfs_np:
                    if return_grad:
                        dist = sdf.forward(points)
                        dir = -sdf.egrad_fun(points)
                        dir /= np.linalg.norm(dir, axis=1, keepdims=True)
                        dir = np.nan_to_num(dir, nan=0, posinf=0, neginf=0)
                        dirs.append(dir)
                    else:
                        dist = sdf.forward(points)
                    dists.append(dist)

            dists = np.concatenate(dists, axis=1)

            if return_grad:
                dirs = np.concatenate(dirs, axis=1).reshape(-1, len(self.scene_sdfs), 3)
                dirs = torch.from_numpy(dirs)
                dirs = torch.nan_to_num(dirs)

            dists = torch.from_numpy(dists)
            dists = torch.nan_to_num(dists, -1)

            if len(self.scene_sdfs) < topk:

                vals, locs = torch.topk(dists, k=len(self.scene_sdfs), largest=False)
                vals = torch.cat([vals, torch.ones([vals.shape[0], topk - len(self.scene_sdfs)]) * (-1)], dim=1)
                if return_grad:
                    dirs = torch.cat([dirs[idx][loc][None,] for idx, loc in enumerate(locs)], dim=0)
                    dirs = torch.cat([dirs, torch.zeros([vals.shape[0], topk - len(self.scene_sdfs), 3])], dim=1)

            else:
                vals, locs = torch.topk(dists, k=topk, largest=False)
                if return_grad:
                    dirs = torch.cat([dirs[idx][loc][None,] for idx, loc in enumerate(locs)], dim=0)

        else:
            vals = torch.ones([points.shape[0], topk]) * (-1)
            if return_grad:
                dirs = torch.zeros([points.shape[0], topk, 3])
        if return_grad:
            return vals, dirs
        else:
            return vals

    def add_obj(self, obj_info):
        worldbody = self.tree.getroot().find("worldbody")

        for k, v in obj_info.items():
            geoms = v['geoms']
            obj_pose = v['obj_pose']
            pos = obj_pose[0][:3]

            quat = obj_pose[0][3:]
            body_node = Element("body", {"pos": "0 0 0"})
            for geom in geoms:
                geom_pos = np.array([float(i) for i in geom['pos'].split(" ")])
                geom_euler = np.array([float(i) for i in geom['euler'].split(" ")])
                geom_rot = sRot.from_euler('XYZ', geom_euler, degrees=True)
                annot_rot = sRot.from_quat(quat[[1, 2, 3, 0]])

                obj_rot = (annot_rot * geom_rot)
                obj_euler = obj_rot.as_euler('XYZ', degrees=True)
                obj_pos = pos + np.matmul(obj_rot.as_matrix(), geom_pos[:, None]).squeeze()

                geom['euler'] = " ".join([f"{i:.4f}" for i in obj_euler])
                geom['pos'] = " ".join([f"{i:.4f}" for i in obj_pos])
                geom['rgba'] = "1 0.6 0.6 1"
                geom_node = Element("geom", geom)
                body_node.append(geom_node)

                if geom['type'] == "box":
                    self.scene_sdfs.append(BoxSDF_F(trans=[float(i) for i in geom['pos'].split(" ")], orientation=obj_rot.as_matrix(), side_lengths=[float(i) * 2 for i in geom['size'].split(" ")]))
                    self.scene_sdfs_np.append(BoxSDF_N(trans=np.array([float(i) for i in geom['pos'].split(" ")]), orientation=obj_rot.as_matrix(), side_lengths=np.array([float(i) * 2 for i in geom['size'].split(" ")])))
                elif geom['type'] == "cylinder":
                    self.scene_sdfs.append(CylinderSDF_F(trans=[float(i) for i in geom['pos'].split(" ")], orientation=obj_rot.as_matrix(), size=[float(i) for i in geom['size'].split(" ")]))
                    self.scene_sdfs_np.append(CylinderSDF_N(trans=np.array([float(i) for i in geom['pos'].split(" ")]), orientation=obj_rot.as_matrix(), size=np.array([float(i) for i in geom['size'].split(" ")])))

            worldbody.append(body_node)

    # def export_xml_string(self):
    #     # tree_str =  etree.tostring(self.tree, pretty_print=True)
    #     if hasattr(self, "skeleton"):
    #         del self.skeleton
    #     return self.tree

    def add_simple_scene(self, scene_and_key):
        # check the xml file
        if 0:
            xml_str = etree.tostring(self.tree, pretty_print=True).decode("utf-8")
            print(xml_str)


        scene_name = scene_and_key.split('_')[0]
        asset_nodes = []
        worldbody = self.tree.getroot().find("worldbody")
        body_node = Element("body", {"pos": "0 0 0"})
        cwd = os.getcwd()

        filename = f'{cwd}/data/scenes/{scene_name}_planes.txt'
        if os.path.exists(filename):
            planes = np.loadtxt(filename)
            planes = planes.reshape(-1, 4, 3)
            for plane in planes:
                pos, size, xyaxes = self.get_scene_attrs_from_plane(plane)
                geom = {
                    "type": "box",
                    "friction": "1. .1 .1",
                    "rgba": "1 0.6 0.6 1",
                    "condim": "1",
                    "contype": "1",
                    "pos": pos,
                    "size": size,
                    "xyaxes": xyaxes,
                }
                geom_node = Element("geom", geom)
                body_node.append(geom_node)
                xy = np.array([float(i) for i in xyaxes.split(' ')]).reshape(2, 3).T
                xyz = np.hstack([xy, np.cross(xy[:, 0], xy[:, 1])[:, None]])
                sides = np.array([float(i) * 2 for i in size.split(" ")])

                self.scene_sdfs.append(BoxSDF_F(trans=[float(i) for i in pos.split(" ")], orientation=xyz, side_lengths=sides))

                self.scene_sdfs_np.append(BoxSDF_N(trans=np.array([float(i) for i in pos.split(" ")]), orientation=xyz, side_lengths=sides))

        filename = f'{cwd}/data/scenes/{scene_name}_rectangles.txt'
        if os.path.exists(filename):
            rectangles = np.loadtxt(filename)
            rectangles = rectangles.reshape(-1, 8, 3)
            for rectangle in rectangles:
                pos, size, xyaxes = self.get_scene_attrs_from_rectangle(rectangle)
                geom = {
                    "type": "box",
                    "friction": "1. .1 .1",
                    "rgba": "1 0.6 0.6 1",
                    "condim": "1",
                    "contype": "1",
                    # "margin": "0.05",
                    "pos": pos,
                    "size": size,
                    "xyaxes": xyaxes,
                }
                geom_node = Element("geom", geom)
                body_node.append(geom_node)

                xy = np.array([float(i) for i in xyaxes.split(' ')]).reshape(2, 3).T
                xyz = np.hstack([xy, np.cross(xy[:, 0], xy[:, 1])[:, None]])
                sides = np.array([float(i) * 2 for i in size.split(" ")])
                self.scene_sdfs.append(BoxSDF_F(trans=[float(i) for i in pos.split(" ")], orientation=xyz, side_lengths=sides))

                self.scene_sdfs_np.append(BoxSDF_N(trans=np.array([float(i) for i in pos.split(" ")]), orientation=xyz, side_lengths=sides))

        filename = f'{cwd}/data/scenes/{scene_name}_cylinders.txt'
        if os.path.exists(filename):
            cylinders = np.loadtxt(filename)
            cylinders = cylinders.reshape(-1, 64, 3)
            for cylinder in cylinders:
                pos, size = self.get_scene_attrs_from_cylinder(cylinder)
                geom = {
                    "type": "cylinder",
                    "friction": "1. .1 .1",
                    "rgba": "1 0.6 0.6 1",
                    "condim": "1",
                    "contype": "1",
                    "pos": pos,
                    "size": size,
                }
                geom_node = Element("geom", geom)
                body_node.append(geom_node)

                # TODO(sh8): There might be a bug because the code
                # in this webpage is wrong.
                # https://iquilezles.org/articles/distfunctions/
                trans = np.array([float(i) for i in pos.split(" ")])
                size = np.array([float(i) for i in size.split(" ")])
                self.scene_sdfs.append(CylinderSDF_F(trans=trans, size=size, orientation=[[1, 0, 0], [0, 0, 1], [0, 1, 0]]))
                self.scene_sdfs.append(CylinderSDF_N(trans=trans, size=size, orientation=[[1, 0, 0], [0, 0, 1], [0, 1, 0]]))

        filename = f'{cwd}/data/scenes/{scene_name}_circles.txt'
        if os.path.exists(filename):
            circles = np.loadtxt(filename)
            circles = circles.reshape(-1, 32, 3)
            for circle in circles:
                pos, size = self.get_scene_attrs_from_circle(circle)
                geom = {
                    "type": "cylinder",
                    "friction": "1. .1 .1",
                    "rgba": "1 0.6 0.6 1",
                    "condim": "1",
                    "contype": "1",
                    "pos": pos,
                    "size": size,
                }
                geom_node = Element("geom", geom)
                body_node.append(geom_node)

                # TODO(sh8): There might be a bug because the code
                # in this webpage is wrong.
                # https://iquilezles.org/articles/distfunctions/
                trans = np.array([float(i) for i in pos.split(" ")])
                size = np.array([float(i) for i in size.split(" ")])
                self.scene_sdfs.append(CylinderSDF_F(trans=trans, size=size, orientation=[[1, 0, 0], [0, 0, 1], [0, 1, 0]]))
                self.scene_sdfs.append(CylinderSDF_N(trans=trans, size=size, orientation=[[1, 0, 0], [0, 0, 1], [0, 1, 0]]))

        self.add_assets(asset_nodes)
        # this is important! Here, the scene is added to the worldbody and directly updates the xml tree
        worldbody.append(body_node)

        # check the xml file
        # here a new body <body pos="0 0 0"> is added with all the scene geometries,
        # to the same treee as the agent body
        if 0:
            # xml_str = etree.tostring(self.tree, pretty_print=True).decode("utf-8")
            fname = "inspect_out/xml_s/lastLine_add_simple_scene.xml"
            self.tree.write(fname, pretty_print=True)
            # print(xml_str)


    def add_agent(self, tree2, agent_id=2):
        # NOT REALLY USED
        """ add a second agent to the xml tree
        here tree2 is from agent2 and self.tree should be from agent1
        """
        # check the xml file
        if 0:
            xml_str = etree.tostring(self.tree, pretty_print=True).decode("utf-8")
            print(xml_str)

        self.export_robot_str()

        # can this be used here? better NOT
        # super().load_from_skeleton_2(betas=betas, gender=gender, num_agent=2)
        # now tree contains info about the agent 2, but it seems that agent 1 is overwritten

        # actuators  and contact will be added to mujoco node
        mujoco_node = self.tree.getroot()
        # the agent specification will be added to worldbody node
        worldbody = self.tree.getroot().find("worldbody")
        # NOTE: if I call this 2 times, it returns none

        tree2_ = copy.deepcopy(tree2)
        agent2_body_node = tree2_.getroot().find("worldbody").find("body")
        agent2_body_node.attrib["name"] = f"Pelvis{agent_id}"
        worldbody.append(agent2_body_node)
        if 0:
            # why is this none? --> it doesn't have <body> element
            fname = "inspect_out/xml_s/add_agent_tree2_node_none.xml"
            tree2.write(fname, pretty_print=True)

        agent2_contacts = tree2_.getroot().find("contact")
        mujoco_node.append(agent2_contacts)

        # for now removing the actuators due to shape incompatibility with the simu controls, that outputs the policy
        # NOTE: actually number of actuators does not affect that. It is the number of joints that matters
        agent2_actuator = tree2_.getroot().find("actuator")
        mujoco_node.append(agent2_actuator)

        # check the xml file
        # here a new body <body pos="0 0 0"> is added with all the geometries
        if 0:
            fname = "inspect_out/xml_s/add_agent_end_self_tree.xml"
            self.tree.write(fname, pretty_print=True)


    def add_agent_v2(self, smpl_robot, agent_id=2):
        # NOT REALLY USED
        """ add a second agent to the xml tree
        here tree2 is from agent2 and self.tree should be from agent1
        """
        # check the xml file
        if 0:
            xml_str = etree.tostring(self.tree, pretty_print=True).decode("utf-8")
            print(xml_str)

        # generate copy of the tree but with modified names according to agent id
        remove_elements = ["equality"]
        tree2, xml_str = self.export_robot_str(smpl_robot=smpl_robot, remove_elements=remove_elements,
                                              agent_id=agent_id)
        if 0:
            fname = "inspect_out/xml_s/add_agent_v2.xml"
            tree2.write(fname, pretty_print=True)
            fname = "inspect_out/xml_s/add_agent_v2_robot1.xml"
            self.tree.write(fname, pretty_print=True)

        # actuators  and contact will be added to mujoco node
        mujoco_node = self.tree.getroot()
        # the agent specification will be added to worldbody node

        worldbody = self.tree.getroot().find("worldbody")
        tree2_ = copy.deepcopy(tree2)

        agent2_asset = tree2_.getroot().find("asset")
        mujoco_node.append(agent2_asset)

        agent2_body_node = tree2_.getroot().find("worldbody").find("body")
        # append body node
        worldbody.append(agent2_body_node)
        # append contact node
        agent2_contacts = tree2_.getroot().find("contact")
        mujoco_node.append(agent2_contacts)
        # append actuator node
        agent2_actuator = tree2_.getroot().find("actuator")
        mujoco_node.append(agent2_actuator)

        if 0:
            fname = "inspect_out/xml_s/add_agent_v2_modif_agent1.xml"
            self.tree.write(fname, pretty_print=True)

        # return tree, xml_str
        

    def get_scene_attrs_from_plane(self, plane):
        c, x, y, w, h = self.get_cxy(plane)
        pos = f'{c[0]:04f} {c[1]:04f} {c[2]:04f}'
        size = f'{w/2:04f} {h/2:04f} 0.01'
        xaxis = f'{x[0]:04f} {x[1]:04f} {x[2]:04f}'
        yaxis = f'{y[0]:04f} {y[1]:04f} {y[2]:04f}'
        xyaxes = xaxis + ' ' + yaxis
        return pos, size, xyaxes

    def get_scene_attrs_from_rectangle(self, rectangle):
        ind = np.argsort(rectangle[:, 2])
        rectangle = rectangle[ind]
        c_top, x, y, w, h = self.get_cxy(rectangle[:4])
        c = np.mean(rectangle, axis=0)
        z = c[2] - c_top[2]
        pos = f'{c[0]:04f} {c[1]:04f} {c[2]:04f}'
        size = f'{w/2:04f} {h/2:04f} {z:04f}'
        xaxis = f'{x[0]:04f} {x[1]:04f} {x[2]:04f}'
        yaxis = f'{y[0]:04f} {y[1]:04f} {y[2]:04f}'
        xyaxes = xaxis + ' ' + yaxis
        return pos, size, xyaxes

    def get_scene_attrs_from_circle(self, circle):
        c = np.mean(circle, axis=0)
        dist = np.linalg.norm(circle[:32] - c, axis=1)
        min_ind = np.argmin(dist)
        w = dist[min_ind]
        pos = f'{c[0]:04f} {c[1]:04f} {c[2]:04f}'
        size = f'{w:04f} 0.01'
        return pos, size

    def get_scene_attrs_from_cylinder(self, cylinder):
        ind = np.argsort(cylinder[:, 2])
        cylinder = cylinder[ind]
        c_top = np.mean(cylinder[:32], axis=0)
        c = np.mean(cylinder, axis=0)
        z = (c[2] - c_top[2])
        dist = np.linalg.norm(cylinder[:32] - c_top, axis=1)
        min_ind = np.argmin(dist)
        w = dist[min_ind]
        pos = f'{c[0]:04f} {c[1]:04f} {c[2]:04f}'
        size = f'{w:04f} {z:04f}'
        return pos, size

    def get_cxy(self, points, c=None):
        if c is None:
            c = np.mean(points, axis=0)
        vec1 = points[0, :] - c
        vec2 = points[1, :] - c
        vec3 = points[2, :] - c
        x = vec1 + vec2
        y = vec2 + vec3
        w = np.linalg.norm(x)
        h = np.linalg.norm(y)
        x = x / (w + 1e-6)
        y = y / (h + 1e-6)
        if w < 0.01:
            _, x, y, w, h = self.get_cxy(points[[0, 2, 1]], c)
        if h < 0.01:
            _, x, y, w, h = self.get_cxy(points[[1, 0, 2]], c)
        return c, x, y, w, h

    def add_scene(self, scene_name):
        """Add a 3D scene to a XML tree"""

        if scene_name is None:
            return
        # scene_height = self.cfg['scene_height'][scene_name]
        # cwd = os.getcwd()
        stl_glob_path = f'/hdd/zen/dev/copycat/Sceneplus/data/scene/{scene_name}_aligned/convex_*.stl'
        stl_paths = glob.glob(stl_glob_path)

        new_stl_paths = []
        for stl_path in stl_paths:
            m = mesh.Mesh.from_file(stl_path)
            z = np.stack([m.v0[:, 2], m.v1[:, 2], m.v2[:, 2]], axis=1)
            min_z = np.min(np.mean(z, axis=1))
            if min_z >= 0.1:
                new_stl_paths.append(stl_path)
        stl_paths = new_stl_paths

        stl_paths = sorted(stl_paths, key=lambda v: int(v.split("/")[-1].split(".")[0][7:]))

        asset_nodes = []
        worldbody = self.tree.getroot().find("worldbody")
        geom_disable_range = np.zeros(len(stl_paths))

        body_node = Element("body", {"pos": "0 0 0"})
        stl_paths_0 = [stl_paths[i] for i in range(len(stl_paths)) if geom_disable_range[i] == 0]
        for idx, stl_path in enumerate(stl_paths_0):
            filename = os.path.basename(stl_path).replace(".stl", "")
            mesh_node = Element("mesh", {"file": stl_path})
            # TODO(sh8): Need to change pos dynamically
            margin = "0"
            color = "0 1 0 1"

            geom_node = Element(
                "geom",
                {
                    "mesh": filename,
                    "type": "mesh",
                    "rgba": color,
                    "pos": '0 0 0',
                    "friction": "1. .1 .1",
                    "condim": "1",
                    "contype": "1",
                    "margin": margin,
                    # "gap": "0"
                })
            asset_nodes.append(mesh_node)
            body_node.append(geom_node)
        self.add_assets(asset_nodes)
        worldbody.append(body_node)

    def add_assets(self, asset_nodes):
        asset = self.tree.getroot().find("asset")
        for asset_node in asset_nodes:
            asset.append(asset_node)

    def export_vis_string_self(self, num=3, smpl_robot=None, fname=None, num_cones=0):
        colors = [
                  "0.9 0.6 0.3 1", "0.7 0.1 0 1", "0.0 0.0 0.7 1",  # beige darker
                  "1.0 0.3 0.3 1", "0.6 0.0 .5 1", "0.6 0.0 .5 1",  # red darker
                  ]
        # Export multiple vis strings
        tree = deepcopy(self.tree)
        if smpl_robot is None:
            vis_tree = deepcopy(self.init_tree)
        else:
            vis_tree = deepcopy(smpl_robot.tree)

        # Removing actuators from the tree
        remove_elements = ["actuator", "contact", "equality"]
        for elem in remove_elements:
            # node = tree.getroot().find(elem)
            node_all = tree.getroot().findall(f".//{elem}")
            for node in node_all:
                if node is None:
                    pass
                else:
                    node.getparent().remove(node)

        # we right to tree and get the vis string from the tree
        ####### tree
        option = tree.getroot().find("option")
        option.addnext(Element("size", {"njmax": "1000"}))
        worldbody = tree.getroot().find("worldbody")
        assets_all = tree.getroot().findall(".//asset")

        # Removing actuators from the tree
        remove_elements = ["actuator", "contact", "equality, body"]
        for elem in remove_elements:
            node_all = vis_tree.getroot().findall(f".//{elem}")
            for node in node_all:
                if node is None:
                    pass
                else:
                    node.getparent().remove(node)

        vis_option = vis_tree.getroot().find("option")
        SubElement(vis_option, "flag", {"contact": "disable"})
        vis_option.addnext(Element("size", {"njmax": "1000"}))
        vis_worldbody = vis_tree.getroot().find("worldbody")
        vis_assets_all = vis_tree.getroot().findall(".//asset")

        vis_body_all = vis_worldbody.findall(".//body")
        for node in vis_body_all:
            if node is None:
                pass
            else:
                node.getparent().remove(node)

        for n, asset in enumerate(assets_all):
            for i in range(1, num):
                meshes = asset.findall("mesh")
                cur_meshes = deepcopy(meshes)
                for mesh in cur_meshes:
                    old_file = mesh.attrib["file"]
                    # modified the mesh file name adding "__" to not be mistaken with the visu meshes
                    mesh.attrib["file"] = mesh.attrib["file"].replace(".stl", f"___{i}.stl")
                    shutil.copy(old_file, mesh.attrib["file"])
                    vis_assets_all[n].append(mesh)

        # this needs to have changed order as such:
        body_all_pelvis = worldbody.findall(".//body")
        body_all_pelvis = [c for c in body_all_pelvis if "Pelvis" in c.attrib["name"]]

        for n, pelvis in enumerate(body_all_pelvis):
            new_body = deepcopy(pelvis)
            new_body.attrib["name"] = "%s" % (new_body.attrib["name"])
            new_body.find("geom").attrib["rgba"] = colors[n * 3]
            for node in new_body.findall(".//body"):
                node.attrib["name"] = "%s" % (node.attrib["name"])
                node.find("geom").attrib["rgba"] = colors[n * 3]
            for node in new_body.findall(".//joint"):
                node.attrib["name"] = "%s" % (node.attrib["name"])
            for node in new_body.findall(".//site"):
                node.attrib["name"] = "%s" % (node.attrib["name"])
            for node in new_body.findall(".//geom"):
                node.attrib["mesh"] = "%s" % (node.attrib["mesh"])
            vis_worldbody.append(new_body)
            for i in range(1, num):
                new_body = deepcopy(pelvis)
                new_body.attrib["name"] = "%d_%s" % (i, new_body.attrib["name"])
                new_body.find("geom").attrib["rgba"] = colors[i + 3*n]
                for node in new_body.findall(".//body"):
                    node.attrib["name"] = "%d_%s" % (i, node.attrib["name"])
                    node.find("geom").attrib["rgba"] = colors[i + 3*n]
                for node in new_body.findall(".//joint"):
                    node.attrib["name"] = "%d_%s" % (i, node.attrib["name"])
                for node in new_body.findall(".//site"):
                    node.attrib["name"] = "%d_%s" % (i, node.attrib["name"])
                for node in new_body.findall(".//geom"):
                    node.attrib["mesh"] = "%s___%d" % (node.attrib["mesh"], i)
                vis_worldbody.append(new_body)


        for i in range(num_cones):
            color = np.random.random(3)
            vis_worldbody.append(Element(
                "geom",
                {
                    "pos": "-0.0200 -0.0471 0.6968",
                    "rgba": f"{color[0]:.3f} {color[1]:.3f} {color[2]:.3f} 1.0",
                    "type": "sphere",
                    "size": "0.0420",
                },
            ))

        if fname is not None:
            print("Writing to file: %s" % fname)
            Path(fname).parent.mkdir(parents=True, exist_ok=True)
            vis_tree.write(fname, pretty_print=True)
        vis_str = etree.tostring(vis_tree, pretty_print=True)
        return vis_str

    def export_robot_str(self, fname=None, num_cones=0,
                         remove_elements = ["actuator", "contact", "equality"], agent_id=1):
        colors = ["0.8 0.6 .4 1", "0.7 0 0 1", "0.0 0.0 0.7 1"]
        tree = deepcopy(self.tree)

        # Removing actuators from the tree
        for elem in remove_elements:
            node = tree.getroot().find(elem)
            if node is None:
                pass
            else:
                node.getparent().remove(node)

        mujoco_node = tree.getroot()
        worldbody = tree.getroot().find("worldbody")

        # remove body that contains scene info, not necessary for robot
        all_bodies = worldbody.findall(".//body")#[-1]
        for elem in all_bodies:
            # print(elem)
            if "name" not in elem.attrib.keys():
                # print('has boxes')
                worldbody.remove(elem)

        asset = tree.getroot().find("asset")
        i = agent_id

        # assets
        # all_asset = list(asset)
        vis_meshes = asset.findall("mesh")
        # cur_meshes = deepcopy(vis_meshes)
        for mesh in vis_meshes:
            old_file = mesh.attrib["file"]
            # remove previous
            asset.remove(mesh)
            mesh.attrib["file"] = mesh.attrib["file"].replace(".stl", f"_{i}.stl")
            shutil.copy(old_file, mesh.attrib["file"])
            asset.append(mesh)

        remove = ['texture', 'material']
        for elem in remove:
            element = asset.findall(elem)
            for cur_asset in element:
                asset.remove(cur_asset)

        str_frmt = "%s_%d"
        str_frmt_2 = "%s_%d"

        # body and elemtents
        body = worldbody.find("body")
        new_body = deepcopy(body)
        new_body.attrib["name"] = str_frmt % (new_body.attrib["name"], i)
        new_body.find("geom").attrib["rgba"] = colors[i]
        for node in new_body.findall(".//body"):
            node.attrib["name"] = str_frmt % (node.attrib["name"], i)
            node.find("geom").attrib["rgba"] = colors[i]
        for node in new_body.findall(".//joint"):
            node.attrib["name"] = str_frmt % (node.attrib["name"], i)
        for node in new_body.findall(".//site"):
            node.attrib["name"] = str_frmt % (node.attrib["name"], i)
        for node in new_body.findall(".//geom"):
            node.attrib["mesh"] = str_frmt_2 % (node.attrib["mesh"], i)
        worldbody.remove(body)
        worldbody.append(new_body)

        # actuator and motors
        agent_actuator = mujoco_node.find("actuator")
        cur_motors = deepcopy(agent_actuator)
        for motor in cur_motors:
            # motor.attrib["name"] = str_frmt % (i, motor.attrib["name"])
            motor.attrib["name"] = str_frmt % (motor.attrib["name"], i)
            # motor.attrib["joint"] = str_frmt % (i, motor.attrib["joint"])
            motor.attrib["joint"] = str_frmt % (motor.attrib["joint"], i)
        mujoco_node.remove(agent_actuator)
        mujoco_node.append(cur_motors)

        # contacts
        agent_contacts = mujoco_node.find("contact")
        cur_exclude = deepcopy(agent_contacts)
        for exclude in cur_exclude:
            # exclude.attrib["name"] = str_frmt % (i, exclude.attrib["name"])
            exclude.attrib["name"] = str_frmt % (exclude.attrib["name"], i)
            # exclude.attrib["body1"] = str_frmt % (i, exclude.attrib["body1"])
            exclude.attrib["body1"] = str_frmt % (exclude.attrib["body1"], i)
            # exclude.attrib["body2"] = str_frmt % (i, exclude.attrib["body2"])
            exclude.attrib["body2"] = str_frmt % (exclude.attrib["body2"], i)
        mujoco_node.remove(agent_contacts)
        mujoco_node.append(cur_exclude)



        # for i in range(num_cones):
        #     color = np.random.random(3)
        #     worldbody.append(Element(
        #         "geom",
        #         {
        #             "pos": "-0.0200 -0.0471 0.6968",
        #             "rgba": f"{color[0]:.3f} {color[1]:.3f} {color[2]:.3f} 1.0",
        #             "type": "sphere",
        #             "size": "0.0420",
        #         },
        #     ))

        if fname is not None:
            print("Writing to file: %s" % fname)
            tree.write(fname, pretty_print=True)
        vis_str = etree.tostring(tree, pretty_print=True)
        return tree, vis_str

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # parser.add_argument("--cfg", default="copycat_ball_1")
    parser.add_argument("--cfg", default="copycat_40")
    args = parser.parse_args()

    cc_cfg = CC_Config(cfg_id="copycat_e_1", base_dir="/hdd/zen/dev/copycat/Copycat/")

    smpl_robot = SceneRobot(
        cc_cfg.robot_cfg,
        data_dir=osp.join(cc_cfg.base_dir, "data/smpl"),
        masterfoot=cc_cfg.masterfoot,
    )

    smpl_robot.load_from_skeleton(torch.zeros((1, 16)), gender=[0], scene_and_key="N3Library_03301_01")
    smpl_robot.write_xml("test.xml")

    # model = mujoco_py.load_model_from_xml(smpl_robot.export_xml_string().decode("utf-8"))

    # print(f"mass {mujoco_py.functions.mj_getTotalmass(model)}")
    # sim = MjSim(model)

    # viewer = MjViewer(sim)

    # while True:
    #     # sim.data.qpos[54] = np.pi / 2
    #     sim.data.qpos[1] -= 0.005
    #     # sim.data.qpos[76:] = grab_data[key]["obj_pose"][i % seq_len]

    #     sim.forward()
    #     viewer.render()

    #     points = torch.from_numpy(sim.data.body_xpos)
    #     sdf = smpl_robot.get_sdf(points)

    #     print(sdf[1], points[1])

    #     # smpl_robot.get_sdf(torch.tensor([[0.799508, 0.556263, 0.145840]]))
