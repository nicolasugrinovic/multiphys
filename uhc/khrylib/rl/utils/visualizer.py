from uhc.khrylib.rl.envs.visual.humanoid_vis import HumanoidVisEnv
import glfw
import math


class Visualizer:

    def __init__(self, vis_file):
        self.fr = 0
        self.num_fr = 0
        self.T_arr = [1, 2, 4, 6, 8, 10, 12, 15, 20, 30, 40, 50, 60]
        self.T = 12
        self.paused = False
        self.reverse = False
        self.repeat = False
        self.vis_file = vis_file

        self.env_vis = HumanoidVisEnv(vis_file, 1, focus=False)
        # print(vis_file)

        visu_mode = self.agent.cfg.visualizer
        viewer_type = self.agent.cfg.viewer_type
        if visu_mode == "viewer":
            # self.env_vis._get_viewer("human")._hide_overlay = True
            self.env_vis._get_viewer(viewer_type)._hide_overlay = True
            # self.env_vis._get_viewer("image")._hide_overlay = True
            # self.env_vis._viewers
            self.env_vis.set_custom_key_callback(self.key_callback, viewer_type)
        else:
            print("Not using viewer, rendering directly from sim.render()")

    def data_generator(self):
        raise NotImplementedError

    def update_pose(self):
        raise NotImplementedError

    def key_callback(self, key, action, mods):

        if action != glfw.RELEASE:
            return False
        if key == glfw.KEY_D:
            self.T = self.T_arr[(self.T_arr.index(self.T) + 1) % len(self.T_arr)]
            print(f"T: {self.T}")
        elif key == glfw.KEY_F:
            self.T = self.T_arr[(self.T_arr.index(self.T) - 1) % len(self.T_arr)]
            print(f"T: {self.T}")
        elif key == glfw.KEY_Q:
            self.data = next(self.data_gen, None)
            if self.data is None:
                print("end of data!!")
                exit()
            self.fr = 0
            self.update_pose()
        elif key == glfw.KEY_W:
            self.fr = 0
            self.update_pose()
        elif key == glfw.KEY_E:
            self.fr = self.num_fr - 1
            self.update_pose()
        elif key == glfw.KEY_G:
            self.repeat = not self.repeat
            self.update_pose()

        elif key == glfw.KEY_S:
            self.reverse = not self.reverse
        elif key == glfw.KEY_RIGHT:
            print("right pressed, this is Visualizer")
            if self.fr < self.num_fr - 1:
                self.fr += 1
            self.update_pose()
        elif key == glfw.KEY_LEFT:
            print("left pressed, this is Visualizer")
            if self.fr > 0:
                self.fr -= 1
            self.update_pose()
        elif key == glfw.KEY_SPACE:
            self.paused = not self.paused
        else:
            return False
        return True

    def render(self, mode):
        img_w = self.agent.env.camera_params["img_w"]
        img_h = self.agent.env.camera_params["img_h"]
        img = self.env_vis.render(mode, width=img_w, height=img_h) # calls self._get_viewer(mode).render()
        return img

    # def show_animation(self):
    #
    #     self.t = 0
    #     while True:
    #         if self.t >= math.floor(self.T):
    #             if not self.reverse:
    #                 if self.fr < self.num_fr - 1:
    #                     self.fr += 1
    #                 elif self.repeat:
    #                     self.fr = 0
    #             elif self.reverse and self.fr > 0:
    #                 self.fr -= 1
    #             self.update_pose()
    #             self.t = 0
    #         self.render()
    #         if not self.paused:
    #             self.t += 1

    def show_animation(self, mode="human", duration=60, cam_num=0):
        frames = []
        self.t = 0
        while True:
            if self.t >= math.floor(self.T): # self.T = 12
                if not self.reverse:
                    if self.fr < self.num_fr - 1:
                        self.fr += 1
                        if mode == "image":
                            rend_img = self.render("image")
                            frames.append(rend_img)
                            if 0:
                                from utils.misc import plot
                                plot(rend_img)
                                plot(frames[-2])
                        if mode == "image" and self.fr >= duration-1:
                            return frames
                    elif self.repeat:
                        self.fr = 0
                elif self.reverse and self.fr > 0:
                    self.fr -= 1
                self.update_pose()

                self.t = 0
            ######################################
            # this renders the simulation result and can be streamed to a image files
            if mode == "human":
                rend_img = self.render("human")
                frames.append(rend_img)

            ######################################
            # self.update_pose() # goes to scripts/eval_scene.py
            if not self.paused:
                self.t += 1


