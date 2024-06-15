import numpy as np
import mujoco_py

from uhc.khrylib.rl.envs.common import mujoco_env


class HumanoidVisEnv(mujoco_env.MujocoEnv):
    def __init__(self, vis_model_file, nframes=6, focus=True):
        mujoco_env.MujocoEnv.__init__(self, vis_model_file, nframes)
        # print(vis_model_file)

        self.set_cam_first = set()
        self.focus = focus

    def step(self, a):
        return np.zeros((10, 1)), 0, False, dict()

    def reset_model(self):
        c = 0
        self.set_state(
            self.np_random.uniform(low=-c, high=c, size=self.model.nq),
            self.np_random.uniform(low=-c, high=c, size=self.model.nv),
        )
        return None

    def sim_forward(self):
        self.sim.forward()

    def set_video_path(
        self, image_path="/tmp/image_%07d.png", video_path="/tmp/video_%07d.mp4"
    ):
        self.viewer._image_path = image_path
        self.viewer._video_path = video_path

    def viewer_setup(self, mode):
        self.viewer.cam.trackbodyid = 1
        if self.focus:
            self.viewer.cam.lookat[:2] = self.data.qpos[:2]
            self.viewer.cam.lookat[2] = 0.8
        if mode not in self.set_cam_first:
            self.viewer.video_fps = 30
            self.viewer.frame_skip = self.frame_skip
            self.viewer.cam.distance = self.model.stat.extent * 1.5
            self.viewer.cam.elevation = -10
            self.viewer.cam.azimuth = 45
            self.set_cam_first.add(mode)
        
            # NUK: second attempt to record video
            self.viewer._record_video = True
            self.viewer._image_path = "inspect_out/frame_%07d.png"
            self.viewer._record_video = "inspect_out/video_%07d.mp4"
        

    def reload_sim_model(self, xml_str, viewer_type):
        del self.sim
        del self.model
        del self.data
        del self.viewer
        del self._viewers

        # when it enters this function then self.model.nv and self.model.nq change sizes
        # why?
        # this xml file contains three agents: Pelvis, 1_Pelvis, 2_Pelvis.
        # it also contains more mesh files. This explains it:
        # 76x3 = 228
        # 75x3 = 225
        if 0:
            fname = "inspect_out/xml_s/xml_str_reload_sim_model.xml"
            with open(fname, "w") as f:
                f.write(xml_str)
        self.model = mujoco_py.load_model_from_xml(xml_str)
        self.sim = mujoco_py.MjSim(self.model)
        self.data = self.sim.data
        # thus here self.init_qpos is (228,) and self.init_qvel is (225,)
        self.init_qpos = self.sim.data.qpos.copy()
        self.init_qvel = self.sim.data.qvel.copy()
        self.viewer = None
        self._viewers = {}
        # self._get_viewer("human")._hide_overlay = True
        self._get_viewer(viewer_type)._hide_overlay = True
        ################ important ################
        # here set_state(self, qpos, qvel) is called
        self.reset()
        ###########################################

        print("Reloading Vis Sim")
