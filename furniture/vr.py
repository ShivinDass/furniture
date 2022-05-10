"""
Human control for the IKEA furniture assembly environment.
The `three_blocks` task requires a robot to move the block on the left to the top of the center block, and attach them.
Then, the right block should be attached on top of the left block.
"""

import gym
import hydra
from omegaconf import OmegaConf, DictConfig

from furniture import agent_names  # list of available agents
from furniture import background_names  # list of available background scenes
from furniture import furniture_names  # list of available furnitures


def main_vr_test(cfg):
    # specify agent, furniture, and background
    agent_name = agent_names[1]
    furniture_name = "three_blocks"
    background_name = background_names[3]
    cfg.ikea_cfg.unity.background = background_name
    # print(cfg.ikea_cfg)
    
    # set correct environment name based on agent_name
    env_name = "IKEA{}-v0".format(agent_name)

    # make environment following arguments
    env = gym.make(
        env_name, furniture_name=furniture_name, unity={"background": background_name}, ikea_cfg=cfg.ikea_cfg
    )

    # manual control of agent using Oculus Quest2
    # env.run_vr_oculus()
    env.collect_oculus_teleop_traj(n_traj=5, file_path="/home/shivin/.furniture/datasets/", file_suffix="teleop_data_full_task_no_rotation.hdf5", append=True)#file_path=None)

    # close the environment instance
    env.close()


@hydra.main(config_path="config", config_name="default_config")
def main(cfg: DictConfig) -> None:
    # make config writable
    OmegaConf.set_struct(cfg, False)

    # set environment config for keyboard control
    cfg.env.ikea_cfg.unity.use_unity = True
    cfg.env.ikea_cfg.render = True
    cfg.env.ikea_cfg.control_type = "ik_quaternion"
    cfg.env.ikea_cfg.max_episode_steps = 10000
    cfg.env.ikea_cfg.screen_size = [1024, 1024]
    cfg.env.ikea_cfg.seed = 89

    #Different camera perspectives
    cfg.env.ikea_cfg.camera_ids = [0, 1]
    
    #Noise for randomizing furniture placement
    cfg.env.ikea_cfg.furn_xyz_rand = 0.1   
    cfg.env.ikea_cfg.furn_rot_rand = 9

    #Relaxing constraints for easier task assembly
    cfg.env.ikea_cfg.alignment_pos_dist = 0.1
    cfg.env.ikea_cfg.alignment_rot_dist_up = 0.8
    cfg.env.ikea_cfg.alignment_rot_dist_forward = 0.8 
    # print(cfg.env.ikea_cfg)

    main_vr_test(cfg.env)


if __name__ == "__main__":
    main()
