import h5py
import cv2
import copy
import pickle
import gym
import time
import os
import hydra
import numpy as np
from omegaconf import OmegaConf, DictConfig
from furniture import furniture_names, agent_names, background_names
from furniture.env.models import furniture_name2id
from assisted_teleop.utils.general_utils import AttrDict


data_file = '/home/shivin/.furniture/datasets/Sawyer_three_blocks_teleop_data_full_task_no_rotation.hdf5'
traj_save_file = '/home/shivin/.furniture/datasets/trajectory_visualization/Sawyer_three_blocks_simple_rollout_images'

block_names = ['1_block_l', '2_block_m', '3_block_r']

def get_block_pos_from_obs(obs):
    pos = {}
    quat = {}
    for i, block in enumerate(block_names):
        pos[block] = obs[i*7 : i*7+3]
        quat[block] = obs[i*7+3 : (i+1)*7]
    return pos, quat

def get_init_action():
    pos = [0, 0, 0]
    rot = [-0.36613403, -0.30808327,  0.49565844,  0.72481258]
    select = [-1]
    connect = [-1]

    return np.array(pos + rot + select + connect)

def rollout(cfg):
    agent_name = agent_names[1]
    furniture_name = "three_blocks"
    background_name = background_names[3]
    cfg.ikea_cfg.unity.background = background_name
    # set correct environment name based on agent_name
    env_name = "IKEA{}-v0".format(agent_name)

    
    # make environment following arguments
    env = gym.make(
        env_name, furniture_name=furniture_name, unity={"background": background_name}, ikea_cfg=cfg.ikea_cfg
    )

    f = h5py.File(data_file, 'r')
    
    env.reset()
    env.env.init_pos, env.env.init_quat = get_block_pos_from_obs(f['observations'][0])
    env.env.fixed_parts = []
    env.reset()

    init_action = get_init_action()
    env.step(init_action)
    
    count_success = 0
    for i,a in enumerate(f['actions']):
        obs, _, done, _ = env.step(a)
        
        env.render()
        # input()
        # time.sleep(0.5)
        
        if f['terminals'][i]:
            env.env.init_pos, env.env.init_quat = get_block_pos_from_obs(f['observations'][i+1])
            env.env.fixed_parts = []
            env.reset()

            env.step(init_action)

            if done:
                count_success += 1
            
            print("Total", count_success, "successes out of", 1 + np.sum(f['terminals'][:i]))
    env.close()

def generate_blended_demo(cfg):
    from assisted_teleop.configs.default_data_configs.furniture import data_spec
    from assisted_teleop.data.furniture.src.furniture_data_loader import OculusVRSequenceSplitDataset

    data_conf = AttrDict()
    data_conf.dataset_spec = data_spec
    data_conf.dataset_spec.subseq_len = 1
    data_conf.device = 'cpu'
    dataset = OculusVRSequenceSplitDataset(data_dir = data_file,
                                data_conf= data_conf, phase='train').seqs
    
    agent_name = agent_names[1]
    furniture_name = "three_blocks"
    background_name = background_names[3]
    cfg.ikea_cfg.unity.background = background_name
    env_name = "IKEA{}-v0".format(agent_name)

    
    # make environment following arguments
    env = gym.make(
        env_name, furniture_name=furniture_name, unity={"background": background_name}, ikea_cfg=cfg.ikea_cfg
    )
    env.reset()

    blended_rollouts = [255*np.ones(shape=(1024, 1024, 3), dtype=np.uint8)]
    start = 0
    for num in range(start, len(dataset), 1):
        seq = dataset[num]

        env.env.init_pos, env.env.init_quat = get_block_pos_from_obs(seq.states[0])
        env.env.fixed_parts = []
        env.reset()
        
        init_action = get_init_action()
        env.step(init_action)
        
        n = num
        last_frame = copy.deepcopy(blended_rollouts[-1])
        for i, a in enumerate(seq.actions):
            env.step(a)
            img = env.render(mode="rgb_array")
            img = img[0][:,:,::-1]
            
            if len(blended_rollouts)-1 < i:
                blended_rollouts.append(
                    cv2.addWeighted(last_frame, n/(n+1), img, 1/(n+1), gamma=0)
                )
            else:
                blended_rollouts[i] = cv2.addWeighted(blended_rollouts[i], n/(n+1), img, 1/(n+1), gamma=0)
        
        for j in range(i,len(blended_rollouts)):
            blended_rollouts[j] = cv2.addWeighted(blended_rollouts[j], n/(n+1), img, 1/(n+1), gamma=0)

        for img in blended_rollouts:
            cv2.imshow("B", img)
            cv2.waitKey(1)
    env.close()

    pickle.dump(blended_rollouts, open(traj_save_file + ".pkl", 'wb'))

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
    cfg.env.ikea_cfg.seed = 1
    cfg.env.ikea_cfg.fix_init_parts = block_names

    #Noise for randomizing furniture placement
    cfg.env.ikea_cfg.furn_xyz_rand = 0.1   
    cfg.env.ikea_cfg.furn_rot_rand = 7

    #Relaxing constraints for easier task assembly
    cfg.env.ikea_cfg.alignment_pos_dist = 0.1
    cfg.env.ikea_cfg.alignment_rot_dist_up = 0.8
    cfg.env.ikea_cfg.alignment_rot_dist_forward = 0.8

    rollout(cfg.env)
    # generate_blended_demo(cfg.env)

def playback_blended_rollouts(traj_save_file):
    r = pickle.load(open(traj_save_file + ".pkl", "rb"))
    img_list = []
    for _ in range(50):
        cv2.imshow("B", r[0])
        cv2.waitKey(1)
    for img in r:
        cv2.imshow("B", img)
        img_list.append(img[:,:,::-1])
        cv2.waitKey(10)
    
    # import imageio
    # imageio.mimsave(traj_save_file + ".gif", img_list, 'GIF-FI', fps=30)
        
    # video = cv2.VideoWriter(traj_save_file + ".avi", 0, 30, (1024,1024))
    # for img in r:
    #     video.write(img)
    # cv2.destroyAllWindows()
    # video.release()
    return

if __name__ == "__main__":
    main()
    # playback_blended_rollouts(traj_save_file)