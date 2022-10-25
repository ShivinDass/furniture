from ast import ImportFrom
import h5py
import cv2
import copy
from getch import getch
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


data_file = '/home/shivin/.furniture/datasets/Sawyer_three_blocks_teleop_data_multi_task.hdf5'
data_file = '/home/icaros/data/furniture/Sawyer_three_blocks_teleop_data_multi_task_gripper_labels.hdf5'

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
    # make environment following arguments
    env = gym.make(
        "IKEA{}-v0".format(agent_names[1]), furniture_name="three_blocks", unity={"background": background_names[3]}, ikea_cfg=cfg.ikea_cfg
    )

    f = h5py.File(data_file, 'r')
    
    seq_end_idxs = np.where(f['terminals'])[0]

    start_index = 0
    env.reset()
    env.env.init_pos, env.env.init_quat = get_block_pos_from_obs(f['observations'][start_index])
    env.env.fixed_parts = []
    env.reset()

    episode_count = 0
    count_success = 0
    # img_list = []
    # obs_list = []
    for i,a in enumerate(f['actions'][start_index:]):
        #a = np.concatenate([a[:3], a[5:]])
        obs, reward, done, _ = env.step(a)
        env.render()
        
        # img = cv2.cvtColor(env.render(mode='rgb_array')[0], cv2.COLOR_BGR2RGB)
        # img_list.append(img)
        # obs_list.append(np.concatenate((obs['object_ob'], obs['robot_ob']), axis=0))

        if f['terminals'][i]:
            if done:
                count_success += 1

            episode_count += 1
            if episode_count < len(seq_end_idxs):
                env.env.init_pos, env.env.init_quat = get_block_pos_from_obs(f['observations'][i+start_index+1])
                env.env.fixed_parts = []
                env.reset()

            # save_dir = '/home/shivin/.furniture/datasets/Sawyer_three_blocks_teleop_data_multi_task/'
            # with h5py.File(os.path.join(save_dir, 'demo{}.hdf5'.format(episode_count)), 'w') as g:
            #     g.create_dataset('observations',data=np.array(obs_list))
            #     g.create_dataset('images', data=np.array(img_list))

            # del img_list
            # del obs_list
            # img_list = []
            # obs_list = []
            
            print("Total", count_success, "successes out of", 1 + np.sum(f['terminals'][start_index: i+start_index]))

    env.close()

def select_rollout_frames(cfg):
    env = gym.make(
        "IKEA{}-v0".format(agent_names[1]), furniture_name="three_blocks", unity={"background": background_names[3]}, ikea_cfg=cfg.ikea_cfg
    )

    f = h5py.File(data_file, 'r')
    
    seq_end_idxs = np.concatenate(([0], np.where(f['terminals'])[0]), axis=0)

    env.reset()
    env.env.init_pos, env.env.init_quat = get_block_pos_from_obs(f['observations'][0])
    env.env.fixed_parts = []
    obs = env.reset()

    selected_frames = {'observations': [], 'images': [], 'captions': []}

    index = 0
    max_ind = 0
    img_list = []
    obs_list = []
    episode_count = 0
    quit_now = False
    while 1>0:

        img = cv2.cvtColor(env.render(mode='rgb_array')[0], cv2.COLOR_BGR2RGB)
        img_list.append(img)
        obs_list.append(np.concatenate((obs['object_ob'], obs['robot_ob']), axis=0))

        while index <= max_ind:
            cv2.imshow('cur_frame', img_list[index-seq_end_idxs[episode_count]-1])
            cv2.waitKey(1)
            
            keypress = getch()

            if keypress == 'd':
                index = min(seq_end_idxs[-1], index+1)
            elif keypress == 'a':
                index = max(seq_end_idxs[episode_count], index-1)
            elif keypress == 'y':
                print()
                caption = input('caption: ')

                selected_frames['captions'].append(caption.encode('ascii', 'ignore'))
                selected_frames['observations'].append(obs_list[index])
                selected_frames['images'].append(img_list[index])

            elif keypress == 'q':
                quit_now = True
                break

        if quit_now:
            save_file = '/home/shivin/.furniture/datasets/Sawyer_three_blocks_teleop_data_multi_task_diverse_frames.hdf5'
            with h5py.File(save_file, 'w') as g:
                g.create_dataset('observations',data=np.array(selected_frames['observations']))
                g.create_dataset('images', data=np.array(selected_frames['images']))
                g.create_dataset('captions', data=selected_frames['captions'])
            break

        if f['terminals'][index]:
            if index < seq_end_idxs[-1]:
                episode_count += 1
                index += 1

                env.env.init_pos, env.env.init_quat = get_block_pos_from_obs(f['observations'][index])
                env.env.fixed_parts = []
                env.reset()

                del img_list
                del obs_list
                img_list = []
                obs_list = []

        a = f['actions'][index]
        obs, reward, done, _ = env.step(a)

        max_ind = max(index, max_ind)


def generate_blended_demo(cfg):
    traj_save_file = '/home/shivin/.furniture/datasets/trajectory_visualization/Sawyer_three_blocks_simple_rollout_images'

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
    cfg.env.ikea_cfg.screen_size = [448, 448]
    cfg.env.ikea_cfg.seed = 1
    cfg.env.ikea_cfg.fix_init_parts = block_names

    #Different camera perspectives
    # cfg.env.ikea_cfg.camera_ids = [0, 1]
    
    # #Noise for randomizing furniture placement
    # cfg.env.ikea_cfg.furn_xyz_rand = 0.1   
    # cfg.env.ikea_cfg.furn_rot_rand = 6

    #Relaxing constraints for easier task assembly
    cfg.env.ikea_cfg.alignment_pos_dist = 0.015
    cfg.env.ikea_cfg.alignment_rot_dist_up = 0.8
    cfg.env.ikea_cfg.alignment_rot_dist_forward = 0.8
    
    #set rewards
    for k in cfg.env.ikea_cfg.reward.keys():
        cfg.env.ikea_cfg.reward[k] = 0
    cfg.env.ikea_cfg.reward.success = 100
    cfg.env.ikea_cfg.reward.pick = 10

    rollout(cfg.env)
    #select_rollout_frames(cfg.env)
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
