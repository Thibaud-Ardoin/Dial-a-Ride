import os
import numpy as np
import csv
import matplotlib.pyplot as plt
from moviepy.editor import *
from matplotlib.image import imsave
import matplotlib
matplotlib.use('Agg')

import tensorflow as tf
from stable_baselines.common.callbacks import BaseCallback, EvalCallback

class MonitorCallback(EvalCallback):
    """
    Callback for saving a model (the check is done every ``check_freq`` steps)
    based on the training reward (in practice, we recommend using ``EvalCallback``).

    :param check_freq: (int)
    :param log_dir: (str) Path to the folder where the model will be saved.
      It must contains the file created by the ``Monitor`` wrapper.
    :param verbose: (int)
    """
    def __init__(self, eval_env, check_freq: int, log_dir: str,sacred=None, n_eval_episodes=5, render=False, verbose=1):
        super(MonitorCallback, self).__init__(verbose=verbose,
                                              eval_env=eval_env,
                                              best_model_save_path=log_dir,
                                              log_path=log_dir,
                                              eval_freq=check_freq,
                                              n_eval_episodes=n_eval_episodes,
                                              deterministic=False,
                                              render=render)
        self.render = render
        self.verbose = verbose
        self.env = eval_env
        self.check_freq = check_freq
        self.log_dir = log_dir
        self.save_path = os.path.join(log_dir, 'best_model')
        self.best_mean_reward = -np.inf
        self.sacred = sacred

        self.sequence = False
        if self.env.__class__.__name__ in ['DarSeqEnv','DummyVecEnv'] :
            self.sequence = True

        self.statistics = {
            'step_reward': [],
            'reward': [],
            'std_reward': [],
            'duration': [],
            # 'policy_loss': [],
            # 'value_loss': [],
            # 'policy_entropy': []
        }


    def _init_callback(self) -> None:
        # Create folder if needed
        if self.log_dir is not None:
            os.makedirs(self.log_dir, exist_ok=True)

    def _on_training_start(self) -> None:
        """
        This method is called before the first rollout starts.
        """
        pass

    def _on_training_end(self) -> None:
        """
        This event is triggered before exiting the `learn()` method.
        """
        pass


    def plot_statistics(self, show=False):
        # Print them
        if self.verbose:
            print('\t ->[Epoch %d]<- mean episodic reward: %.3f' % (self.num_timesteps + 1, self.statistics['reward'][-1]))
            print('\t * Mean duration : %0.3f' % (self.statistics['duration'][-1]))
            print('\t * Mean std_reward : %0.3f' % (self.statistics['std_reward'][-1]))
            print('\t * Mean step_reward : %0.3f' % (self.statistics['step_reward'][-1]))
            # print('\t ** policy_loss : %0.3f' % (self.statistics['policy_loss'][-1]))
            # print('\t ** value_loss : %0.3f' % (self.statistics['value_loss'][-1]))
            # print('\t ** policy_entropy : %0.3f' % (self.statistics['policy_entropy'][-1]))

        # Create plot of the statiscs, saved in folder
        colors = [plt.cm.tab20(0),plt.cm.tab20(1),plt.cm.tab20c(2),
                  plt.cm.tab20c(3), plt.cm.tab20c(4),
                  plt.cm.tab20c(5),plt.cm.tab20c(6),plt.cm.tab20c(7)]
        fig, (axis) = plt.subplots(1, len(self.statistics), figsize=(20, 10))
        fig.suptitle(' - PPO Training: ' + self.log_dir)
        for i, key in enumerate(self.statistics):
            # Sacred (The one thing to keep here)
            if self.sacred :
                self.sacred.get_logger().report_scalar(title='Train stats',
                series=key, value=self.statistics[key][-1], iteration=self.num_timesteps)
                # self.sacred.log_scalar(key, self.statistics[key][-1], len(self.statistics[key]))
            axis[i].plot(self.statistics[key], color=colors[i])
            axis[i].set_title(' Plot of ' + key)
        if show :
            fig.show()
        fig.savefig(self.log_dir + '/result_figure.jpg')
        fig.clf()
        plt.close(fig)

        # Save the statistics as CSV file
        if not self.sacred:
            try:
                with open(self.log_dir + '/statistics.csv', 'w') as csvfile:
                    writer = csv.DictWriter(csvfile, fieldnames=self.statistics.keys())
                    writer.writeheader()
                    # for key in statistics
                    writer.writerow(self.statistics)
            except IOError:
                print("I/O error")


    def save_image_batch(self, images, rewards, txt='test'):
        ''' Saving some examples of input -> output to see how the model behave '''
        print(' - Saving some examples - ')
        number_i = min(len(images), 50)
        plt.figure()
        fig, axis = plt.subplots(number_i, 2, figsize=(10, 50)) #2 rows for input, output
        fig.tight_layout()
        fig.suptitle(' - examples of network - ')
        for i in range(min(self.batch_size, number_i)):
            input_map = indices2image(data[0][i], self.image_size)
            axis[i, 0].imshow(input_map)
            im = indice_map2image(outputs[i], self.image_size).cpu().numpy()
            normalized = (im - im.min() ) / (im.max() - im.min())
            axis[i, 1].imshow(normalized)
        img_name = self.path_name + '/example_' + str(self.num_timesteps) + '.png'
        plt.savefig(img_name)
        plt.close()
        if self.sacred :
            self.sacred.add_artifact(img_name, content_type='image')

    def save_example(self, observations, rewards, number):
        noms = []
        dir = self.log_dir + '/example/' + str(self.num_timesteps) + '/ex_number' + str(number)
        if dir is not None:
            os.makedirs(dir, exist_ok=True)

            for i, obs in enumerate(observations):
                save_name = dir + '/' + str(i) + '_r=' + str(rewards[i]) + '.png'  #[np.array(img) for i, img in enumerate(images)
                image = self.norm_image(obs, scale=1)
                imsave(save_name, image)
                noms.append(save_name)

        # Save the imges as video
        video_name = dir + 'r=' + str(np.sum(rewards)) + '.mp4'
        clips = [ImageClip(m).set_duration(2)
              for m in noms]

        concat_clip = concatenate_videoclips(clips, method="compose")
        concat_clip.write_videofile(video_name, fps=24, verbose=None, logger=None)

        if self.sacred :
            self.sacred.get_logger().report_media('video', 'Res_' + str(number) + '_Rwd=' + str(np.sum(rewards)),
                                                  iteration=self.num_timesteps // self.check_freq,
                                                  local_path=video_name)
        del concat_clip
        del clips


    def norm_image(self, image, type=None, scale=10):
        image = np.kron(image, np.ones((scale, scale)))
        if type=='rgb':
            ret = np.empty((image.shape[0], image.shape[0], 3), dtype=np.uint8)
            ret[:, :, 0] = image.copy()
            ret[:, :, 1] = image.copy()
            ret[:, :, 2] = image.copy()
            image = ret.copy()
        return (255 * (image - np.min(image)) / (np.max(image) - np.min(image))).astype(np.uint8)

    def save_gif(self, observations, rewards):
        # print(observations)
        # print(rewards)
        # length = min(len(observations), 10)
        # observations = 255 * ((np.array(observations) + 1) / (np.max(observations) + 1)).astype(np.uint8)
        save_name = self.log_dir + '/example' + str(self.num_timesteps) + '.gif'
        images = [self.norm_image(observations[i]) for i in range(len(observations)) if rewards[i] >= 0]  #[np.array(img) for i, img in enumerate(images)]
        # imageio.mimsave(save_name, images, fps=1)
        if self.sacred :
            self.sacred.get_logger().report_media('GIF', 'isgif', iteration=self.num_timesteps, local_path=save_name)


    def save_video(self, observations, rewards):
        save_name = self.log_dir + '/example' + str(self.num_timesteps) + '.mp4'
        images = [self.norm_image(observations[i], type='rgb') for i in range(len(observations)) if rewards[i] >= 0]  #[np.array(img) for i, img in enumerate(images)

        clips = [ImageClip(m).set_duration(2)
              for m in images]

        concat_clip = concatenate_videoclips(clips, method="compose").resize(100)
        concat_clip.write_videofile(save_name, fps=24, verbose=False)
        if self.sacred :
            self.sacred.get_logger().report_media('video', 'results', iteration=self.num_timesteps, local_path=save_name)


    def _on_step(self) -> bool:
        """
        In addition to EvalCallback we needs
        Examples of eviroonment elements -> Save them as gif for exemple
        Statistics to save -> save as plot and in database
            -> reward, length, loss, additional metrics (accuraccy, best move ?)
        """
        # super(MonitorCallback, self)._on_step()
        if self.num_timesteps % self.check_freq == 0 :
            episode_rewards, episode_lengths = [], []
            for i in range(self.n_eval_episodes):
                obs = self.env.reset()
                done, state = False, None
                if self.sequence :
                    observations = [self.env.get_image_representation()]
                else :
                    observations = [obs.copy()]
                episode_reward = [0.0]
                episode_lengths.append(0)
                while not done:
                    action, state = self.model.predict(obs, state=state, deterministic=False)
                    new_obs, reward, done, _info = self.env.step(action)

                    obs = new_obs

                    if self.sequence:
                        observations.append(self.env.get_image_representation())
                    else :
                        observations.append(obs.copy())
                    episode_reward.append(reward)
                    episode_lengths[-1] += 1
                    if self.render:
                        self.env.render()
                episode_rewards.append(np.sum(episode_reward))
                # self.save_gif(observations, episode_reward)
                # self.save_example(observations, episode_reward,number=i)

            self.statistics['reward'].append(np.mean(episode_rewards))
            self.statistics['std_reward'].append(np.std(episode_rewards))
            self.statistics['step_reward'].append(np.mean([episode_rewards[i]/episode_lengths[i] for i in range(len(episode_lengths))]))
            self.statistics['duration'].append(np.mean(episode_lengths))
            # self.statistics['policy_loss'].append(self.model.pg_loss.numpy())
            # self.statistics['value_loss'].append(self.model.vf_loss.numpy())
            # self.statistics['policy_entropy'].append(self.model.entropy.numpy())
            self.plot_statistics()

            # Save best model
            if self.statistics['reward'][-1] == np.max(self.statistics['reward']):
                save_path = self.log_dir + '/best_model'
                if self.verbose > 0:
                    print("Saving new best model to {}".format(save_path))
                self.model.save(save_path)

            return True
