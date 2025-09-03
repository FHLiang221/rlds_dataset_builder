from typing import Iterator, Tuple, Any
import os
import glob
import json
import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds
import tensorflow_hub as hub


class SimplerEnvSuccessDataset(tfds.core.GeneratorBasedBuilder):
    """DatasetBuilder for SimplerEnv successful demonstration data."""

    VERSION = tfds.core.Version('1.0.0')
    RELEASE_NOTES = {
        '1.0.0': 'Initial release with successful demonstrations only.',
    }

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._embed = hub.load("https://tfhub.dev/google/universal-sentence-encoder-large/5")

    def _info(self) -> tfds.core.DatasetInfo:
        """Dataset metadata (homepage, citation,...)."""
        return self.dataset_info_from_configs(
            features=tfds.features.FeaturesDict({
                'steps': tfds.features.Dataset({
                    'observation': tfds.features.FeaturesDict({
                        'image': tfds.features.Image(
                            shape=(224, 224, 3),
                            dtype=np.uint8,
                            encoding_format='png',
                            doc='Main camera RGB observation from SimplerEnv.',
                        ),
                        'proprio': tfds.features.Tensor(
                            shape=(8,),
                            dtype=np.float32,
                            doc='Proprioceptive state [EEF pose (7D: x,y,z,qx,qy,qz,qw), gripper (1D)].',
                        ),
                    }),
                    'action': tfds.features.Tensor(
                        shape=(7,),
                        dtype=np.float32,
                        doc='Robot action vector.',
                    ),
                    'discount': tfds.features.Scalar(
                        dtype=np.float32,
                        doc='Discount if provided, default to 1.'
                    ),
                    'reward': tfds.features.Scalar(
                        dtype=np.float32,
                        doc='Reward if provided, 1 on final step for demos.'
                    ),
                    'is_first': tfds.features.Scalar(
                        dtype=np.bool_,
                        doc='True on first step of the episode.'
                    ),
                    'is_last': tfds.features.Scalar(
                        dtype=np.bool_,
                        doc='True on last step of the episode.'
                    ),
                    'is_terminal': tfds.features.Scalar(
                        dtype=np.bool_,
                        doc='True on last step of the episode if it is a terminal step.'
                    ),
                    'language_instruction': tfds.features.Text(
                        doc='Language instruction for the task.'
                    ),
                    'language_embedding': tfds.features.Tensor(
                        shape=(512,),
                        dtype=np.float32,
                        doc='Universal sentence encoder embedding of language instruction.'
                    ),
                }),
                'episode_metadata': tfds.features.FeaturesDict({
                    'file_path': tfds.features.Text(
                        doc='Path to the original data file.'
                    ),
                    'episode_id': tfds.features.Scalar(
                        dtype=np.int64,
                        doc='Original episode ID.'
                    ),
                    'env_name': tfds.features.Text(
                        doc='Environment name.'
                    ),
                }),
            }))

    def _split_generators(self, dl_manager: tfds.download.DownloadManager):
        """Define data splits."""
        # For now, we'll put all successful episodes in the train split
        # You can modify this to create train/val splits as needed
        return {
            'train': self._generate_examples(),
        }

    def _generate_examples(self) -> Iterator[Tuple[str, Any]]:
        """Generator of examples for each split."""
        
        def _parse_example(episode_path):
            # Load the npz file
            data = np.load(episode_path, allow_pickle=True)
            
            # Extract episode metadata
            episode_id = int(data['episode_id'])
            env_name = str(data['env_name'])
            language_instruction = str(data['language'])
            
            # Extract trajectory data
            images = data['images']  # Shape: (T, 224, 224, 3)
            actions = data['actions']  # Shape: (T, 7)
            states = data['states']  # Shape: (T, 8)
            rewards = data['rewards']  # Shape: (T,)
            is_terminal = data['is_terminal']  # Shape: (T,)
            is_first = data['is_first']  # Shape: (T,)
            
            # Compute language embedding
            language_embedding = self._embed([language_instruction])[0].numpy()
            
            # Build episode steps
            episode = []
            seq_len = len(images)
            
            for i in range(seq_len):
                # Ensure image is uint8
                image = images[i].astype(np.uint8)
                
                # Use the combined 8D state as 'proprio' (matches OpenVLA-OFT format)
                proprio_state = states[i].astype(np.float32)  # [8D: EEF pose + gripper]
                
                episode.append({
                    'observation': {
                        'image': image,
                        'proprio': proprio_state,
                    },
                    'action': actions[i].astype(np.float32),
                    'discount': 1.0,
                    'reward': float(rewards[i]),
                    'is_first': bool(is_first[i]),
                    'is_last': i == (seq_len - 1),
                    'is_terminal': bool(is_terminal[i]),
                    'language_instruction': language_instruction,
                    'language_embedding': language_embedding,
                })
            
            # Create output sample
            sample = {
                'steps': episode,
                'episode_metadata': {
                    'file_path': episode_path,
                    'episode_id': episode_id,
                    'env_name': env_name,
                }
            }
            
            return sample

        # Find all successful episodes across all task directories
        base_path = "/project/fhliang/projects/SimplerEnv/demo_collection/collected_data"
        success_files = []
        
        # Traverse all task directories and collect successful episodes
        for task_dir in glob.glob(os.path.join(base_path, "*_all_episodes")):
            success_dir = os.path.join(task_dir, "successes")
            if os.path.exists(success_dir):
                success_files.extend(glob.glob(os.path.join(success_dir, "success_episode_*.npz")))
        
        print(f"Found {len(success_files)} successful episodes to process")
        
        # Process each successful episode using sequential numbering like the switch dataset
        episode_count = 0
        for episode_path in sorted(success_files):
            try:
                sample = _parse_example(episode_path)
                episode_count += 1
                yield f"success_episode_{episode_count:06d}", sample
            except Exception as e:
                print(f"Error processing {episode_path}: {e}")
                continue
            
# cd rlds_dataset_builder/simpler_env_success_dataset      
# tfds build --overwrite
# python vis_data.py simpler_env_success_dataset
# dataset location: ~/tensorflow_datasets/simpler_env_success_dataset/1.0.0/
# /home/fhliang/tensorflow_datasets/simpler_env_success_dataset/1.0.0/