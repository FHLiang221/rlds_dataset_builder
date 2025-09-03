# SimplerEnv Success Dataset

This dataset builder converts SimplerEnv collected demonstration data to RLDS format, filtering to include only successful episodes.

## Data Format

The dataset processes `.npz` files from SimplerEnv's demo collection that contain:
- `images`: RGB observations (224x224x3)
- `actions`: 7-DOF robot actions  
- `states`: 8-dimensional robot state
- `rewards`: Per-step rewards
- `is_terminal`: Terminal step flags
- `is_first`: First step flags
- `language`: Natural language instruction
- `episode_id`: Episode identifier
- `env_name`: Environment name

## Usage

Build the dataset:
```bash
cd /project/fhliang/projects/rlds_dataset_builder/simpler_env_success_dataset
tfds build --overwrite
```

The dataset automatically discovers all successful episodes from:
`/project/fhliang/projects/SimplerEnv/demo_collection/collected_data/*/successes/`

## Output Format

The RLDS dataset includes:
- **observations**: RGB image (224x224x3) and robot state (8D)
- **actions**: 7-DOF robot action vectors
- **language_instruction**: Natural language task description
- **language_embedding**: Universal sentence encoder embedding (512D)
- **episode_metadata**: Original file path, episode ID, environment name