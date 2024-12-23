import sys
sys.path.append(".")

from mimo.utils.general_utils import set_memory_limit, parse_args
from mimo.training.training_pipeline import TrainingPipeline

def main(cpu_memory_limit_gb=60, training_config="./mimo/configs/training/cfg_phase2.yaml"):
    set_memory_limit(cpu_memory_limit_gb)

    pipe = TrainingPipeline(training_config)

    pipe.train()

if __name__ == "__main__":
    args = parse_args(main)
    main(**vars(args))


# accelerate config
#    - multi-GPU
#    - numa efficiency
#    - fp16

# accelerate launch mimo/training/main.py -c 1540

# mlflow ui -p 5003