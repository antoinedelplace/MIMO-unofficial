import sys
sys.path.append(".")

from utils.general_utils import set_memory_limit, parse_args
from training.training_pipeline import TrainingPipeline

def main(cpu_memory_limit_gb=60):
    set_memory_limit(cpu_memory_limit_gb)

    pipe = TrainingPipeline()

    pipe.train()

if __name__ == "__main__":
    args = parse_args(main)
    main(**vars(args))


# accelerate config
# accelerate launch training/main.py