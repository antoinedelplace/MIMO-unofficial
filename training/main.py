from utils.general_utils import set_memory_limit, parse_args
from training_pipeline import TrainingPipeline

def make_grid(images, rows, cols):
    w, h = images[0].size
    grid = Image.new('RGB', size=(cols*w, rows*h))
    for i, image in enumerate(images):
        grid.paste(image, box=(i%cols*w, i//cols*h))
    return grid

def evaluate(config, epoch, pipeline):
    # Sample some images from random noise (this is the backward diffusion process).
    # The default pipeline output type is `List[PIL.Image]`
    images = pipeline(
        batch_size = config.eval_batch_size, 
        generator=torch.manual_seed(config.seed),
    ).images

    # Make a grid out of the images
    image_grid = make_grid(images, rows=4, cols=4)

    # Save the images
    test_dir = os.path.join(config.output_dir, "samples")
    os.makedirs(test_dir, exist_ok=True)
    image_grid.save(f"{test_dir}/{epoch:04d}.png")

def main(batch_size=16,
         workers=8,
         cpu_memory_limit_gb=60):
    set_memory_limit(cpu_memory_limit_gb)

    pipe = TrainingPipeline()

    pipe.train()

if __name__ == "__main__":
    args = parse_args(main)
    main(**vars(args))


#