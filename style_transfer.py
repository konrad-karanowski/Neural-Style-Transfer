import os

import hydra
from utils import read_img, train_model, VideoWriter


@hydra.main(config_path='config', config_name='default.yaml')
def main(config):
    content_name = config.content_img.split("/")[-1].split(".")[0]
    output_path = os.path.join(config.outputs, content_name)
    if not os.path.exists(output_path):
        os.mkdir(output_path)
    save_name = os.path.join(output_path,
                             f'{content_name}_{config.style_img.split("/")[-1].split(".")[0]}_sw_{config.style_weight}_cw_{config.content_weight}_tw_{config.variance_weight}')
    video_writer = VideoWriter(
        fps=config.fps,
        show=config.show,
        save_freq=config.save_freq,
        name=save_name
    )
    content_img = read_img(config.content_img, config.target_shape, config.normalize)
    style_img = read_img(config.style_img, config.target_shape, config.normalize)
    train_model(
        content_img=content_img,
        style_img=style_img,
        lr=config.lr,
        max_epochs=config.epochs,
        content_weight=config.content_weight,
        style_weight=config.style_weight,
        variance_weight=config.variance_weight,
        content_layers=config.content_layers,
        style_layers=config.style_layers,
        initialization_method=config.init_method,
        optimizer_str=config.optimizer,
        video_writer=video_writer
    )
    video_writer.save()
    print('End')


if __name__ == '__main__':
    main()
