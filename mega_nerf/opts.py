import configargparse


def get_opts_base():
    parser = configargparse.ArgParser(config_file_parser_class=configargparse.YAMLConfigFileParser)
    parser.add_argument('--config_file', is_config_file=True)

    parser.add_argument('--dataset_type', type=str, default='filesystem', choices=['filesystem', 'memory'])
    parser.add_argument('--chunk_paths', type=str, nargs='+', default=None)
    parser.add_argument('--num_chunks', type=int, default=200)
    parser.add_argument('--disk_flush_size', type=int, default=5000000)
    parser.add_argument('--train_every', type=int, default=1)

    parser.add_argument('--cluster_mask_path', type=str, default=None)

    parser.add_argument('--ckpt_path', type=str, default=None)
    parser.add_argument('--container_path', type=str, default=None)

    parser.add_argument('--near', type=float, default=0)
    parser.add_argument('--far', type=float, default=None)
    parser.add_argument('--ray_altitude_range', nargs='+', type=float, default=None)
    parser.add_argument('--coarse_samples', type=int, default=256,
                        help='number of coarse samples')
    parser.add_argument('--fine_samples', type=int, default=512,
                        help='number of additional fine samples')

    parser.add_argument('--train_scale_factor', type=int, default=1)

    parser.add_argument('--pos_xyz_dim', type=int, default=12)
    parser.add_argument('--pos_dir_dim', type=int, default=4)
    parser.add_argument('--layers', type=int, default=8)
    parser.add_argument('--skip_layers', type=int, nargs='+', default=[4])
    parser.add_argument('--layer_dim', type=int, default=256)
    parser.add_argument('--bg_layer_dim', type=int, default=256)
    parser.add_argument('--appearance_dim', type=int, default=48, help='number of embeddings for appearance')

    parser.add_argument('--use_cascade', default=False, action='store_true')

    parser.add_argument('--train_mega_nerf', type=str, default=None)
    parser.add_argument('--boundary_margin', type=int, default=1.15)

    parser.add_argument('--sh_deg', type=int, default=None)

    parser.add_argument('--no_center_pixels', dest='center_pixels', default=True, action='store_false')
    parser.add_argument('--no_shifted_softplus', dest='shifted_softplus', default=True, action='store_false')

    parser.add_argument('--batch_size', type=int, default=1024, help='batch size')
    parser.add_argument('--image_pixel_batch_size', type=int, default=64 * 1024)
    parser.add_argument('--model_chunk_size', type=int, default=32 * 1024,
                        help='chunk size to split the input to avoid OOM')

    parser.add_argument('--perturb', type=float, default=1.0, help='factor to perturb depth sampling points')
    parser.add_argument('--noise_std', type=float, default=1.0, help='std dev of noise added to regularize sigma')

    parser.add_argument('--lr', type=float, default=5e-4, help='learning rate')
    parser.add_argument('--lr_decay_factor', type=float, default=0.1)

    parser.add_argument('--no_bg_nerf', dest='bg_nerf', default=True, action='store_false')
    parser.add_argument('--no_ellipse_bounds', dest='ellipse_bounds', default=True, action='store_false')

    parser.add_argument('--train_iterations', type=int, default=500000)
    parser.add_argument('--val_interval', type=int, default=50000)
    parser.add_argument('--ckpt_interval', type=int, default=10000)

    parser.add_argument('--no_amp', dest='amp', default=True, action='store_false')
    parser.add_argument('--detect_anomalies', default=False, action='store_true')
    parser.add_argument('--random_seed', type=int, default=42)

    return parser
