DATA_ROOT = '/home/tuannh/Documents/datasets/anomaly_datasets'

ANOMALY_CONFIG = dict(
    prompt_temperature=0.07,
    seg_text_mode="prompted",
    early_pool=True,
)


def add_anomaly_arguments(parser, str2bool):
    parser.add_argument('--selected_layers', type=int, nargs='+', default=None)
    for name, default in ANOMALY_CONFIG.items():
        if isinstance(default, bool):
            parser.add_argument('--' + name, type=str2bool, default=default)
        elif isinstance(default, list):
            parser.add_argument('--' + name, type=float, nargs='+', default=default)
        else:
            parser.add_argument('--' + name, type=type(default), default=default)
