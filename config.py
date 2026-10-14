DATA_ROOT = '/home/tuannh/Documents/datasets/anomaly_datasets'

ANOMALY_CONFIG = dict(
    use_tail_branch=True,
    use_prompt_ensemble=True,
    use_self_reference=True,
    use_multi_tail=True,
    use_adaptive_tail=True,
    use_uncertainty_fusion=True,
    prompt_temperature=0.07,
    normal_fraction=0.70,
    num_normal_prototypes=3,
    kmeans_iterations=5,
    tail_ratios=[0.01, 0.03, 0.10, 0.25],
    tail_gate_hidden_dim=16,
    ranking_margin=0.2,
    lambda_rank=0.0,
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
