from src.models.mlp import build_MLP
from src.models.vit import build_VisionTransformer

def get_model(architecture, config, verbose=True):
    if 'MLP' in architecture:
        model = build_MLP(config)
    elif 'VisionTransformer' in architecture:
        model = build_VisionTransformer(config)
    else:
        raise NotImplementedError(
            f'The model architecture {architecture} is not implemented yet..')
    if verbose:
        print(f'Loaded model with {architecture} architecture, input shape {config.data_shape}, {config.num_classes} classes.')

    if 'MaskDropout' in list(module.__class__.__name__ for module in model.modules()):
        model.track_dropout_mask = lambda track=True: track_dropout_mask(model, track)
        model.use_dropout_mask = lambda use=True: use_dropout_mask(model, use)
        model.get_dropout_layers = lambda positions=None: get_dropout_layers(model, positions)
        if verbose:
            print('Attatched MaskDropout track functionalities!')
    return model

def track_dropout_mask(model, track=True):
    for module in model.modules():
        if module.__class__.__name__ == 'MaskDropout':
            module.track_mask = track

def use_dropout_mask(model, use=True):
    for module in model.modules():
        if module.__class__.__name__ == 'MaskDropout':
            module.use_mask = use

def get_dropout_layers(model, positions=None):
    all_dl_layers = list(module for module in model.modules() if module.__class__.__name__ == 'MaskDropout')
    if positions is not None:
        return [dl for i, dl in enumerate(all_dl_layers) if i in positions]
    else: 
        return all_dl_layers