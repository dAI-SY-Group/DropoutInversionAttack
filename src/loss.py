import torch

def TV(x):
    """Anisotropic TV."""
    if len(x.shape) == 3: #single image
        dx = torch.mean(torch.abs(x[:, :, :-1] - x[:, :, 1:]))
        dy = torch.mean(torch.abs(x[:, :-1, :] - x[:, 1:, :]))
    else:
        dx = torch.mean(torch.abs(x[:, :, :, :-1] - x[:, :, :, 1:]))
        dy = torch.mean(torch.abs(x[:, :, :-1, :] - x[:, :, 1:, :]))
    return dx + dy


class GCosineDistance:
    def __init__(self):
        super().__init__()
        self.format = '.6f'
        self.name = 'GCosineDistance'
        self.subject_to = 'min'
    def __call__(self, prediction, target):
        total_loss = 0
        p_norm = 0
        t_norm = 0
        for layer, ((p_name, p_grad), (t_name, t_grad)) in enumerate(zip(prediction.items(), target.items())):
            assert p_name == t_name, f'Layer names for gradients do not match at layer {layer}!. {p_name} != {t_name} !'
            partial_loss = (p_grad * t_grad).sum()
            partial_p_norm = p_grad.pow(2).sum()
            partial_t_norm = t_grad.pow(2).sum()
            
            total_loss += partial_loss
            p_norm += partial_p_norm
            t_norm += partial_t_norm
        total_loss = 1 - total_loss / ((p_norm.sqrt()*t_norm.sqrt())+1e-8)
        return total_loss