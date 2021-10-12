import torch
import torch.nn.functional as F
from torch.autograd import Variable


def normalize(x, mean, std):
    mean = torch.tensor(mean).view(3, 1, 1).expand(x.shape[0], 3, x.shape[2], x.shape[2]).to(x.device)
    std = torch.tensor(std).view(3, 1, 1).expand(x.shape[0], 3, x.shape[2], x.shape[2]).to(x.device)
    return (x - mean)/std


# untarget attack under threshold
def attack_magnitude_untarget(model,
              x_natural,
              y,
              ori_logit,
              device,
              step_size=2/255,
              epsilon=16/255,
              perturb_steps=100,
              threshold=4.0,
              distance='l_inf',
              isnormalize=True,
              mean=None,
              std=None
           ):
    model.eval()
    if distance == 'l_inf':
        x_adv = x_natural.detach() + 0.001 * torch.randn(x_natural.shape).to(device).detach()    # random start
    elif distance == 'l_2':
        delta = torch.zeros_like(x_natural).to(device).detach()
        delta.normal_()
        d_flat = delta.view(delta.size(0), -1)
        n = d_flat.norm(p=2, dim=1).view(delta.size(0), 1, 1, 1)
        r = torch.zeros_like(n).uniform_(0, 1)
        delta *= r / n * epsilon
        x_adv = x_natural.detach() + delta

    x_adv = torch.clamp(x_adv, min=0, max=1)     # x_adv is in [0,1]

    for _ in range(perturb_steps):
        x_adv.requires_grad_()
        with torch.enable_grad():
            if isnormalize:
                outs = model(normalize(x_adv, mean, std))
            else:
                outs = model(x_adv)
            if (type(outs).__name__ == 'tuple'):
                output = outs[0]
            else:
                output = outs
            loss_ce = F.cross_entropy(output, y)

        grad = torch.autograd.grad(loss_ce, [x_adv])[0]
        is_valid = False
        while not is_valid:
            if distance == 'l_inf':
                x_adv_tmp = x_adv.detach() + step_size * torch.sign(grad.detach())
                x_adv_tmp = torch.min(torch.max(x_adv_tmp, x_natural - epsilon), x_natural + epsilon)
                x_adv_tmp = torch.clamp(x_adv_tmp, 0.0, 1.0)
            elif distance == 'l_2':
                g_norm = torch.norm(grad.view(grad.shape[0], -1), dim=1).view(-1, 1, 1, 1)
                scaled_g = grad / (g_norm + 1e-10)
                x_adv_tmp = x_adv.detach() + step_size * scaled_g
                delta = x_adv_tmp - x_natural
                delta = delta.renorm(p=2, dim=0, maxnorm=epsilon)
                x_adv_tmp = torch.clamp(x_natural + delta, 0.0, 1.0)

            if isnormalize:
                outs = model(normalize(x_adv_tmp, mean, std))
            else:
                outs = model(x_adv_tmp)
            if type(outs).__name__ == 'tuple':
                output = outs[0]
            else:
                output = outs

            pred_adv = torch.argmax(output, dim=1)
            if pred_adv[0] == y[0]:
                x_adv = x_adv_tmp
                break  # to next iteration
            # out_adv = output[:, pred_adv[0]]
            out_gt = output[:, y[0]]
            dist = ori_logit - out_gt
            if step_size < 1e-6:
                return x_adv_tmp
            # print('step_size:', step_size)
            # print('dist:', ori_logit, out_gt, dist)
            if dist > threshold + 0.5:
                is_valid = False
                step_size /= 2
            elif dist > threshold:
                print('Successful attacking to threshold.')
                return x_adv_tmp
            else:
                is_valid = True
                x_adv = x_adv_tmp

    print('Not successful to threshold after attacking {} steps.'.format(perturb_steps))
    return x_adv


# target attack under threshold
def attack_magnitude_target(model,
              x_natural,
              y,
              target_y,
              device,
              step_size=2/255,
              epsilon=16/255,
              perturb_steps=100,
              threshold=5.0,
              distance='l_inf',
              isnormalize=True,
              mean=None,
              std=None
           ):
    model.eval()
    if distance == 'l_inf':
        x_adv = x_natural.detach() + 0.001 * torch.randn(x_natural.shape).to(device).detach()    # random start
    elif distance == 'l_2':
        delta = torch.zeros_like(x_natural).to(device).detach()
        delta.normal_()
        d_flat = delta.view(delta.size(0), -1)
        n = d_flat.norm(p=2, dim=1).view(delta.size(0), 1, 1, 1)
        r = torch.zeros_like(n).uniform_(0, 1)
        delta *= r / n * epsilon
        x_adv = x_natural.detach() + delta
    x_adv = torch.clamp(x_adv, min=0, max=1)     # x_adv is in [0,1]

    for _ in range(perturb_steps):
        x_adv.requires_grad_()
        with torch.enable_grad():
            if isnormalize:
                outs = model(normalize(x_adv, mean, std))
            else:
                outs = model(x_adv)
            if (type(outs).__name__ == 'tuple'):
                output = outs[0]
            else:
                output = outs
            loss = output[:, target_y] - output[:, y]

        grad = torch.autograd.grad(loss, [x_adv])[0]     # 是放入x_adv还是  normalize(x_adv)
        is_valid = False
        while not is_valid:
            if distance == 'l_inf':
                x_adv_tmp = x_adv.detach() + step_size * torch.sign(grad.detach())
                x_adv_tmp = torch.min(torch.max(x_adv_tmp, x_natural - epsilon), x_natural + epsilon)
                x_adv_tmp = torch.clamp(x_adv_tmp, 0.0, 1.0)
            elif distance == 'l_2':
                g_norm = torch.norm(grad.view(grad.shape[0], -1), dim=1).view(-1, 1, 1, 1)
                scaled_g = grad / (g_norm + 1e-10)
                x_adv_tmp = x_adv.detach() + step_size * scaled_g
                # print('change mean:', torch.mean(step_size * scaled_g))
                delta = x_adv_tmp - x_natural
                delta = delta.renorm(p=2, dim=0, maxnorm=epsilon)
                x_adv_tmp = torch.clamp(x_natural + delta, 0.0, 1.0)

            if isnormalize:
                outs = model(normalize(x_adv_tmp, mean, std))
            else:
                outs = model(x_adv_tmp)

            if type(outs).__name__ == 'tuple':
                output = outs[0]
            else:
                output = outs

            out_gt = output[:, y[0]]
            out_target = output[:, target_y[0]]
            dist = out_gt - out_target
            # print('gt - target dist:', dist)
            # print('*********')
            if dist < threshold - 0.5:
                is_valid = False
                step_size /= 2
            elif dist > threshold:
                is_valid = True
                x_adv = x_adv_tmp
            else:
                print('Successful attacking.')
                return x_adv_tmp

    print('Not successful after attacking {} steps.'.format(perturb_steps))
    return x_adv


