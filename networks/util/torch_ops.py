import torch


def rgb2hsv(rgb):
    shape = rgb.size()
    assert shape[-1] == 3
    if len(shape) == 2:
        r, g, b = rgb[:, 0], rgb[:, 1], rgb[:, 2]
        mx = torch.max(rgb, dim=1)[0]
        mn = torch.min(rgb, dim=1)[0]
        df = mx - mn + 1e-6
        hr = (60 * ((g - b) / df) + 360) % 360
        hg = (60 * ((b - r) / df) + 120) % 360
        hb = (60 * ((r - g) / df) + 240) % 360
        m0 = torch.eq(mx, mn).detach().float()
        mr = torch.eq(mx, r).detach().float()
        mg = torch.eq(mx, g).detach().float()
        mb = torch.eq(mx, b).detach().float()
        h = (1.0 - m0) * (hr * mr + hg * mg + hb * mb)
        s = df / (mx + 1e-6)
        v = mx
        return h, s, v
    elif len(shape) == 3:
        rgb = rgb.contiguous().view((-1, 3))
        h, s, v = rgb2hsv(rgb)
        h = h.view((shape[0], shape[1]))
        s = s.view((shape[0], shape[1]))
        v = v.view((shape[0], shape[1]))
        return h, s, v
    else:
        raise NotImplementedError


if __name__ == '__main__':
    ###################################################################
    # test rgb2hsv
    import numpy as np
    import skimage.color as sk_clr
    clr = np.random.rand(10, 1, 3).astype(np.float32)
    # clr_uint8 = np.uint8(clr * 255)
    hsv = sk_clr.rgb2hsv(clr)
    clr_ = torch.from_numpy(clr)
    h, s, v = rgb2hsv(clr_)
    h = h.detach().cpu().numpy()
    s = s.detach().cpu().numpy()
    v = v.detach().cpu().numpy()
    print(hsv[:, :, 0])
    print(h / 360)
    print(np.sum(np.abs(hsv[:, :, 0] - h / 360)))
