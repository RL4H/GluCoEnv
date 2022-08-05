import torch
torch.set_default_dtype(torch.float32)
import warnings


def risk_index(BG, horizon):
    # BG is in mg/dL
    # horizon in samples
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        BG_to_compute = BG[:, -horizon:]
        BG_to_compute[BG_to_compute < 1] = 1
        fBG = 1.509 * (torch.log(BG_to_compute) ** 1.084 - 5.381)
        rl = 10 * (fBG * (fBG < 0)) ** 2
        rh = 10 * (fBG * (fBG > 0)) ** 2
        rl_count = torch.count_nonzero(rl, dim=1)
        rh_count = torch.count_nonzero(rh, dim=1)
        LBGI = torch.unsqueeze(torch.nan_to_num(torch.sum(rl, 1) / rl_count), 1)
        HBGI = torch.unsqueeze(torch.nan_to_num(torch.sum(rh, 1) / rh_count), 1)
        RI = LBGI + HBGI
    return (LBGI, HBGI, RI)


# X_MAX = 0
# X_MIN = -risk_index(torch.ones(1,1) * 600, 1)[-1]
# print(X_MAX)
def reward(bg_hist, **kwargs):
    X_MAX, X_MIN = 0, -100
    r = -risk_index(bg_hist, 1)[-1]
    rew = ((r - X_MIN) / (X_MAX - X_MIN))
    failure1 = (bg_hist[:, -1:] < 40)
    failure2 = (bg_hist[:, -1:] > 600)
    rew = rew * torch.logical_not(failure1 + failure2) + (failure1 * -15)
    return rew


if __name__ == '__main__':
    # horizon = 2
    # risk_hist = []
    # t = []
    # for x in range(40, 600):
    #     bg = torch.ones(2, 1) * x
    #     l, h, ri = risk_index(bg, horizon)
    #     risk_hist.append(ri[0])
    #     t.append(x)
    # import matplotlib.pyplot as plt
    # plt.plot(t, risk_hist)
    # plt.show()

    horizon = 2
    risk_hist = []
    t = []
    for x in range(0, 610):
        bg = torch.ones(2, 1) * x
        ri = reward(bg)
        risk_hist.append(ri[0])
        t.append(x)
    import matplotlib.pyplot as plt

    plt.plot(t, risk_hist)
    plt.show()
