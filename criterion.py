import torch.nn as nn
from distance import KLDiv
from torch.autograd import Variable

class forward_MCL(nn.Module):
    # Forward Meta Classification Likelihood (MCL)

    eps = 1e-12 # Avoid calculating log(0). Use the small value of float16.

    def __init__(self):
        super(forward_MCL, self).__init__()
        return

    def forward(self, prob1, prob2, s_label, q):
        P = prob1.mul(prob2)
        P = P.sum(1)
        P = P * q[0][0] + (1 - P) * q[1][0]
        P.mul_(s_label).add_(s_label.eq(-1).type_as(P))
        negLog_P = -P.add_(forward_MCL.eps).log_()
        return negLog_P.mean()

class reweight_MCL(nn.Module):
    # Reweight Meta Classification Likelihood (MCL)

    eps = 1e-12 # Avoid calculating log(0). Use the small value of float16.

    def __init__(self):
        super(reweight_MCL, self).__init__()
        return

    def forward(self, prob1, prob2, s_label, q):
        cleanP1 = prob1.mul(prob2)
        cleanP1 = cleanP1.sum(1)
        noiseP1 = cleanP1 * q[0][0] + (1 - cleanP1) * q[1][0]
        coef1 = cleanP1.div(noiseP1)         # coefficient for instance with \hat{Y} = 1
        coef0 = (1 - cleanP1).div(1 - noiseP1)      # coefficient for instance with \hat{Y} = 0
        coef0[s_label == 1] = coef1[s_label == 1]       # denote the both coefficient by coef0
        coef0 = Variable(coef0, requires_grad=True)
        cleanP1.mul_(s_label).add_(s_label.eq(-1).type_as(cleanP1))
        cleanP1 = cleanP1.mul(coef0)
        negLog_P = -cleanP1.add_(reweight_MCL.eps).log_()
        return negLog_P.mean()
