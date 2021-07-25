import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.autograd import Variable
from pairwise import PairEnum



class Learner_Likelihood(nn.Module):
    def __init__(self, model, criterion, optimizer, scheduler):
        super(Learner_Likelihood, self).__init__()
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.epoch = 0

    def forward(self, x):
        logits = self.model.forward(x)
        prob = F.softmax(logits, dim=1)
        return prob

    def forward_with_criterion(self, inputs, simi, q, mask=None, **kwargs):
        prob = self.forward(inputs)
        prob1, prob2 = PairEnum(prob, mask)
        return self.criterion(prob1, prob2, simi, q), prob

    def learn_class(self, x, targets, q, **kwargs):
        out = self.model.forward(x)
        prob = F.softmax(out, dim=1)
        prob_q = prob.mm(q)
        logprob = prob_q.log()
        nnloss = nn.NLLLoss()
        loss = nnloss(logprob, targets)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        loss = loss.detach()
        prob = prob.detach()
        return loss, prob

    def learn_class_val(self, x, targets, q, **kwargs):
        out = self.model.forward(x)
        prob = F.softmax(out, dim=1)
        prob_q = prob.mm(q)
        logprob = prob_q.log()
        nnloss = nn.NLLLoss()
        loss = nnloss(logprob, targets)
        loss = loss.detach()
        prob = prob.detach()

        return loss, prob

    def learn(self, inputs, targets, q, **kwargs):
        loss, out = self.forward_with_criterion(inputs, targets, q, **kwargs)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        loss = loss.detach()
        out = out.detach()
        return loss, out

    def set_lr(self, lr):
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr

    def est_step_schedule(self, epoch):
        self.epoch = epoch
        self.scheduler.step(self.epoch)
        for param_group in self.optimizer.param_groups:
            print('LR:', param_group['lr'])

    def step_schedule(self, epoch, dataset):
        self.epoch = epoch

        if dataset == 'cifar10':
            if self.epoch < 80:
                self.set_lr(0.001)
            elif self.epoch < 120:
                self.set_lr(0.0001)
            elif self.epoch < 160:
                self.set_lr(0.00001)
            elif self.epoch < 180:
                self.set_lr(0.000001)
            else:
                self.set_lr(0.0000005)
            for param_group in self.optimizer.param_groups:
                print('LR:', param_group['lr'])
        else:
            self.scheduler.step(self.epoch)
            for param_group in self.optimizer.param_groups:
                print('LR:', param_group['lr'])

    def save_model(self, savename):
        model_state = self.model.state_dict()
        if isinstance(self.model, torch.nn.DataParallel):
            # Get rid of 'module' before the name of states
            model_state = self.model.module.state_dict()
        for key in model_state.keys():  # Always save it to cpu
            model_state[key] = model_state[key].cpu()
        print('=> Saving model to:', savename)
        torch.save(model_state, savename + '.pth')
        print('=> Done')

    def snapshot(self, savename, KPI=-1):
        model_state = self.model.state_dict()
        optim_state = self.optimizer.state_dict()
        checkpoint = {
            'epoch': self.epoch,
            'model': model_state,
            'optimizer': optim_state
        }
        print('=> Saving checkpoint to:', savename + '.checkpoint.pth')
        torch.save(checkpoint, savename + '.checkpoint.pth')
        print('=> Done')
        if KPI >= self.KPI:
            print('=> New KPI:', KPI, 'previous KPI:', self.KPI)
            self.KPI = KPI
            self.save_model(savename + '.model')

    def resume(self, resumefile):
        print('=> Loading checkpoint:', resumefile)
        checkpoint = torch.load(resumefile, map_location=lambda storage, loc: storage)  # Load to CPU as the default!
        self.epoch = checkpoint['epoch']
        print('=> resume epoch:', self.epoch)
        self.model.load_state_dict(checkpoint['model'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        print('=> Done')
        return self.epoch

