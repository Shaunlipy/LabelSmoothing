import torch

class LabelSmooth1(torch.nn.Module):
    def __init__(self, epsilon=0.1, reduction='none'):
        super().__init__()
        self.epsilon = epsilon
        self.reduction = reduction
    def forward(self, preds, target):
        n = preds.size()[-1]
        log_preds = torch.nn.functional.log_softmax(preds, dim=-1)
        loss = -log_preds.sum(dim=-1)
        nll = torch.nn.functional.nll_loss(log_preds, target, reduction='none')
        return (self.epsilon* loss / n + (1-self.epsilon) * nll)

class LabelSmooth2(torch.nn.Module):
    def __init__(self, epsilon=0.1, reduction='none'):
        super().__init__()
        self.epsilon = epsilon
        self.reduction = reduction
    def forward(self, preds, target):
        n_class = preds.size(-1)
        one_hot = torch.zeros_like(preds).scatter(1, target.view(-1, 1), 1)
        one_hot = one_hot * (1 - self.epsilon) + (1 - one_hot) * self.epsilon / (n_class - 1)
        log_prb = torch.nn.functional.log_softmax(preds, dim=1)
        loss = -(one_hot * log_prb).sum(dim=1)
        return loss

if __name__ == '__main__':
    torch.manual_seed(0)
    x = torch.randn(2, 5)
    y = torch.tensor([0,2]).long()
    criterion1 = LabelSmooth1()
    criterion2 = LabelSmooth2()
    loss1 = criterion1(x, y)
    loss2 = criterion2(x, y)
    print('1: {}'.format(loss1))
    print('2: {}'.format(loss2))
