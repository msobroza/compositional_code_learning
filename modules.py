import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.nn.modules.loss import _Loss
import math


class KLDivLossGumbel(_Loss):
    r"""The `Kullback-Leibler divergence`_ Loss

    KL divergence is a useful distance measure for continuous distributions
    and is often useful when performing direct regression over the space of
    (discretely sampled) continuous output distributions.

    As with `NLLLoss`, the `input` given is expected to contain
    *log-probabilities*, however unlike `ClassNLLLoss`, `input` is not
    restricted to a 2D Tensor, because the criterion is applied element-wise.

    This criterion expects a `target` `Tensor` of the same size as the
    `input` `Tensor`.

    The loss can be described as:

    .. math:: loss(x, target) = 1/n \sum(target_i * (log(target_i) - x_i))

    By default, the losses are averaged for each minibatch over observations
    **as well as** over dimensions. However, if the field
    `size_average` is set to ``False``, the losses are instead summed.

    .. _Kullback-Leibler divergence:
        https://en.wikipedia.org/wiki/Kullback-Leibler_divergence

    Args:
        size_average (bool, optional: By default, the losses are averaged
            for each minibatch over observations **as well as** over
            dimensions. However, if ``False`` the losses are instead summed.
        reduce (bool, optional): By default, the losses are averaged
            over observations for each minibatch, or summed, depending on
            size_average. When reduce is ``False``, returns a loss per batch
            element instead and ignores size_average. Default: ``True``

    Shape:
        - input: :math:`(N, *)` where `*` means, any number of additional
          dimensions
        - target: :math:`(N, *)`, same shape as the input
        - output: scalar. If `reduce` is ``True``, then :math:`(N, *)`,
            same shape as the input

    """
    def __init__(self, categorical_dim, latent_dim, size_average=False, reduce=True):
        super(KLDivLossGumbel, self).__init__(size_average)
        self.categorical_dim = categorical_dim
        self.latent_dim = latent_dim
        self.size_average=size_average
        self.reduce=reduce

    def forward(self, logits, log_logits):
        kl_tmp = logits*(log_logits-math.log(1.0/self.latent_dim))
        kl_tmp = torch.sum(torch.sum(kl_tmp, 2),1)
        if self.size_average:
            kl_tmp=(1.0/(self.latent_dim*self.categorical_dim))*kl_tmp
        if self.reduce:
            result=torch.sum(kl_tmp, 0)
            result=(1.0/float(logits.size(0)))*result
        else:
            result=kl_tmp
        return result




def sample_gumbel(shape, eps=1e-10, out=None):
    """
    Sample from Gumbel(0, 1)
    based on
    https://github.com/ericjang/gumbel-softmax/blob/3c8584924603869e90ca74ac20a6a03d99a91ef9/Categorical%20VAE.ipynb ,
    (MIT license)
    """
    U = out.resize_(shape).uniform_() if out is not None else torch.rand(shape)
    return - torch.log(eps - torch.log(U + eps))


def gumbel_softmax_sample(logits, tau=1, eps=1e-10):
    """
    Draw a sample from the Gumbel-Softmax distribution
    based on
    https://github.com/ericjang/gumbel-softmax/blob/3c8584924603869e90ca74ac20a6a03d99a91ef9/Categorical%20VAE.ipynb
    (MIT license)
    """
    dims = len(logits.size())
    gumbel_noise = sample_gumbel(logits.size(), eps=eps, out=logits.data.new())
    y = logits + Variable(gumbel_noise)
    return F.softmax(y / tau, dims - 1)


def gumbel_softmax(logits, tau=1, hard=False, eps=1e-10):
    """
    Sample from the Gumbel-Softmax distribution and optionally discretize.
    Args:
      logits: [batch_size, n_class] unnormalized log-probs
      tau: non-negative scalar temperature
      hard: if True, take argmax, but differentiate w.r.t. soft sample y
    Returns:
      [batch_size, n_class] sample from the Gumbel-Softmax distribution.
      If hard=True, then the returned sample will be one-hot, otherwise it will
      be a probability distribution that sums to 1 across classes
    Constraints:
    - this implementation only works on batch_size x num_features tensor for now
    based on
    https://github.com/ericjang/gumbel-softmax/blob/3c8584924603869e90ca74ac20a6a03d99a91ef9/Categorical%20VAE.ipynb ,
    (MIT license)
    """
    shape = logits.size()
    assert len(shape) == 2
    y_soft = gumbel_softmax_sample(logits, tau=tau, eps=eps)
    if hard:
        _, k = y_soft.data.max(-1)
        # this bit is based on
        # https://discuss.pytorch.org/t/stop-gradients-for-st-gumbel-softmax/530/5
        y_hard = logits.data.new(*shape).zero_().scatter_(-1, k.view(-1, 1), 1.0)
        # this cool bit of code achieves two things:
        # - makes the output value exactly one-hot (since we add then
        #   subtract y_soft value)
        # - makes the gradient equal to y_soft gradient (since we strip
        #   all other gradients)
        y = Variable(y_hard - y_soft.data) + y_soft
    else:
        y = y_soft
    return y


def hard_binarization(input, categorical_dim, latent_dim):
    input = input.view(-1, categorical_dim, latent_dim)
    first_max_values, first_max_ids = torch.max(input, dim=2,  keepdim=True)
    mask = Variable(torch.zeros(input.size()).cuda().scatter_(2, first_max_ids.data, 1), requires_grad=False)
    return mask
