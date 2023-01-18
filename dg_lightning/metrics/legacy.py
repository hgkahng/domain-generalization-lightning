
import torch


@torch.no_grad()
def entropy(a: torch.LongTensor) -> torch.FloatTensor:
    """H(A)"""
    _, counts = torch.unique(a, return_counts=True, dim=0)
    probs = counts / len(a)
    return (-probs * torch.log(probs)).sum()


@torch.no_grad()
def joint_entropy(a: torch.LongTensor, b: torch.LongTensor) -> torch.FloatTensor:
    """H(A,B)"""
    a, b = a.view(-1, 1), b.view(-1, 1)
    return entropy(torch.hstack([a, b]))


@torch.no_grad()
def conditional_entropy(a: torch.LongTensor, b: torch.LongTensor) -> torch.FloatTensor:
    """H(A|B) = H(A,B) - H(B)"""
    return joint_entropy(a, b) - entropy(b)


@torch.no_grad()
def mutual_info(a: torch.LongTensor, b: torch.LongTensor) -> torch.FloatTensor:
    """I(A;B) = H(A) - H(A|B)"""
    return entropy(a) - conditional_entropy(a, b)


@torch.no_grad()
def conditional_mutual_info(a: torch.LongTensor, b: torch.LongTensor, c: torch.LongTensor) -> torch.FloatTensor:
    """I(A;B|C)=\sum_c\{I(A;B|C=c)\}"""
    if c.ndim != 1:
        raise IndexError
    uniques, counts = torch.unique(c, return_counts=True, dim=0)
    probs = counts / len(c)
    cmi = 0.
    for value, p in zip(uniques, probs):
        cond = c.eq(value)
        cmi += p * mutual_info(a[cond], b[cond]) 
    return cmi



