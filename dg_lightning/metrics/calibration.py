
import torch


@torch.no_grad()
def _compute_calibration_bins(preds, targets, num_bins: int = 10):
    """
    Compute bins used to calculate {ece, mce} metrics.
    Arguments:
        preds: 1d `torch.FloatTensor`, with values \in [0, 1]
        targets: 1d `torch.LongTensor` with values \in {0, 1}
        num_bins: int, number of bins to calculate
    Returns:
        (bins, binned, accs, confs, sizes) where
            bins: 1d `torch.FloatTensor` of bins of shape (num_bins, )
            binned: 1d `torch.LongTensor` of bin indicators
            accs: 1d `torch.FloatTensor` of shape (num_bins, )
            confs: 1d `torch.FloatTensor` of shape (num_bins, )
            sizes: 1d `torch.FloatTensor` of shape (num_bins, )
    """
    
    # assign each prediction to a bin; `binned` carries bin indices
    end: float = 1.0
    start: float = end / num_bins
    bins = torch.linspace(start, end, num_bins, device=preds.device)
    binned = torch.bucketize(preds, bins, right=False)  # same shape as `preds`

    # save {accuracy, confidence size} for each bin
    sizes = torch.zeros(num_bins, dtype=torch.float32, device=preds.device)
    accs, confs = torch.zeros_like(sizes), torch.zeros_like(sizes)

    for i in range(num_bins):  # 0, 1, ..., `num_bins-1`
        mask: torch.BoolTensor = binned.eq(i)
        sizes[i] = len(preds[mask])
        if sizes[i] > 0:
            accs[i] = targets[mask].sum().div(sizes[i])  # sum of vector with values in {0, 1}
            confs[i] = preds[mask].sum().div(sizes[i])   # sum of vector with values in [0, 1]

    return bins, binned, accs, confs, sizes


@torch.no_grad()
def expected_calibration_error(preds: torch.FloatTensor, targets: torch.LongTensor, num_bins: int = 10) -> float:
    """Measures expected calibration error."""
    bins, binned, bin_accs, bin_confs, bin_sizes = _compute_calibration_bins(preds, targets, num_bins)
    ece: float = 0.
    for i in range(num_bins):
        abs_conf_diff = abs(bin_accs[i] - bin_confs[i])
        ece += (bin_sizes[i] / bin_sizes.sum()) * abs_conf_diff
    return ece


@torch.no_grad()
def maximum_calibration_error(preds: torch.FloatTensor, targets: torch.LongTensor, num_bins: int = 10) -> float:
    """Measures maximum calibration error."""
    bins, binned, bin_accs, bin_confs, bin_sizes = _compute_calibration_bins(preds, targets, num_bins)
    mce: float = 0.
    for i in range(num_bins):
        abs_conf_diff = abs(bin_accs[i] - bin_confs[i])
        mce = max(mce, abs_conf_diff)
    return mce