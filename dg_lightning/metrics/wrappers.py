
import typing
import torch


@torch.no_grad()
def macro_average_metric_wrapper(metric_fn: callable,
                                 y_pred: torch.FloatTensor,
                                 y_true: torch.Tensor,
                                 group: torch.LongTensor = None,
                                 ignore_group_index: typing.Union[int, typing.Iterable[int]] = [-1],
                                 **kwargs):
    """Wrapper function for evaluating the macro-averaged metric."""

    # if group is not provided, use the true target value as groups
    if (group is None) and (y_true.dtype != torch.long):
        raise NotImplementedError
    group = y_true if group is None else group; assert (group.dtype == torch.long)

    # find unique groups, and exclude groups in `ignore_group_index`
    unique_groups = torch.unique(group, sorted=True)
    unique_groups = torch.tensor([g for g in unique_groups if g not in ignore_group_index])
    if len(unique_groups) < 1:
        raise ValueError(f"No groups to evaluate when ignoring groups = {ignore_group_index}")

    # buffer
    groupwise_metric_values = torch.zeros_like(unique_groups)

    # compute metric group-wise. also ignore groups whose
    # group indicator is in `ignore_group_index`
    for i, g in enumerate(unique_groups):
        _mask = group.eq(g)
        _metric_value = metric_fn(y_pred[_mask], y_true[_mask], **kwargs)
        groupwise_metric_values[i] = _metric_value
    
    return groupwise_metric_values.mean()


@torch.no_grad()
def worst_group_metric_wrapper(metric_fn: callable,
                               y_pred: torch.FloatTensor,
                               y_true: torch.Tensor,
                               group: torch.LongTensor = None,
                               return_group: bool = False,
                               ignore_group_index: typing.Union[int, typing.Iterable[int]] = [-1],
                               **kwargs):
    """Wrapper function for evaluating the worst group metric."""
    
    # if group is not provided, use the true target value as groups
    if (group is None) and (y_true.dtype != torch.long):
        raise NotImplementedError
    group = y_true if group is None else group; assert (group.dtype == torch.long)
    
    # Find unique groups, and exclude groups in `ignore_group_index`
    unique_groups = torch.unique(group, sorted=True)
    unique_groups = torch.tensor([g for g in unique_groups if g not in ignore_group_index])
    if len(unique_groups) < 1:
        raise ValueError(f"No groups to evaluate when ignoring groups = {ignore_group_index}")

    # buffer
    groupwise_metric_values = torch.zeros_like(unique_groups)
    worst_is_highest: bool = metric_fn.__name__ in ['mean_squared_error', 'loss', ]   # TODO: add more
    if worst_is_highest:
        groupwise_metric_values.fill_(float('inf'))
   
    # compute metric group-wise. also ignore
    # groups whose group indicator is in `ignore_group_index`
    for i, g in enumerate(unique_groups):
        _mask = group.eq(g)
        _metric_value = metric_fn(y_pred[_mask], y_true[_mask], **kwargs)
        groupwise_metric_values[i] = _metric_value

    if worst_is_highest:
        worst_idx = torch.argmax(groupwise_metric_values)
    else:
        worst_idx = torch.argmin(groupwise_metric_values)

    if return_group:
        return groupwise_metric_values[worst_idx], unique_groups[worst_idx]
    else:
        return groupwise_metric_values[worst_idx]
    