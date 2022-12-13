import numpy as np

from statsmodels.stats.inter_rater import fleiss_kappa, aggregate_raters
from tqdm import tqdm


NITER = 1000


def permutation_test_diff(x, y, quantity=np.mean, paired=False, niter=NITER, seed=88, show_pbar=False):
    """
    Permutation test to obtain a p-value for the difference in `quantity`
    between 2 quantities. Default quantity is mean.

    Randomly shuffle the 2 quantities into 2 groups and calculate the metric.
    If paired, randomly re-permute the indices (essentially, "unpair" them) and calculate the metric.

    Two-sided p-value: proportion of trials where |observed| > |random|.

    Arguments:
      x:  list, np.ndarray, or pd.Series
      y:  list, np.ndarray, or pd.Series
      quantity: python func, whose arguments are y_pred and y_true
                use functools.partial to incorporate additional arguments
      paired: whether observations are paired
      niter: number of trials to run
      seed: random state
    """
    num_x, num_y = len(x), len(y)
    x = np.asarray(x)
    y = np.asarray(y)
    np.random.seed(seed)
    if paired:
        assert num_x == num_y
        obs = quantity(x - y)
        indices = np.arange(num_x)
    else:
        obs = quantity(x) - quantity(y)
        combined = np.concatenate([x, y])
    trial_values = []
    pbar = tqdm(range(niter), total=niter) if show_pbar else range(niter)
    for _ in pbar:
        # If observations are paired, then you can't group them all together, shuffle, and re-distribute
        # Instead you have to re-permute the indices for each group
        if paired:
            indices1 = np.random.permutation(indices)
            x_perm = x[indices1]
            indices2 = np.random.permutation(indices1)
            y_perm = y[indices2]
            trial = quantity(x_perm - y_perm)
        else:
            np.random.shuffle(combined)
            x_perm = combined[:num_x]
            y_perm = combined[num_x:]
            assert len(x_perm) == num_x
            assert len(y_perm) == num_y
            trial = quantity(x_perm) - quantity(y_perm)
        trial_values.append(trial)
    return 1. - np.mean(np.abs(obs) > np.abs(trial_values))


def permutation_test_metric_diff(p1, p2, gt, metric, niter=NITER, seed=88, show_pbar=False):
    """
    Permutation test to obtain a p-value for the difference in metrics
    between 2 predictions. By definition, these are paired observations.

    Randomly switch predictions between the 2 predictors and calculate the metric.
    Two-sided p-value: proportion of trials where |observed| > |random|.

    Arguments:
      p1: list, np.ndarray, or pd.Series
      p2: list, np.ndarray, or pd.Series
      gt: list, np.ndarray, or pd.Series
      metric: python func, whose arguments are y_pred and y_true
              use functools.partial to incorporate additional arguments
      niter: number of trials to run
      seed: random state
    """
    assert len(p1) == len(p2) == len(gt)
    num_samples = len(p1)
    p1 = np.asarray(p1)
    p2 = np.asarray(p2)
    gt = np.asarray(gt)
    np.random.seed(seed)
    result1 = metric(y_pred=p1, y_true=gt)
    result2 = metric(y_pred=p2, y_true=gt)
    observed = result1 - result2
    trial_values = []
    pbar = tqdm(range(niter), total=niter) if show_pbar else range(niter)
    for _ in pbar:
        # With probability 0.5, sample indices to switch groups
        switch = np.random.binomial(1, 0.5, num_samples)
        rows_to_switch = [_ for _ in range(num_samples) if switch[_]]
        permuted1 = p1.copy()
        permuted2 = p2.copy()
        permuted1[rows_to_switch] = p2[rows_to_switch]
        permuted2[rows_to_switch] = p1[rows_to_switch]
        perm_result1 = metric(y_pred=permuted1, y_true=gt)
        perm_result2 = metric(y_pred=permuted2, y_true=gt)
        trial_values.append(perm_result1 - perm_result2)
    return 1. - np.mean(np.abs(observed) > np.abs(trial_values))


def permutation_test_metric_diff_unpaired(p1, p2, gt1, gt2, metric, niter=NITER, seed=88, show_pbar=False):
    """
    Permutation test to obtain a p-value for the difference in metrics
    between 2 predictions when the predictions are not paired.

    For example, performance on dataset X vs. Y.

    Randomly shuffle across the two groups and calculate the metric.
    Two-sided p-value: proportion of trials where |observed| > |random|.

    Arguments:
      p1: list, np.ndarray, or pd.Series
      p2: list, np.ndarray, or pd.Series
      gt: list, np.ndarray, or pd.Series
      metric: python func, whose arguments are y_pred and y_true
              use functools.partial to incorporate additional arguments
      niter: number of trials to run
      seed: random state
    """
    n, m = len(p1), len(p2)
    p1 = np.asarray(p1)
    p2 = np.asarray(p2)
    gt1 = np.asarray(gt1)
    gt2 = np.asarray(gt2)
    np.random.seed(seed)
    result1 = metric(y_pred=p1, y_true=gt1)
    result2 = metric(y_pred=p2, y_true=gt2)
    observed = result1 - result2
    trial_values = []
    pbar = tqdm(range(niter), total=niter) if show_pbar else range(niter)
    combined_p, combined_gt = np.concatenate([p1, p2]), np.concatenate([gt1, gt2])
    indices = np.arange(n+m)
    for _ in pbar:
        np.random.shuffle(indices)
        index1, index2 = indices[:n], indices[n:]
        perm_result1 = metric(y_pred=combined_p[index1], y_true=combined_gt[index1])
        perm_result2 = metric(y_pred=combined_p[index2], y_true=combined_gt[index2])
        trial_values.append(perm_result1 - perm_result2)
    return 1. - np.mean(np.abs(observed) > np.abs(trial_values))


def bootstrap_ci_single(x, quantity=np.mean, alpha=0.05, return_dist=False, niter=NITER, seed=88, show_pbar=False):
    """
    Obtain a bootstrapped (1-alpha)% two-tailed confidence interval for a given quantity of a sample x.

    :param x: sample of values
    :param quantity: quantity of interest, default is mean
    :param alpha: for confidence interval
    :param return_dist: return distribution of bootstrapped values
    :param niter: number of bootstrap iterations
    :param seed: random state
    :return: upper and lower bound of CI, +/- distribution of bootstrap values
    """
    x = np.asarray(x)
    num_x = len(x)
    np.random.seed(seed)
    bootstrapped = []
    pbar = tqdm(range(niter), total=niter) if show_pbar else range(niter)
    for _ in pbar:
        # Sample with replacement
        bootstrap_indices = np.random.choice(range(num_x), num_x, replace=True)
        x_boot = x[bootstrap_indices]
        bootstrapped.append(quantity(x_boot))
    alpha *= 100.
    lower_bound = np.percentile(bootstrapped, alpha / 2.)
    upper_bound = np.percentile(bootstrapped, 100. - alpha / 2.)
    if return_dist:
        return (quantity(x), lower_bound, upper_bound), bootstrapped
    else:
        return quantity(x), lower_bound, upper_bound


def bootstrap_ci_diff(x, y, quantity=np.mean, paired=False, alpha=0.05, return_dist=False, niter=NITER, seed=88,
                      show_pbar=False):
    """
    Obtain a bootstrapped (1-alpha)% two-tailed confidence interval for the difference in `quantity` between
    two samples x and y.

    :param x: sample of values
    :param y: sample of values
    :param quantity: quantity of interest, default is mean
    :param paired: whether observations are paired
    :param alpha: for confidence interval
    :param return_dist: return distribution of bootstrapped values
    :param niter: number of bootstrap iterations
    :param seed: random state
    :return: upper and lower bound of CI, +/- distribution of bootstrap values
    """
    x = np.asarray(x)
    y = np.asarray(y)
    num_x, num_y = len(x), len(y)
    if paired:
        assert num_x == num_y
        obs = quantity(x - y)
    else:
        obs = quantity(x) - quantity(y)
    np.random.seed(seed)
    bootstrapped = []
    pbar = tqdm(range(niter), total=niter) if show_pbar else range(niter)
    for _ in pbar:
        # Sample with replacement
        if paired:
            bootstrap_indices = np.random.choice(range(num_x), num_x, replace=True)
            x_boot = x[bootstrap_indices]
            y_boot = y[bootstrap_indices]
            bootstrapped.append(quantity(x_boot - y_boot))
        else:
            bootstrap_indices_x = np.random.choice(range(num_x), num_y, replace=True)
            bootstrap_indices_y = np.random.choice(range(num_y), num_y, replace=True)
            x_boot = x[bootstrap_indices_x]
            y_boot = y[bootstrap_indices_y]
            bootstrapped.append(quantity(x_boot) - quantity(y_boot))
    alpha *= 100.
    lower_bound = np.percentile(bootstrapped, alpha / 2.)
    upper_bound = np.percentile(bootstrapped, 100. - alpha / 2.)
    if return_dist:
        return (obs, lower_bound, upper_bound), bootstrapped
    else:
        return obs, lower_bound, upper_bound


def bootstrap_ci_single_metric(p, gt, metric, alpha=0.05, return_dist=False, niter=NITER, seed=88, show_pbar=False):
    """
    Bootstrap to obtain a (1-alpha)% (two-tailed) CI for a performance metric.

    Randomly sample with replacement from predictions/ground truth and obtain
    bootstrapped values for the difference in metrics.

    Note: produces results for metric(p1) - metric(p2)

    Then calculate percentiles of the resulting bootstrap distribution.

    Arguments:
      p:  list, np.ndarray, or pd.Series
      gt: list, np.ndarray, or pd.Series
      metric: python func, whose arguments are y_pred and y_true
              use functools.partial to incorporate additional arguments
      alpha: alpha value for confidence interval
      return_dist: if True, return also the bootstrap distribution
      niter: number of trials to run
      seed: random state
    """
    assert len(p) == len(gt)
    num_samples = len(p)
    p = np.asarray(p)
    gt = np.asarray(gt)
    np.random.seed(seed)
    bootstrapped = []
    obs = metric(y_pred=p, y_true=gt)
    pbar = tqdm(range(niter), total=niter) if show_pbar else range(niter)
    for _ in pbar:
        # Sample with replacement
        bootstrap_indices = np.random.choice(range(num_samples), num_samples, replace=True)
        bootstrap_pred = p[bootstrap_indices]
        bootstrap_true = gt[bootstrap_indices]
        bootstrapped.append(metric(y_pred=bootstrap_pred, y_true=bootstrap_true))
    alpha *= 100.
    lower_bound = np.percentile(bootstrapped, alpha / 2.)
    upper_bound = np.percentile(bootstrapped, 100. - alpha / 2.)
    if return_dist:
        return (obs, lower_bound, upper_bound), bootstrapped
    else:
        return obs, lower_bound, upper_bound


def bootstrap_ci_metric_diff(p1, p2, gt, metric, alpha=0.05, return_dist=False, niter=NITER, seed=88, show_pbar=False):
    """
    Bootstrap to obtain a (1-alpha)% (two-tailed) CI for the difference in metrics
    between 2 predictions.

    Randomly sample with replacement from predictions/ground truth and obtain
    bootstrapped values for the difference in metrics.

    Note: produces results for metric(p1) - metric(p2)

    Then calculate percentiles of the resulting bootstrap distribution.

    Arguments:
      p1:  list, np.ndarray, or pd.Series
      p2:  list, np.ndarray, or pd.Series
      gt: list, np.ndarray, or pd.Series
      metric: python func, whose arguments are y_pred and y_true
              use functools.partial to incorporate additional arguments
      alpha: alpha value for confidence interval
      return_dist: if True, return also the bootstrap distribution
      niter: number of trials to run
      seed: random state
    """
    assert len(p1) == len(p2) == len(gt)
    num_samples = len(p1)
    p1 = np.asarray(p1)
    p2 = np.asarray(p2)
    gt = np.asarray(gt)
    np.random.seed(seed)
    bootstrapped = []
    pbar = tqdm(range(niter), total=niter) if show_pbar else range(niter)
    obs = metric(y_pred=p1, y_true=gt) - metric(y_pred=p2, y_true=gt)
    for _ in pbar:
        # Sample with replacement
        bootstrap_indices = np.random.choice(range(num_samples), num_samples, replace=True)
        bootstrap_pred1 = p1[bootstrap_indices]
        bootstrap_pred2 = p2[bootstrap_indices]
        bootstrap_true  = gt[bootstrap_indices]
        boot_result1 = metric(y_pred=bootstrap_pred1, y_true=bootstrap_true)
        boot_result2 = metric(y_pred=bootstrap_pred2, y_true=bootstrap_true)
        bootstrapped.append(boot_result1 - boot_result2)
    alpha *= 100.
    lower_bound = np.percentile(bootstrapped, alpha / 2.)
    upper_bound = np.percentile(bootstrapped, 100. - alpha / 2.)
    if return_dist:
        return (obs, lower_bound, upper_bound), bootstrapped
    else:
        return obs, lower_bound, upper_bound


def permutation_test_compare_fleiss_kappa(group1, group2, num_cat, niter=NITER, seed=88, show_pbar=False):
    """
    This function performs a permutation test to compare intergrader agreement as measured by Fleiss' kappa
    between two GROUPS (e.g., N>2).

    :param group1: (N, num_graders)
    :param group2: (M, num_graders)
    :param num_cat: number of possible grades
    :param niter:
    :param seed:
    :param show_pbar:
    :return:
    """
    assert len(group1) == len(group2)
    # Reformat for fleiss_kappa function
    n, m = group1.shape[1], group2.shape[1]
    group1_reformat, _ = aggregate_raters(group1, n_cat=num_cat)
    group2_reformat, _ = aggregate_raters(group2, n_cat=num_cat)
    obs = fleiss_kappa(group1_reformat) - fleiss_kappa(group2_reformat)
    pbar = tqdm(range(niter), total=niter) if show_pbar else range(niter)
    combined = np.hstack([group1, group2])
    np.random.seed(seed)
    trial_values = []
    for _ in pbar:
       # For each row, randomly permute the predictions
        permuted_row_indices = [np.random.permutation(np.arange(n+m)) for _ in range(len(group1))]
        permuted = np.hstack([group1, group2]).copy()
        permuted = np.stack([row[index] for row, index in zip(permuted, permuted_row_indices)])
        g1_perm, _ = aggregate_raters(permuted[:, :n], n_cat=num_cat)
        g2_perm, _ = aggregate_raters(permuted[:, n:], n_cat=num_cat)
        trial = fleiss_kappa(g1_perm) - fleiss_kappa(g2_perm)
        trial_values.append(trial)
    return 1. - np.mean(np.abs(obs) > np.abs(trial_values))


def bootstrap_ci_compare_fleiss_kappa(group1, group2, num_cat, alpha=0.05, return_dist=False, niter=NITER, seed=88, show_pbar=False):
    """
    Bootstrap of the difference in intergrader agreement as measured by Fleiss' kappa
    between two GROUPS (e.g., N>2).
    """
    assert len(group1) == len(group2)
    # Reformat for fleiss_kappa function
    group1_reformat, _ = aggregate_raters(group1, n_cat=num_cat)
    group2_reformat, _ = aggregate_raters(group2, n_cat=num_cat)
    obs = fleiss_kappa(group1_reformat) - fleiss_kappa(group2_reformat)
    pbar = tqdm(range(niter), total=niter) if show_pbar else range(niter)
    np.random.seed(seed)
    bootstrapped = []
    for _ in pbar:
        # Randomly sample indices
        indices = np.random.choice(np.arange(len(group1)), len(group1), replace=True)
        g1_boot, _ = aggregate_raters(group1[indices], n_cat=num_cat)
        g2_boot, _ = aggregate_raters(group2[indices], n_cat=num_cat)
        bootstrapped.append(fleiss_kappa(g1_boot) - fleiss_kappa(g2_boot))
    alpha *= 100.
    lower_bound = np.percentile(bootstrapped, alpha / 2.)
    upper_bound = np.percentile(bootstrapped, 100. - alpha / 2.)
    if return_dist:
        return (obs, lower_bound, upper_bound), bootstrapped
    else:
        return obs, lower_bound, upper_bound


def bootstrap_ci_single_fleiss_kappa(group, num_cat, alpha=0.05, return_dist=False, niter=NITER, seed=88, show_pbar=False):
    """
    Bootstrapped (1-alpha)% confidence interval for Fleiss' kappa.
    """
    # Reformat for fleiss_kappa function
    group_reformat, _ = aggregate_raters(group, n_cat=num_cat)
    obs = fleiss_kappa(group_reformat)
    pbar = tqdm(range(niter), total=niter) if show_pbar else range(niter)
    np.random.seed(seed)
    bootstrapped = []
    for _ in pbar:
        # Randomly sample indices
        indices = np.random.choice(np.arange(len(group)), len(group), replace=True)
        g_boot, _ = aggregate_raters(group[indices], n_cat=num_cat)
        bootstrapped.append(fleiss_kappa(g_boot))
    alpha *= 100.
    lower_bound = np.percentile(bootstrapped, alpha / 2.)
    upper_bound = np.percentile(bootstrapped, 100. - alpha / 2.)
    if return_dist:
        return (obs, lower_bound, upper_bound), bootstrapped
    else:
        return obs, lower_bound, upper_bound


def permutation_grouped_test(p1, p2, gt, metric, niter=NITER, seed=88):
    '''
    Permutation test to obtain a p-value for the difference in metrics
    between 2 predictions, with multiple readers. Example use case: predictions
    of multiple experts with/without AI.

    Randomly switch predictions between the 2 predictors and calculate the metric.
    Two-sided p-value: proportion of trials where |observed| > |random|.

    Arguments:
      p1:  np.ndarray with shape (n, N) where n = number of samples
                    and N = number of ratings
      p2:  np.ndarray with shape (n, N) where n = number of samples
                    and N = number of ratings
      gt: list, np.ndarray, or pd.Series
      metric: python func, whose arguments are y_pred and y_true
              use functools.partial to incorporate additional arguments
      niter: number of trials to run
      seed: random state
    '''
    assert len(p1) == len(p2) == len(gt)
    assert p1.shape[-1] == p2.shape[-1], f'Group 1 has {p1.shape[-1]} raters whereas Group 2 has {p2.shape[-1]} raters\nPlease use `permutation_uneven_grouped_test` instead'
    num_samples = len(p1)
    p1 = np.asarray(p1)
    p2 = np.asarray(p2)
    gt = np.asarray(gt)
    np.random.seed(seed)
    result1 = np.mean([metric(y_pred=p1[:,i], y_true=gt) for i in range(p1.shape[-1])])
    result2 = np.mean([metric(y_pred=p2[:,i], y_true=gt) for i in range(p2.shape[-1])])
    observed = result1 - result2
    trial_values = []
    for i in tqdm(range(niter), total=niter):
        # With probability 0.5, sample indices to switch groups
        switch = np.random.binomial(1, 0.5, num_samples)
        rows_to_switch = [_ for _ in range(num_samples) if switch[_]]
        permuted1 = p1.copy()
        permuted2 = p2.copy()
        permuted1[rows_to_switch] = p2[rows_to_switch]
        permuted2[rows_to_switch] = p1[rows_to_switch]
        perm_result1 = np.mean([metric(y_pred=permuted1[:,i], y_true=gt) for i in range(permuted1.shape[-1])])
        perm_result2 = np.mean([metric(y_pred=permuted2[:,i], y_true=gt) for i in range(permuted2.shape[-1])])
        trial_values.append(perm_result1 - perm_result2)
    return 1. - np.mean(np.abs(observed) > np.abs(trial_values))


def permutation_uneven_grouped_test(p1, p2, gt, metric, niter=NITER, seed=88):
    '''
    This differs from `permutation_grouped_test` because it works if the
    number of raters in each group is different. The difference is that
    it shuffles all raters randomly and splits into 2 groups versus just switching
    the entire row.

    Permutation test to obtain a p-value for the difference in metrics
    between 2 predictions, with multiple readers. Example use case: predictions
    of multiple experts with/without AI.

    Randomly switch predictions between the 2 predictors and calculate the metric.
    Two-sided p-value: proportion of trials where |observed| > |random|.

    Arguments:
      p1:  np.ndarray with shape (n, N) where n = number of samples
                    and N = number of ratings
      p2:  np.ndarray with shape (n, N) where n = number of samples
                    and N = number of ratings
      gt: list, np.ndarray, or pd.Series
      metric: python func, whose arguments are y_pred and y_true
              use functools.partial to incorporate additional arguments
      niter: number of trials to run
      seed: random state
    '''
    assert len(p1) == len(p2) == len(gt)
    num_samples = len(p1)
    p1 = np.asarray(p1)
    p2 = np.asarray(p2)
    num_raters1 = p1.shape[1]
    num_raters2 = p2.shape[1]
    gt = np.asarray(gt)
    np.random.seed(seed)
    result1 = np.mean([metric(y_pred=p1[:,i], y_true=gt) for i in range(p1.shape[-1])])
    result2 = np.mean([metric(y_pred=p2[:,i], y_true=gt) for i in range(p2.shape[-1])])
    observed = result1 - result2
    trial_values = []
    # Join predictions
    concat_predictions = np.concatenate((p1, p2), axis=1)
    for i in tqdm(range(niter), total=niter):
        permuted = concat_predictions.copy()
        # Shuffle ratings across raters
        for row in range(permuted.shape[0]):
            permuted[row] = np.random.permutation(permuted[row])
        permuted1 = permuted[:,:num_raters1]
        permuted2 = permuted[num_raters1:,]
        perm_result1 = np.mean([metric(y_pred=permuted1[:,i], y_true=gt) for i in range(permuted1.shape[-1])])
        perm_result2 = np.mean([metric(y_pred=permuted2[:,i], y_true=gt) for i in range(permuted2.shape[-1])])
        trial_values.append(perm_result1 - perm_result2)
    return 1. - np.mean(np.abs(observed) > np.abs(trial_values))


def bootstrap_grouped_ci(p1, p2, gt, metric, alpha=0.05, return_dist=False, niter=NITER, seed=88):
    '''
    Bootstrap to obtain a (1-alpha/2)% (two-tailed) CI for the difference in metrics
    between 2 predictions, with multiple readers. Example use case: predictions
    of multiple experts with/without AI.

    Randomly sample with replacement from predictions/ground truth and obtain
    bootstrapped values for the difference in metrics.

    Note: produces results for metric(p1) - metric(p2)

    Then calculate percentiles of the resulting bootstrap distribution.

    Arguments:
      p1:  np.ndarray with shape (n, N) where n = number of samples
                    and N = number of ratings
      p2:  np.ndarray with shape (n, N) where n = number of samples
                    and N = number of ratings
      gt: list, np.ndarray, or pd.Series
      metric: python func, whose arguments are y_pred and y_true
              use functools.partial to incorporate additional arguments
      alpha: alpha value for confidence interval
      return_dist: if True, return also the bootstrap distribution
      niter: number of trials to run
      seed: random state
    '''
    assert len(p1) == len(p2) == len(gt)
    num_samples = len(p1)
    p1 = np.asarray(p1)
    p2 = np.asarray(p2)
    gt = np.asarray(gt)
    np.random.seed(seed)
    bootstrapped = []
    for i in tqdm(range(niter), total=niter):
        # Sample with replacement
        bootstrap_indices = np.random.choice(range(num_samples), num_samples, replace=True)
        bootstrap_pred1 = p1[bootstrap_indices]
        bootstrap_pred2 = p2[bootstrap_indices]
        bootstrap_true  = gt[bootstrap_indices]
        boot_result1 = np.mean([metric(y_pred=bootstrap_pred1[:,i], y_true=bootstrap_true) for i in range(bootstrap_pred1.shape[-1])])
        boot_result2 = np.mean([metric(y_pred=bootstrap_pred2[:,i], y_true=bootstrap_true) for i in range(bootstrap_pred2.shape[-1])])
        bootstrapped.append(boot_result1 - boot_result2)
    alpha *= 100.
    lower_bound = np.percentile(bootstrapped, alpha / 2.)
    upper_bound = np.percentile(bootstrapped, 100. - alpha / 2.)
    if return_dist:
        return (lower_bound, upper_bound), bootstrapped
    else:
        return (lower_bound, upper_bound)

