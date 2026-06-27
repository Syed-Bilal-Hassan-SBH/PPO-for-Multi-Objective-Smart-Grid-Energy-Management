"""
Statistical Analysis Utilities for Publication-Ready Results

Provides comprehensive statistical testing and analysis tools:
- T-tests (parametric)
- Wilcoxon rank-sum test (non-parametric)
- Mann-Whitney U test
- Confidence intervals (bootstrap and parametric)
- Effect size calculations (Cohen's d)
- Normality tests
- Multi-seed result aggregation

Author: Publication-Ready RL Framework
"""

import numpy as np
from scipy import stats
from typing import Dict, List, Tuple, Optional, Union
import json
from pathlib import Path
from dataclasses import dataclass, asdict


@dataclass
class StatisticalTestResult:
    """Results from a statistical significance test."""
    test_name: str
    statistic: float
    p_value: float
    significant: bool
    alpha: float
    effect_size: Optional[float] = None
    interpretation: Optional[str] = None


@dataclass
class DescriptiveStats:
    """Descriptive statistics for a dataset."""
    mean: float
    median: float
    std: float
    se: float  # Standard error
    min_val: float
    max_val: float
    q1: float  # 25th percentile
    q3: float  # 75th percentile
    n: int
    ci_lower: float  # 95% CI lower bound
    ci_upper: float  # 95% CI upper bound


def compute_descriptive_stats(data: np.ndarray, confidence: float = 0.95) -> DescriptiveStats:
    """
    Compute comprehensive descriptive statistics.
    
    Args:
        data: 1D array of values
        confidence: Confidence level for CI (default 0.95)
        
    Returns:
        DescriptiveStats object
    """
    data = np.asarray(data)
    n = len(data)
    mean = np.mean(data)
    std = np.std(data, ddof=1)
    se = std / np.sqrt(n)
    
    # Confidence interval (parametric)
    ci_delta = stats.t.ppf((1 + confidence) / 2, n - 1) * se
    ci_lower = mean - ci_delta
    ci_upper = mean + ci_delta
    
    return DescriptiveStats(
        mean=float(mean),
        median=float(np.median(data)),
        std=float(std),
        se=float(se),
        min_val=float(np.min(data)),
        max_val=float(np.max(data)),
        q1=float(np.percentile(data, 25)),
        q3=float(np.percentile(data, 75)),
        n=n,
        ci_lower=float(ci_lower),
        ci_upper=float(ci_upper)
    )


def compute_confidence_interval(
    data: np.ndarray,
    confidence: float = 0.95,
    method: str = 'parametric'
) -> Tuple[float, float]:
    """
    Compute confidence interval for data.
    
    Args:
        data: 1D array of values
        confidence: Confidence level (default 0.95)
        method: 'parametric' or 'bootstrap'
        
    Returns:
        (lower_bound, upper_bound)
    """
    data = np.asarray(data)
    
    if method == 'parametric':
        mean = np.mean(data)
        se = stats.sem(data)
        ci_delta = stats.t.ppf((1 + confidence) / 2, len(data) - 1) * se
        return (mean - ci_delta, mean + ci_delta)
    
    elif method == 'bootstrap':
        # Bootstrap confidence interval
        n_bootstrap = 10000
        bootstrap_means = []
        for _ in range(n_bootstrap):
            sample = np.random.choice(data, size=len(data), replace=True)
            bootstrap_means.append(np.mean(sample))
        
        alpha = 1 - confidence
        lower = np.percentile(bootstrap_means, alpha/2 * 100)
        upper = np.percentile(bootstrap_means, (1 - alpha/2) * 100)
        return (lower, upper)
    
    else:
        raise ValueError(f"Unknown method: {method}")


def test_normality(data: np.ndarray, alpha: float = 0.05) -> Tuple[bool, float]:
    """
    Test if data is normally distributed using Shapiro-Wilk test.
    
    Args:
        data: 1D array of values
        alpha: Significance level
        
    Returns:
        (is_normal, p_value)
    """
    if len(data) < 3:
        # Shapiro-Wilk requires at least 3 samples
        return True, 1.0
    
    statistic, p_value = stats.shapiro(data)
    is_normal = p_value > alpha
    return is_normal, p_value


def compute_cohens_d(group1: np.ndarray, group2: np.ndarray) -> float:
    """
    Compute Cohen's d effect size.
    
    Args:
        group1: First group data
        group2: Second group data
        
    Returns:
        Cohen's d (positive means group1 > group2)
    """
    group1 = np.asarray(group1)
    group2 = np.asarray(group2)
    
    n1, n2 = len(group1), len(group2)
    var1, var2 = np.var(group1, ddof=1), np.var(group2, ddof=1)
    
    # Pooled standard deviation
    pooled_std = np.sqrt(((n1 - 1) * var1 + (n2 - 1) * var2) / (n1 + n2 - 2))
    
    # Cohen's d
    d = (np.mean(group1) - np.mean(group2)) / pooled_std
    return float(d)


def interpret_cohens_d(d: float) -> str:
    """
    Interpret Cohen's d effect size.
    
    Args:
        d: Cohen's d value
        
    Returns:
        Interpretation string
    """
    abs_d = abs(d)
    if abs_d < 0.2:
        return "negligible"
    elif abs_d < 0.5:
        return "small"
    elif abs_d < 0.8:
        return "medium"
    else:
        return "large"


def compare_methods(
    method1_scores: np.ndarray,
    method2_scores: np.ndarray,
    alpha: float = 0.05,
    test: str = 'auto'
) -> StatisticalTestResult:
    """
    Compare two methods using appropriate statistical test.
    
    Args:
        method1_scores: Scores from method 1
        method2_scores: Scores from method 2
        alpha: Significance level
        test: 'auto', 'ttest', 'wilcoxon', or 'mannwhitneyu'
        
    Returns:
        StatisticalTestResult object
    """
    method1_scores = np.asarray(method1_scores)
    method2_scores = np.asarray(method2_scores)
    
    # Compute effect size
    effect_size = compute_cohens_d(method1_scores, method2_scores)
    
    # Choose test automatically if requested
    if test == 'auto':
        # Check normality
        normal1, _ = test_normality(method1_scores, alpha)
        normal2, _ = test_normality(method2_scores, alpha)
        
        if normal1 and normal2 and len(method1_scores) >= 5 and len(method2_scores) >= 5:
            test = 'ttest'
        else:
            test = 'wilcoxon'
    
    # Perform test
    if test == 'ttest':
        # Independent samples t-test
        statistic, p_value = stats.ttest_ind(method1_scores, method2_scores)
        test_name = "Independent t-test"
        
    elif test == 'wilcoxon':
        # Wilcoxon rank-sum test (Mann-Whitney U)
        statistic, p_value = stats.ranksums(method1_scores, method2_scores)
        test_name = "Wilcoxon rank-sum test"
        
    elif test == 'mannwhitneyu':
        # Mann-Whitney U test
        statistic, p_value = stats.mannwhitneyu(
            method1_scores, method2_scores, alternative='two-sided'
        )
        test_name = "Mann-Whitney U test"
        
    else:
        raise ValueError(f"Unknown test: {test}")
    
    # Determine significance
    significant = p_value < alpha
    
    # Create interpretation
    if significant:
        if np.mean(method1_scores) > np.mean(method2_scores):
            interpretation = f"Method 1 significantly outperforms Method 2 (p={p_value:.4f}, d={effect_size:.2f})"
        else:
            interpretation = f"Method 2 significantly outperforms Method 1 (p={p_value:.4f}, d={effect_size:.2f})"
    else:
        interpretation = f"No significant difference (p={p_value:.4f}, d={effect_size:.2f})"
    
    return StatisticalTestResult(
        test_name=test_name,
        statistic=float(statistic),
        p_value=float(p_value),
        significant=significant,
        alpha=alpha,
        effect_size=effect_size,
        interpretation=interpretation
    )


def aggregate_multi_seed_results(
    seed_results: Dict[int, Dict[str, float]],
    metrics: Optional[List[str]] = None
) -> Dict[str, DescriptiveStats]:
    """
    Aggregate results across multiple random seeds.
    
    Args:
        seed_results: Dictionary mapping seed -> metrics dict
        metrics: List of metric names to aggregate (None = all)
        
    Returns:
        Dictionary mapping metric name -> DescriptiveStats
    """
    # Collect all metrics
    if metrics is None:
        metrics = list(next(iter(seed_results.values())).keys())
    
    aggregated = {}
    
    for metric in metrics:
        # Collect values across seeds
        values = [seed_results[seed][metric] for seed in seed_results.keys()]
        values = np.array(values)
        
        # Compute statistics
        aggregated[metric] = compute_descriptive_stats(values)
    
    return aggregated


def generate_significance_marker(p_value: float) -> str:
    """
    Generate significance marker for p-value.
    
    Args:
        p_value: P-value from statistical test
        
    Returns:
        Marker string: '***' (p<0.001), '**' (p<0.01), '*' (p<0.05), 'ns' (not significant)
    """
    if p_value < 0.001:
        return "***"
    elif p_value < 0.01:
        return "**"
    elif p_value < 0.05:
        return "*"
    else:
        return "ns"


def generate_statistical_report(
    results_dict: Dict[str, np.ndarray],
    baseline_name: str,
    output_file: Optional[Path] = None
) -> str:
    """
    Generate comprehensive statistical report comparing methods.
    
    Args:
        results_dict: Dictionary mapping method name -> array of scores
        baseline_name: Name of baseline method to compare against
        output_file: Optional path to save report
        
    Returns:
        Markdown-formatted report string
    """
    report_lines = ["# Statistical Analysis Report\n"]
    
    # Descriptive statistics for each method
    report_lines.append("## Descriptive Statistics\n")
    report_lines.append("| Method | Mean | Median | Std | SE | 95% CI | N |")
    report_lines.append("|--------|------|--------|-----|----|----|---|")
    
    stats_dict = {}
    for method_name, scores in results_dict.items():
        stats_obj = compute_descriptive_stats(scores)
        stats_dict[method_name] = stats_obj
        
        ci_str = f"[{stats_obj.ci_lower:.2f}, {stats_obj.ci_upper:.2f}]"
        report_lines.append(
            f"| {method_name} | {stats_obj.mean:.2f} | {stats_obj.median:.2f} | "
            f"{stats_obj.std:.2f} | {stats_obj.se:.2f} | {ci_str} | {stats_obj.n} |"
        )
    
    # Pairwise comparisons with baseline
    if baseline_name in results_dict:
        report_lines.append(f"\n## Comparisons with {baseline_name}\n")
        report_lines.append("| Method | Mean Diff | Cohen's d | Effect | Test | p-value | Sig | Result |")
        report_lines.append("|--------|-----------|-----------|--------|------|---------|-----|--------|")
        
        baseline_scores = results_dict[baseline_name]
        
        for method_name, scores in results_dict.items():
            if method_name == baseline_name:
                continue
            
            # Test
            test_result = compare_methods(scores, baseline_scores)
            
            # Mean difference
            mean_diff = np.mean(scores) - np.mean(baseline_scores)
            
            # Effect interpretation
            effect = interpret_cohens_d(test_result.effect_size)
            
            # Significance marker
            sig_marker = generate_significance_marker(test_result.p_value)
            
            # Win/Tie/Loss
            if test_result.significant:
                result = "Win ✅" if mean_diff > 0 else "Loss ❌"
            else:
                result = "Tie ➖"
            
            report_lines.append(
                f"| {method_name} | {mean_diff:+.2f} | {test_result.effect_size:.2f} | "
                f"{effect} | {test_result.test_name} | {test_result.p_value:.4f} | "
                f"{sig_marker} | {result} |"
            )
    
    # Summary
    report_lines.append("\n## Legend")
    report_lines.append("- **Sig**: *** (p<0.001), ** (p<0.01), * (p<0.05), ns (not significant)")
    report_lines.append("- **Effect**: Cohen's d interpretation (small: 0.2-0.5, medium: 0.5-0.8, large: >0.8)")
    report_lines.append("- **Result**: Win (significantly better), Loss (significantly worse), Tie (no significant difference)")
    
    report = "\n".join(report_lines)
    
    # Save if requested
    if output_file is not None:
        output_file = Path(output_file)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        with open(output_file, 'w') as f:
            f.write(report)
        print(f"Statistical report saved to: {output_file}")
    
    return report


def save_statistics_json(
    stats_dict: Dict[str, DescriptiveStats],
    output_file: Path
) -> None:
    """
    Save descriptive statistics to JSON file.
    
    Args:
        stats_dict: Dictionary mapping metric -> DescriptiveStats
        output_file: Output JSON file path
    """
    output_file = Path(output_file)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    
    # Convert to dict
    json_dict = {
        metric: asdict(stats) for metric, stats in stats_dict.items()
    }
    
    with open(output_file, 'w') as f:
        json.dump(json_dict, f, indent=2)
    
    print(f"Statistics saved to: {output_file}")


if __name__ == "__main__":
    # Example usage
    print("Statistical Utilities - Example Usage\n")
    
    # Generate example data
    np.random.seed(42)
    method1_scores = np.random.normal(100, 15, 5)  # Mean=100, std=15, n=5
    method2_scores = np.random.normal(85, 12, 5)   # Mean=85, std=12, n=5
    
    print("Example: Comparing two methods")
    print(f"Method 1 scores: {method1_scores}")
    print(f"Method 2 scores: {method2_scores}\n")
    
    # Descriptive stats
    stats1 = compute_descriptive_stats(method1_scores)
    print(f"Method 1 stats: Mean={stats1.mean:.2f} ± {stats1.se:.2f}, "
          f"95% CI=[{stats1.ci_lower:.2f}, {stats1.ci_upper:.2f}]")
    
    stats2 = compute_descriptive_stats(method2_scores)
    print(f"Method 2 stats: Mean={stats2.mean:.2f} ± {stats2.se:.2f}, "
          f"95% CI=[{stats2.ci_lower:.2f}, {stats2.ci_upper:.2f}]\n")
    
    # Statistical test
    result = compare_methods(method1_scores, method2_scores)
    print(f"Statistical test: {result.test_name}")
    print(f"p-value: {result.p_value:.4f}")
    print(f"Significant: {result.significant}")
    print(f"Cohen's d: {result.effect_size:.2f} ({interpret_cohens_d(result.effect_size)})")
    print(f"Interpretation: {result.interpretation}\n")
    
    # Generate report
    results_dict = {
        "PPO": method1_scores,
        "Random": method2_scores
    }
    report = generate_statistical_report(results_dict, baseline_name="Random")
    print(report)
