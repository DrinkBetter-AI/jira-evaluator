#!/usr/bin/env python3
"""Quick test to verify PR section error handling works correctly."""

import pandas as pd
import numpy as np

# Simulate the _number_or function
def _number_or(value, default=None):
    """Return value if it's a valid number, else default."""
    try:
        if value is None or (isinstance(value, float) and np.isnan(value)):
            return default
        return value
    except (TypeError, ValueError):
        return default

# Simulate the fixed _open_pr_signals function
def _open_pr_signals(open_prs, open_count_exact):
    """Test version of the PR signals function."""
    TODAY_NO_REVIEWER_DAYS = 2.0
    
    try:
        total = _number_or(open_count_exact, float("nan"))
        if open_prs is None or (isinstance(open_prs, pd.DataFrame) and open_prs.empty):
            return {
                "total": int(total) if total == total else 0,
                "unapproved": 0,
                "never_reviewed": 0,
                "oldest_unreviewed_days": None,
                "no_reviewer_asked": 0,
            }

        live = open_prs.copy()
        if "is_draft" in live.columns:
            live = live[~live["is_draft"].fillna(False).astype(bool)]
        
        # After filtering drafts, check if we have any PRs left
        if live.empty:
            return {
                "total": int(total) if total == total else 0,
                "unapproved": 0,
                "never_reviewed": 0,
                "oldest_unreviewed_days": None,
                "no_reviewer_asked": 0,
            }

        approvals = live.get("approving_reviews", pd.Series(0, index=live.index))
        reviews = live.get("total_reviews", pd.Series(0, index=live.index))
        requests = live.get("review_requests", pd.Series(0, index=live.index))
        age = live.get("age_days", pd.Series(0.0, index=live.index)).fillna(0.0)

        unapproved = pd.Series(approvals, index=live.index).fillna(0).astype(int) == 0
        never = pd.Series(reviews, index=live.index).fillna(0).astype(int) == 0
        unasked = never & (pd.Series(requests, index=live.index).fillna(0).astype(int) == 0)

        oldest = age[never].max() if bool(never.any()) else None
        # Handle NaN values in oldest
        if oldest is not None and pd.isna(oldest):
            oldest = None
            
        return {
            "total": int(total) if total == total else int(len(live)),
            "unapproved": int(unapproved.sum()),
            "never_reviewed": int(never.sum()),
            "oldest_unreviewed_days": None if oldest is None else float(oldest),
            "no_reviewer_asked": int((unasked & (age > TODAY_NO_REVIEWER_DAYS)).sum()),
        }
    except Exception as e:
        print(f"Error in _open_pr_signals: {str(e)}")
        # Return safe defaults on any error
        return {
            "total": 0,
            "unapproved": 0,
            "never_reviewed": 0,
            "oldest_unreviewed_days": None,
            "no_reviewer_asked": 0,
        }

# Test cases
def test_pr_signals():
    """Run test cases for PR signals function."""
    print("Testing PR signals function...\n")
    
    # Test 1: None input
    print("Test 1: None input")
    result = _open_pr_signals(None, None)
    print(f"Result: {result}")
    assert result["total"] == 0
    assert result["unapproved"] == 0
    print("✓ Passed\n")
    
    # Test 2: Empty DataFrame
    print("Test 2: Empty DataFrame")
    empty_df = pd.DataFrame()
    result = _open_pr_signals(empty_df, 0)
    print(f"Result: {result}")
    assert result["total"] == 0
    print("✓ Passed\n")
    
    # Test 3: DataFrame with all drafts
    print("Test 3: DataFrame with all drafts")
    draft_df = pd.DataFrame({
        "is_draft": [True, True],
        "approving_reviews": [0, 0],
        "total_reviews": [0, 0],
        "review_requests": [0, 0],
        "age_days": [5.0, 10.0]
    })
    result = _open_pr_signals(draft_df, 2)
    print(f"Result: {result}")
    assert result["total"] == 2
    assert result["unapproved"] == 0  # All filtered out
    print("✓ Passed\n")
    
    # Test 4: Normal DataFrame with PRs
    print("Test 4: Normal DataFrame with PRs")
    normal_df = pd.DataFrame({
        "is_draft": [False, False, False],
        "approving_reviews": [0, 1, 0],
        "total_reviews": [0, 2, 1],
        "review_requests": [0, 1, 1],
        "age_days": [5.0, 10.0, 3.0]
    })
    result = _open_pr_signals(normal_df, 3)
    print(f"Result: {result}")
    assert result["total"] == 3
    assert result["unapproved"] == 2  # 2 PRs without approving reviews
    assert result["never_reviewed"] == 1  # 1 PR with 0 total reviews
    print("✓ Passed\n")
    
    # Test 5: DataFrame with NaN values
    print("Test 5: DataFrame with NaN values")
    nan_df = pd.DataFrame({
        "is_draft": [False, False],
        "approving_reviews": [np.nan, 0],
        "total_reviews": [np.nan, 0],
        "review_requests": [np.nan, 0],
        "age_days": [np.nan, 5.0]
    })
    result = _open_pr_signals(nan_df, 2)
    print(f"Result: {result}")
    assert result["total"] == 2
    print("✓ Passed\n")
    
    print("All tests passed! ✓")

if __name__ == "__main__":
    test_pr_signals()
