# Performance Fix: Page Navigation Delay

## Problem
When clicking between pages in the Streamlit dashboard, there was a significant delay (1-2 seconds) before the new page would load. This was particularly noticeable when navigating between the Engineering pages (Today, People, Delivery, Code, Planning) and the Business page.

## Root Cause
The issue was caused by **eager loading** of the `pages.business` module at the top of `app.py`:

```python
# OLD CODE (lines 228-352)
from pages.business import (
    ORDER_BOOK_DAYS,
    ORDER_BOOK_TTL_SECONDS,
    _order_book,
    # ... 120+ more imports
)
```

The `pages/business.py` module is **4,372 lines** long and has extensive imports and module-level code. Every time ANY page was loaded (including Engineering pages), Python had to:
1. Import and parse the entire 4,372-line business.py module
2. Execute all its top-level code
3. Import all of business.py's dependencies (pandas, plotly, various API clients, etc.)

This happened on **every page navigation** because Streamlit reruns the entire app.py script on each page change.

## Solution
Implemented **lazy loading** of the `pages.business` module using a lazy import pattern:

```python
# NEW CODE (lines 228-246)
def _lazy_import_business():
    """Lazy import of pages.business module to improve performance."""
    from pages import business
    return business

def _business_readable() -> bool:
    """Check if business page is accessible (lazy wrapper)."""
    business = _lazy_import_business()
    return business._business_readable()

def _render_business() -> None:
    """Render the business page (lazy wrapper)."""
    business = _lazy_import_business()
    return business._render_business()
```

The `pages.business` module is now only imported when:
1. The Business page is actually being rendered
2. The "Refresh data" button is clicked on the Business page

## Performance Impact
- **Before**: ~1.5-2.0 seconds delay when navigating between pages
- **After**: Instant navigation between Engineering pages (no business.py import)
- **Business page**: First load imports the module (one-time cost), subsequent navigations use cached import

## Testing
All existing tests pass, including:
- `tests/test_pages.py` - Validates page navigation structure
- Performance verified with custom test showing:
  - `pages.business` is NOT imported when app.py loads
  - `pages.business` IS imported on-demand when accessed

## Files Modified
- `app.py`: 
  - Removed 125 lines of eager imports (lines 228-352)
  - Added 19 lines of lazy loading wrappers (lines 228-246)
  - Updated `_clear_page_caches()` to lazy-load business caches (lines 6462-6543)

## Backward Compatibility
✅ All existing functionality preserved
✅ All tests pass
✅ No changes to external API or user-facing behavior
