# Investigator Filtering Logic

## Overview

The system now filters out **low-activity investigators** from both the dashboard display and embedding generation. This focuses the dashboard on actively publishing researchers.

## Filter Criteria

An investigator is **excluded** if they meet BOTH conditions:
1. **5-year total funding > $1,000,000** AND
2. **Papers in last 12 months ≤ 2**

### Rationale

This filter targets investigators who:
- Have substantial funding (indicating they were active historically)
- But have very low recent publication output (0-2 papers in the last year)

These investigators may be:
- In administrative roles
- Retired but still listed on grants
- On sabbatical or medical leave
- Transitioning careers

By excluding them, we focus the dashboard on currently active researchers.

## Where Filtering is Applied

### 1. Main Dashboard (`ddash2.py:812-819`)

```python
# Filter out low-activity investigators (>$1M funding but ≤2 current year papers)
all_investigators = db.query(InvestigatorStat).all()
inv_list = [
    inv for inv in all_investigators
    if not (inv.funding_past_5y > 1000000 and inv.n_p2 <= 2)
]
```

**Effect**: These investigators do not appear in the main table.

### 2. Smart Search (`ddash2.py:1032-1036`)

```python
# Filter out low-activity investigators (>$1M funding but ≤2 current year papers)
investigators = [
    inv for inv in all_investigators
    if not (inv.funding_past_5y > 1000000 and inv.n_p2 <= 2)
]
```

**Effect**: These investigators are excluded from smart search results.

### 3. Embedding Generation (`generate_embeddings.py:291-297`)

```python
for inv_id, funding_5y, n_p2 in result:
    # Skip if funding > $1M but only 0-2 papers in last 12 months
    if funding_5y > 1000000 and n_p2 <= 2:
        logger.info(f"Skipping {inv_id}: High funding (${funding_5y:,.0f}) but low recent output ({n_p2} papers)")
        skipped += 1
        continue
    inv_ids.append(inv_id)
```

**Effect**: Embeddings are not generated for these investigators (saves API costs).

## Example Scenarios

### Excluded Investigators

| Investigator | 5y Funding | Papers (12m) | Excluded? | Reason |
|-------------|-----------|--------------|-----------|--------|
| Dr. Smith   | $2.5M     | 1           | ✅ Yes    | High funding, low output |
| Dr. Jones   | $3.0M     | 0           | ✅ Yes    | High funding, no papers |
| Dr. Brown   | $1.5M     | 2           | ✅ Yes    | High funding, minimal output |

### Included Investigators

| Investigator | 5y Funding | Papers (12m) | Excluded? | Reason |
|-------------|-----------|--------------|-----------|--------|
| Dr. Taylor  | $2.0M     | 5           | ❌ No     | High funding, active |
| Dr. Wilson  | $800k     | 1           | ❌ No     | Moderate funding |
| Dr. Davis   | $1.2M     | 3           | ❌ No     | High funding, active |
| Dr. Miller  | $0        | 10          | ❌ No     | No funding, but active |

## Monitoring Filtered Investigators

### View Filtered Count

When running `generate_embeddings.py`, you'll see:

```
2025-01-28 10:15:30 [INFO] Skipping SMITH_JOHN: High funding ($2,500,000) but low recent output (1 papers)
2025-01-28 10:15:30 [INFO] Skipping JONES_MARY: High funding ($3,000,000) but low recent output (0 papers)
...
2025-01-28 10:20:45 [INFO] Found 387 investigators to process (45 skipped)
```

### Database Query

To see who's being filtered:

```sql
SELECT id, name, funding_past_5y, n_p2
FROM investigator_stats_v3
WHERE funding_past_5y > 1000000 AND n_p2 <= 2
ORDER BY funding_past_5y DESC;
```

## Adjusting Filter Thresholds

To modify the filtering criteria, update these values:

### Change Funding Threshold

Default: `1000000` ($1M)

**To change to $2M:**
1. In `ddash2.py`, find both instances: `inv.funding_past_5y > 1000000`
2. Change to: `inv.funding_past_5y > 2000000`
3. In `generate_embeddings.py`, find: `if funding_5y > 1000000`
4. Change to: `if funding_5y > 2000000`

### Change Paper Threshold

Default: `2` papers

**To change to 1 paper:**
1. In `ddash2.py`, find both instances: `inv.n_p2 <= 2`
2. Change to: `inv.n_p2 <= 1`
3. In `generate_embeddings.py`, find: `if funding_5y > 1000000 and n_p2 <= 2`
4. Change to: `if funding_5y > 1000000 and n_p2 <= 1`

### Disable Filtering Entirely

To disable filtering (show all investigators):

1. In `ddash2.py:812-819`, change to:
   ```python
   inv_list = db.query(InvestigatorStat).order_by(InvestigatorStat.funding_current.desc()).all()
   ```

2. In `ddash2.py:1032-1036`, change to:
   ```python
   investigators = all_investigators
   ```

3. In `generate_embeddings.py:291-297`, comment out the filtering logic:
   ```python
   # for inv_id, funding_5y, n_p2 in result:
   #     if funding_5y > 1000000 and n_p2 <= 2:
   #         ...
   inv_ids = [row[0] for row in result]
   ```

## Impact Analysis

Typical expected impact:
- **~10-15%** of investigators filtered out
- Reduces dashboard clutter
- Saves embedding API costs
- Focuses user attention on active researchers

## Future Enhancements

Potential improvements:
1. **Admin override**: Show filtered investigators in a separate "Inactive" tab
2. **Time-based**: Auto-unfilter if investigator publishes again
3. **Configurable**: Make thresholds editable via environment variables
4. **Notification**: Alert when previously active investigator becomes filtered
