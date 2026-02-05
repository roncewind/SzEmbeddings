# PostgreSQL Tuning for OpenSanctions Load

## Settings Changed for Bulk Loading

### Initial Attempt (REVERTED - caused 12x slowdown)

| Setting | Original Value | Optimized Value | Context | Requires Restart |
|---------|---------------|-----------------|---------|------------------|
| `synchronous_commit` | `on` | `OFF` | user | No |
| `work_mem` | `4 MB` | `256MB` | user | No |
| `shared_buffers` | `12 GB` | `12 GB` | postmaster | Yes (unchanged) |
| `effective_cache_size` | `4 GB` | `32GB` | user | No |
| `max_wal_size` | `1 GB` | `4GB` | sighup | No (reload) |
| `checkpoint_timeout` | `5 min` (300s) | `30min` | sighup | No (reload) |
| `autovacuum` | `on` | `OFF` | sighup | No (reload) |
| `maintenance_work_mem` | `64 MB` | `1GB` | user | No |

**Result**: All settings reverted after causing severe performance degradation.

### Warning Suppression (ACTIVE - testing for I/O bottleneck)

| Setting | Original Value | Current Value | Context | Requires Restart |
|---------|---------------|--------------|---------|------------------|
| `log_min_messages` | `warning` | `ERROR` | sighup | No (reload) |
| `client_min_messages` | `notice` | `ERROR` | user | No |

**Rationale**: Constant "SET LOCAL can only be used in transaction blocks" warnings from Senzing may be causing unbuffered log I/O bottleneck.

## How to Revert Settings

### Option 1: Revert Database-Level Settings
```sql
ALTER DATABASE senzing RESET synchronous_commit;
ALTER DATABASE senzing RESET work_mem;
ALTER DATABASE senzing RESET effective_cache_size;
ALTER DATABASE senzing RESET maintenance_work_mem;
```

### Option 2: Edit postgresql.conf
Restore these lines to original values:
```
max_wal_size = 1GB
checkpoint_timeout = 5min
autovacuum = on
```

Then reload PostgreSQL:
```bash
sudo systemctl reload postgresql
```

### Option 3: Set Back to Original Values Explicitly
```sql
ALTER DATABASE senzing SET synchronous_commit = ON;
ALTER DATABASE senzing SET work_mem = '4MB';
ALTER DATABASE senzing SET effective_cache_size = '4GB';
ALTER DATABASE senzing SET maintenance_work_mem = '64MB';
```

And in postgresql.conf:
```
max_wal_size = 1GB
checkpoint_timeout = 300s
autovacuum = on
```

## Performance Impact

**Expected**: 1.5-2x speedup
**Actual**: 12x SLOWER (5.4 records/min vs 69 records/min)

### Unoptimized Performance (Baseline)
- Rate: 69 records/min (4,136/hour)
- 47,400 records in 11 hours
- 93.8% success rate

### Optimized Performance (Current)
- Rate: 5.4 records/min (325/hour)
- 1,950 records in 6 hours
- TBD success rate

## Code Changes

Added to `sz_load_embeddings.py`:
```python
from senzing import SzEngine, SzEngineFlags  # Added SzEngineFlags import
```

Note: Tried using `SZ_WITHOUT_INFO` flag but it doesn't exist in this version.
Default behavior (`SZ_NO_FLAGS`) already skips detailed info.

## Re-enable After Load Completes

**CRITICAL**: After load completes, restore normal logging and run maintenance:

```bash
sudo -u postgres psql -d senzing
```

```sql
-- Restore logging levels
ALTER DATABASE senzing RESET log_min_messages;    -- Back to 'warning'
ALTER DATABASE senzing RESET client_min_messages; -- Back to 'notice'

-- Run maintenance (autovacuum should already be 'on' after revert)
VACUUM ANALYZE;
```
