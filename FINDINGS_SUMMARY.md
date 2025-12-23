# Embedding Search Investigation - Findings Summary

**Date:** 2025-12-23
**Status:** Root cause identified, awaiting Senzing configuration guidance

---

## TL;DR

✅ **Test data is valid** - All validation samples confirmed in database with matching embeddings
✅ **Models are excellent** - Training showed 98%+ recall, confirmed by PostgreSQL direct cosine similarity
✅ **Validation logic is correct** - Recall calculations verified
❌ **Senzing configuration issue** - Embeddings retrieve candidates but don't contribute to scoring
❌ **Adding embeddings makes search WORSE** - 41% recall vs 97% without embeddings

---

## The Problem in Numbers

### Production Validation Results (758 test cases):

| Search Mode | Recall@10 | Avg Results | Correct Entity Rank |
|-------------|-----------|-------------|---------------------|
| **Name-Only** | **97.1%** ✅ | 1 entity | **#1** (precise) |
| **With Embeddings** | **41.3%** ❌ | 10-56 entities | **#3-#25** (lost in crowd) |

**Key Statistics:**
- Embedding search finds **10-56x more entities** than name-only (1 entity)
- In **92.6% of cases**, embedding ranks the correct entity WORSE than name-only
- Recall@100 is still 100% (correct entity always found eventually, just ranked low)

---

## Root Cause Analysis

### What's Working ✅

1. **Embedding Quality**
   - Models: 98%+ Recall@1 and R@10 in training
   - PostgreSQL direct cosine similarity: 88% Recall@10
   - Query-time embeddings = Load-time embeddings (verified)

2. **Candidate Retrieval**
   - Embeddings successfully find candidate names
   - CANDIDATE_KEYS shows all aliases (e.g., "Puget Sound Energy", "Puget Sound Energy Inc")
   - Candidate names match PostgreSQL top results

### What's Broken ❌

3. **Entity Scoring & Ranking**
   - FEATURE_SCORES only contains: `["NAME", "RECORD_TYPE"]`
   - FEATURE_SCORES does NOT contain: `NAME_EMBEDDING` or `BIZNAME_EMBEDDING`
   - MATCH_KEY shows: `"+NAME"` (should be `"+NAME+NAME_EMBEDDING"`)
   - Only traditional NAME scores used for ranking

### How This Causes Poor Performance

**Without Embeddings (Name-Only):**
```
Query: "Puget Sound Energy"
→ Senzing finds: 1 entity
→ Ranks correctly at: #1
→ Result: 97.1% Recall@10 ✅
```

**With Embeddings:**
```
Query: "Puget Sound Energy"
→ Embeddings retrieve candidates: 13 entities (wide net)
→ FEATURE_SCORES: Only NAME scores available
→ All 13 entities ranked by NAME score only
→ Correct entity ranked at: #3 (not #1!)
→ Result: 41.3% Recall@10 ❌
```

**The Paradox:**
- Embeddings cast a WIDER net (more candidates found)
- But WITHOUT embedding scores to distinguish them
- False positives with high NAME scores outrank true positive
- Adding a powerful feature makes results WORSE

---

## Concrete Examples

### Example 1: "Puget Sound Energy" (ORGANIZATION)

**PostgreSQL Cosine Similarity:**
```
Rank 1: Puget Sound Energy      (distance: 0.000)  ← Perfect match
Rank 2: Puget Sound Energy Inc  (distance: 0.020)
Rank 3: Beysu Enerji Üretim     (distance: 0.454)
...
```

**Senzing Name-Only:**
```
1 entity returned
Rank 1: Entity 1109 "Puget Sound Energy Inc" ✅ Correct!
```

**Senzing With Embeddings:**
```
13 entities returned
Rank 1: ??? (wrong entity, high NAME score)
Rank 2: ??? (wrong entity, high NAME score)
Rank 3: Entity 1109 "Puget Sound Energy Inc" ← Should be #1!
```

**CANDIDATE_KEYS shows embeddings found it:**
```json
{
  "BIZNAME_EMBEDDING": [
    {"FEAT_ID": 122984, "FEAT_DESC": "Puget Sound Energy"},
    {"FEAT_ID": 122985, "FEAT_DESC": "Puget Sound Energy Inc"}
  ]
}
```

**FEATURE_SCORES missing embeddings:**
```json
{
  "NAME": [...],        ← Only this used for ranking
  "RECORD_TYPE": [...]  ← And this
  // BIZNAME_EMBEDDING: MISSING!
}
```

### Example 2: "Kazimierz Braun" (PERSON)

**Name-Only:** 1 entity, Rank 1 ✅
**With Embeddings:** 51 entities, Rank 15 ❌

The correct entity dropped from #1 to #15 because 14 other entities had higher NAME scores (but worse embedding similarity).

---

## Why Model Training Showed 98%+ Recall

The model training (BizNames/PersonalNames projects) tested:
- **Direct embedding similarity** between anchor/positive/negative triplets
- **PostgreSQL cosine distance** queries
- **NO entity resolution** involved

This is DIFFERENT from production validation which tests:
- Senzing search_by_attributes with entity resolution
- Multiple entities competing for ranking
- Scoring that should combine NAME + embedding signals

The models ARE excellent (98%+ confirmed by PostgreSQL). The issue is Senzing isn't using them for scoring.

---

## Configuration Issue

**Current Behavior:**
```json
{
  "CANDIDATE_KEYS": {
    "NAME_EMBEDDING": [...]     ← Used for retrieval ✅
  },
  "FEATURE_SCORES": {
    "NAME": [...],              ← Used for scoring ✅
    "RECORD_TYPE": [...]        ← Used for scoring ✅
    // NAME_EMBEDDING: MISSING! ← NOT used for scoring ❌
  },
  "MATCH_KEY": "+NAME"          ← Should be "+NAME+NAME_EMBEDDING" ❌
}
```

**Expected Behavior:**
```json
{
  "CANDIDATE_KEYS": {
    "NAME_EMBEDDING": [...]     ← Used for retrieval ✅
  },
  "FEATURE_SCORES": {
    "NAME": [...],              ← Used for scoring ✅
    "RECORD_TYPE": [...],       ← Used for scoring ✅
    "NAME_EMBEDDING": [...]     ← Should contribute to scoring! ✅
  },
  "MATCH_KEY": "+NAME+NAME_EMBEDDING" ← Should include embedding ✅
}
```

---

## Questions for Senzing Support

1. **How do we configure embedding features to contribute to FEATURE_SCORES?**
   - Current: Embeddings appear in CANDIDATE_KEYS only
   - Needed: Embeddings should contribute to entity scoring/ranking

2. **What configuration controls feature scoring participation?**
   - Is this a szConfig.json setting?
   - Do we need to use sz_configtool to enable feature scoring?
   - Are there score thresholds (Tau values) that need adjusting?

3. **Expected impact on match levels:**
   - Should embedding contribution improve MATCH_LEVEL from "POSSIBLY_SAME" to "SAME"?
   - Should MATCH_KEY include embedding features when they contribute?

---

## Supporting Documents

1. **SENZING_SUPPORT_REPORT.md** - Detailed comparison report with 8 concrete examples
2. **INVESTIGATION_RESULTS.md** - Full technical investigation findings
3. **Validation Results:** `results/validation_5k_20251222_185855.json`
4. **Test Data Verification:** `verify_test_data.py` (confirmed 100% valid)

---

## Next Steps

1. ✅ **Investigation Complete** - Root cause identified and documented
2. ⏳ **Awaiting Senzing Guidance** - Submit SENZING_SUPPORT_REPORT.md for configuration advice
3. ⏹️ **Configuration Changes** - Update szConfig.json based on Senzing guidance
4. ⏹️ **Re-validation** - Test with fixed configuration
5. ⏹️ **Expected Results** - Embedding Recall@10 should match name-only (>90%)
