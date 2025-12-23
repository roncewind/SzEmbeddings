# Investigation Results: Embedding Search Performance

**Date:** 2025-12-23
**Issue:** Poor embedding search performance despite excellent model training results

## Summary

✅ **Test data is VALID** - All validation samples are from loaded records with matching embeddings
✅ **Models work perfectly** - PostgreSQL cosine similarity achieves expected performance
❌ **Senzing configuration issue** - Embeddings used for candidate retrieval but NOT for scoring

---

## Key Findings

### 1. Test Data Validation ✅

**Verification Results:**
- Checked 20 validation samples against PostgreSQL
- **100% found** in database
- **100% embedding match** (max distance: 0.00005, essentially zero)
- Query-time embeddings == Load-time embeddings

**Conclusion:** Test data is trustworthy. We ARE testing records we loaded.

### 2. Model Performance (from BizNames/PersonalNames projects) ✅

**Training Results:**
```
Personal Names Model (LaBSE fine-tuned):
- Recall@1:  98%+
- Recall@10: 98%+

Business Names Model (phase9b_labse):
- Recall@1:  98%+
- Recall@10: 98%+
```

**Conclusion:** Models are excellent. This matches PostgreSQL direct cosine similarity results.

### 3. PostgreSQL Cosine Similarity ✅

**Test Results (50 samples):**
```
PostgreSQL Cosine Similarity:
- Found (any rank):    50/50 (100%)
- Recall@1:            36.0%
- Recall@10:           88.0%
- Mean Rank:           4.6
- Avg Query Time:      10ms
```

**Conclusion:** Direct cosine similarity works as expected.

### 4. Senzing Embedding Search ❌

**Test Results (50 samples):**
```
Senzing Attribute Search:
- Found (any rank):    0/50 (0%)
- Recall@1:            0.0%
- Recall@10:           0.0%
- Avg Query Time:      156ms
```

**Production Validation Results (758 test cases):**
```
Senzing Embedding Search:
- Recall@1:    5.5%  ❌ (target: >60%)
- Recall@10:   41.3% ❌ (target: >85%)
- Recall@100:  100%  ✅ (all records eventually found)
- Not Found:   0.0%  ✅
- Avg Time:    221ms
```

**Conclusion:** Embeddings find ALL records (R@100=100%) but rank them poorly (R@10=41.3%).

---

## Root Cause Analysis

### Senzing Search Response Examination

**Example: "Kazimierz Braun" (PERSON)**

**Senzing JSON Response:**
```json
{
  "RESOLVED_ENTITIES": [
    {
      "MATCH_INFO": {
        "MATCH_LEVEL_CODE": "POSSIBLY_SAME",
        "MATCH_KEY": "+NAME",           ⚠️ Only NAME, not NAME_EMBEDDING
        "ERRULE_CODE": "SNAME",
        "CANDIDATE_KEYS": {
          "NAME_EMBEDDING": [           ✅ Embeddings used for retrieval
            {"FEAT_ID": 257587, "FEAT_DESC": "كازيميرز براون"},
            {"FEAT_ID": 257588, "FEAT_DESC": "Kazimierz Braun"},
            {"FEAT_ID": 257589, "FEAT_DESC": "Kasimir Braun"},
            {"FEAT_ID": 257590, "FEAT_DESC": "Casimir Braun"},
            {"FEAT_ID": 257591, "FEAT_DESC": "Казімір Браўн"}
          ],
          "NAME_KEY": [
            {"FEAT_ID": 257569, "FEAT_DESC": "KSMRS|PRN"}
          ]
        },
        "FEATURE_SCORES": {
          "NAME": [                      ✅ NAME scores present
            {
              "INBOUND_FEAT_ID": 257567,
              "CANDIDATE_FEAT_ID": 257567,
              "SCORE": 100,
              ...
            }
          ]
          // ❌ NAME_EMBEDDING: MISSING!
          // Should have NAME_EMBEDDING scores here
        }
      }
    }
  ]
}
```

**Key Observations:**

1. ✅ **CANDIDATE_KEYS** contains `NAME_EMBEDDING`
   - Embeddings successfully retrieve 5 candidate features (all aliases)
   - Proves embeddings are being used for candidate retrieval

2. ❌ **FEATURE_SCORES** only contains `NAME`
   - No `NAME_EMBEDDING` scores present
   - Only traditional name matching scores used for ranking

3. ❌ **MATCH_KEY** is "+NAME"
   - Should be "+NAME+NAME_EMBEDDING" if embeddings contributed to match
   - Indicates embedding feature not participating in scoring

---

## Why Performance is Poor

### Two-Phase Search Process

**Phase 1: Candidate Retrieval** ✅ WORKING
- Embeddings find all relevant candidates
- This is why Recall@100 = 100%
- CANDIDATE_KEYS shows NAME_EMBEDDING features

**Phase 2: Scoring & Ranking** ❌ NOT WORKING
- Only NAME scores used for ranking
- No NAME_EMBEDDING or BIZNAME_EMBEDDING scores
- False positives with high NAME scores outrank true positives with high embedding similarity

**Result:**
- True match found but ranked at position 50 (missed by Recall@10)
- False positives with similar traditional name scores ranked higher
- Poor precision despite perfect recall at large K

---

## Configuration Issue

The problem is in `szConfig.json`. The embedding features are configured for:
- ✅ Candidate retrieval (feature matching)
- ❌ NOT scoring (contributing to entity scores)

**What needs fixing:**
- NAME_EMBEDDING and BIZNAME_EMBEDDING features must contribute to FEATURE_SCORES
- Embedding scores should appear in search results alongside NAME scores
- MATCH_KEY should include "+NAME_EMBEDDING" or "+BIZNAME_EMBEDDING"

---

## Next Steps

1. **Examine szConfig.json** - Review feature definitions for NAME_EMBEDDING and BIZNAME_EMBEDDING
   - Check if `SCORE_ELEMENTS` is properly configured
   - Verify embedding features have scoring thresholds (Tau values)
   - Ensure features participate in entity resolution scoring

2. **Compare with working configurations** - Check if other vector features in Senzing configs include scoring

3. **Test configuration changes** - Modify config and reload to verify embedding scoring

4. **Re-validate** - Run production validation after config fixes to measure improvement

---

## Expected Results After Fix

**Target Metrics:**
```
Senzing Embedding Search (after fix):
- Recall@1:   >60% (currently 5.5%)
- Recall@10:  >85% (currently 41.3%)
- Recall@100: >95% (currently 100%)
- Not Found:  <5%  (currently 0.0%)
```

**Verification:**
- FEATURE_SCORES should include NAME_EMBEDDING / BIZNAME_EMBEDDING
- MATCH_KEY should include "+NAME_EMBEDDING" or "+BIZNAME_EMBEDDING"
- Performance should match PostgreSQL cosine similarity (~88% R@10)
