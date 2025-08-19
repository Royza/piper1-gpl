# Version 1.0 Rollback Instructions

## Quick Rollback Commands

### If using Git (recommended):
```bash
# See commit history
git log --oneline

# Rollback to version 1.0 (before v1.1 improvements)
git reset --hard <commit_hash_before_v1.1>

# Or create a rollback branch
git checkout -b rollback-to-v1.0 <commit_hash_before_v1.1>
```

### Manual Rollback (if needed):

#### Core Files to Restore:
1. `src/piper/train/vits/dataset.py` - DataLoader configurations
2. `training_web_app.py` - Web server and system monitoring
3. `templates/index.html` - Web interface
4. `src/piper/train/vits/lightning.py` - Training logic

#### Version 1.0 Key Characteristics:
- DataLoader: Basic configuration without persistent_workers
- Progress tracking: Regex-based log parsing
- Cache: No size limits or cleanup
- GPU memory: Basic monitoring only
- TensorBoard: Not integrated

#### Configuration Rollback:
```python
# DataLoader settings (v1.0)
DataLoader(
    dataset,
    batch_size=self.batch_size,
    num_workers=self.num_workers,
    collate_fn=collate_fn
    # No persistent_workers, pin_memory, or prefetch_factor
)

# System monitoring (v1.0)
detect_system_capabilities()  # Basic GPU/CPU detection
calculate_optimal_settings()  # Simple batch size/workers calculation
```

#### Features to Disable for v1.0:
- TensorBoard logging
- Advanced cache management
- Learning rate finder
- Enhanced progress tracking
- Automatic GPU memory cleanup

## Verification Steps:
1. Check web interface loads correctly
2. Verify training starts without errors
3. Confirm system monitoring works
4. Test file upload and validation
5. Validate export functionality

## Emergency Contacts:
- Check GitHub issues for known problems
- Refer to original documentation in docs/
- Test with small dataset first before full rollback

---
*Created: January 2025 - Version 1.0 Baseline*
