"""
Checkpoint Utilities for Intention Collapse Experiments
=========================================================
Robust checkpointing for long-running Colab experiments.

Features:
- JSONL incremental saving with flush
- Automatic resume from interruption
- Error handling per item (no abort on single failure)
- Performance metrics logging

Usage:
    from checkpoint_utils import ExperimentRunner
    
    runner = ExperimentRunner(
        run_id="mistral_gsm8k_baseline",
        output_dir="/content/drive/MyDrive/intention_collapse_v3"
    )
    
    for problem in runner.iterate(problems):
        result, acts = run_single_problem(...)
        runner.save_item(result, acts)
    
    runner.finalize()

Version: 1.0.0
"""

import json
import os
import time
import traceback
import numpy as np
from datetime import datetime
from typing import List, Dict, Any, Optional, Generator, Callable
from dataclasses import dataclass, asdict
from pathlib import Path


__version__ = "1.2.0"


# =============================================================================
# DATA STRUCTURES
# =============================================================================

@dataclass
class RunProgress:
    """Tracks progress of a run."""
    run_id: str
    started_at: str
    last_updated: str
    total_items: int
    completed_items: int
    successful_items: int
    failed_items: int
    elapsed_seconds: float
    items_per_minute: float
    estimated_remaining_minutes: float
    last_completed_idx: int
    failed_indices: List[int]
    
    def to_dict(self) -> Dict:
        return asdict(self)


@dataclass 
class ItemError:
    """Records an error for a single item."""
    idx: int
    error_type: str
    error_message: str
    traceback_snippet: str
    timestamp: str


# =============================================================================
# JSONL WRITER (with flush)
# =============================================================================

class JSONLWriter:
    """
    Incremental JSONL writer with immediate flush.
    
    Each write is immediately flushed to disk, ensuring no data loss
    even if the process crashes.
    
    Args:
        filepath: Path to the JSONL file
        fsync_every: If >0, call fsync every N writes (0=every write, which is safest but slowest)
    """
    
    def __init__(self, filepath: str, fsync_every: int = 25):
        self.filepath = filepath
        self.fsync_every = fsync_every
        self._file = None
        self._count = 0
    
    def open(self, mode: str = 'a'):
        """Open file for writing."""
        self._file = open(self.filepath, mode, encoding='utf-8')
        if mode == 'a' and os.path.exists(self.filepath):
            # Count existing lines for resume
            with open(self.filepath, 'r', encoding='utf-8') as f:
                self._count = sum(1 for _ in f)
    
    def write(self, obj: Dict):
        """Write object as JSON line and flush immediately."""
        if self._file is None:
            raise RuntimeError("File not opened. Call open() first.")
        
        line = json.dumps(obj, ensure_ascii=False)
        self._file.write(line + '\n')
        self._file.flush()  # Always flush to Python buffer
        
        # fsync to OS - configurable frequency for performance
        if self.fsync_every == 0 or (self._count % self.fsync_every == 0):
            os.fsync(self._file.fileno())
        
        self._count += 1
    
    def close(self):
        """Close the file."""
        if self._file:
            self._file.close()
            self._file = None
    
    @property
    def count(self) -> int:
        return self._count
    
    def __enter__(self):
        self.open()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()


def read_jsonl(filepath: str) -> List[Dict]:
    """Read all objects from a JSONL file."""
    if not os.path.exists(filepath):
        return []
    
    results = []
    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                results.append(json.loads(line))
    return results


def count_jsonl_lines(filepath: str) -> int:
    """Count lines in JSONL file without loading all into memory."""
    if not os.path.exists(filepath):
        return 0
    
    with open(filepath, 'r', encoding='utf-8') as f:
        return sum(1 for _ in f)


# =============================================================================
# EXPERIMENT RUNNER
# =============================================================================

class ExperimentRunner:
    """
    Robust experiment runner with checkpointing.
    
    Handles:
    - Incremental saving to JSONL
    - Progress tracking with progress.json
    - Automatic resume from interruption
    - Error handling per item
    - Performance metrics
    
    Example:
        runner = ExperimentRunner("mistral_gsm8k", output_dir)
        
        for problem in runner.iterate(problems, condition="baseline"):
            try:
                result, acts = run_single_problem(...)
                runner.save_item(result, acts)
            except Exception as e:
                runner.record_error(problem.idx, e)
        
        summary = runner.finalize()
    """
    
    def __init__(
        self,
        run_id: str,
        output_dir: str,
        condition: str = "",
        sync_interval: int = 10,  # Sync progress.json every N items
    ):
        """
        Initialize runner.
        
        Args:
            run_id: Unique identifier for this run (e.g., "mistral_gsm8k")
            output_dir: Directory to save outputs (usually Google Drive)
            condition: Current condition being run (baseline/enhanced/babble)
            sync_interval: How often to sync progress.json
        """
        self.run_id = run_id
        self.output_dir = Path(output_dir)
        self.condition = condition
        self.sync_interval = sync_interval
        
        # Create output directory
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # File paths
        self.results_path = self.output_dir / f"{run_id}_{condition}_results.jsonl"
        self.activations_path = self.output_dir / f"{run_id}_{condition}_activations.npz"
        self.progress_path = self.output_dir / f"{run_id}_{condition}_progress.json"
        self.errors_path = self.output_dir / f"{run_id}_{condition}_errors.jsonl"
        
        # State
        self._writer: Optional[JSONLWriter] = None
        self._error_writer: Optional[JSONLWriter] = None
        self._activations: List[np.ndarray] = []
        self._activation_idxs: List[int] = []  # Track which idx each activation belongs to
        self._start_time: float = 0
        self._total_items: int = 0
        self._current_idx: int = 0
        self._completed: int = 0
        self._successful: int = 0
        self._failed: int = 0
        self._failed_indices: List[int] = []
        self._resume_from: int = 0
    
    def _get_resume_point(self) -> int:
        """Determine where to resume from."""
        # Primary source: count JSONL lines
        jsonl_count = count_jsonl_lines(str(self.results_path))
        
        # Backup source: progress.json
        progress_count = 0
        if self.progress_path.exists():
            try:
                with open(self.progress_path, 'r') as f:
                    progress = json.load(f)
                    progress_count = progress.get('completed_items', 0)
            except:
                pass
        
        # Trust JSONL as source of truth, but warn if mismatch
        if progress_count > 0 and jsonl_count != progress_count:
            print(f"  ⚠️ Mismatch: JSONL has {jsonl_count} items, progress.json says {progress_count}")
            print(f"  → Trusting JSONL (source of truth)")
        
        return jsonl_count
    
    def _recompute_counts_from_results(self) -> tuple:
        """
        Recompute success/fail counts by streaming through results.jsonl.
        
        This is called on resume to get accurate counts including errors.
        Only counts valid JSON lines (ignores truncated/corrupted lines).
        
        Returns: (total, successful, failed, failed_indices)
        """
        if not self.results_path.exists():
            return 0, 0, 0, []
        
        total = 0
        failed = 0
        failed_indices = []
        
        with open(self.results_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                except json.JSONDecodeError:
                    # Ignore truncated/corrupted lines (e.g., from mid-write crash)
                    # This ensures resume_from only counts complete items
                    continue
                
                # Only count successfully parsed lines
                total += 1
                if obj.get('status') == 'error':
                    failed += 1
                    if 'idx' in obj:
                        failed_indices.append(obj['idx'])
        
        successful = total - failed
        return total, successful, failed, failed_indices
    
    def _load_existing_activations(self) -> tuple:
        """
        Load existing activations and their indices if resuming.
        
        Returns: (activations_list, idxs_list)
        """
        if not self.activations_path.exists():
            return [], []
        
        try:
            data = np.load(str(self.activations_path))
            activations = []
            idxs = []
            
            if 'activations' in data:
                acts_array = data['activations']
                activations = [acts_array[i] for i in range(acts_array.shape[0])]
            
            if 'idxs' in data:
                idxs = data['idxs'].tolist()
            else:
                # Legacy format without idxs - assume sequential
                idxs = list(range(len(activations)))
            
            return activations, idxs
            
        except Exception as e:
            print(f"  ⚠️ Could not load existing activations: {e}")
        
        return [], []
    
    def start(self, total_items: int) -> int:
        """
        Start or resume the run.
        
        Returns: index to start from (0 if fresh, >0 if resuming)
        """
        self._total_items = total_items
        self._start_time = time.time()
        
        # Check for resume - use valid line count as source of truth
        self._resume_from = self._get_resume_point()
        
        if self._resume_from > 0:
            print(f"\n🔄 RESUMING from item {self._resume_from}/{total_items}")
            
            # Recompute accurate counts from results.jsonl (only valid JSON lines)
            total, successful, failed, failed_indices = self._recompute_counts_from_results()
            
            # Use valid line count as the real resume point (handles truncated lines)
            self._resume_from = total
            self._completed = total
            self._successful = successful
            self._failed = failed
            self._failed_indices = failed_indices
            
            print(f"  Previous: {successful} successful, {failed} failed")
            
            # DON'T load activations to memory here - _save_activations() handles
            # loading from disk + appending new ones. Loading here would cause
            # duplicates since _save_activations() also loads existing.
            self._activations = []
            self._activation_idxs = []
            if self.activations_path.exists():
                # Count existing activations for info only
                try:
                    data = np.load(str(self.activations_path))
                    n_existing = len(data['activations']) if 'activations' in data else 0
                    print(f"  Activations file exists ({n_existing} items, will append on save)")
                except:
                    print(f"  Activations file exists (will append on save)")
        else:
            print(f"\n🚀 STARTING fresh run: {total_items} items")
        
        # Open writers in append mode
        self._writer = JSONLWriter(str(self.results_path))
        self._writer.open('a')
        
        self._error_writer = JSONLWriter(str(self.errors_path))
        self._error_writer.open('a')
        
        return self._resume_from
    
    def should_skip(self, idx: int) -> bool:
        """Check if this index should be skipped (already completed)."""
        return idx < self._resume_from
    
    def iterate(self, items: List[Any], condition: str = "") -> Generator:
        """
        Iterate over items with automatic resume.
        
        Usage:
            for problem in runner.iterate(problems, "baseline"):
                result, acts = process(problem)
                runner.save_item(result, acts)
        """
        if condition:
            self.condition = condition
            # Update paths for this condition
            self.results_path = self.output_dir / f"{self.run_id}_{condition}_results.jsonl"
            self.activations_path = self.output_dir / f"{self.run_id}_{condition}_activations.npz"
            self.progress_path = self.output_dir / f"{self.run_id}_{condition}_progress.json"
            self.errors_path = self.output_dir / f"{self.run_id}_{condition}_errors.jsonl"
        
        start_from = self.start(len(items))
        
        for idx, item in enumerate(items):
            self._current_idx = idx
            
            if idx < start_from:
                continue  # Skip already completed
            
            yield item
    
    def save_item(self, result: Any, activations: Optional[np.ndarray] = None):
        """
        Save a single result item.
        
        Args:
            result: ExperimentResult or dict with results
            activations: Optional numpy array of activations
        """
        # Convert to dict if needed
        if hasattr(result, 'to_dict'):
            result_dict = result.to_dict()
        elif hasattr(result, '__dict__'):
            result_dict = result.__dict__
        else:
            result_dict = result
        
        # Add metadata
        result_dict['_checkpoint_idx'] = self._current_idx
        result_dict['_checkpoint_time'] = datetime.now().isoformat()
        
        # Write to JSONL (immediately flushed)
        self._writer.write(result_dict)
        
        # Store activations with their index for alignment (Fix #3)
        if activations is not None:
            self._activations.append(activations)
            self._activation_idxs.append(self._current_idx)
        
        # Update counts
        self._completed += 1
        self._successful += 1
        
        # Sync progress periodically
        if self._completed % self.sync_interval == 0:
            self._sync_progress()
            self._save_activations()  # Also save activations periodically
    
    def record_error(self, idx: int, error: Exception):
        """Record an error for a single item (without aborting)."""
        error_record = ItemError(
            idx=idx,
            error_type=type(error).__name__,
            error_message=str(error),
            traceback_snippet=traceback.format_exc()[-500:],  # Last 500 chars
            timestamp=datetime.now().isoformat()
        )
        
        # Write to errors.jsonl for detailed tracking
        self._error_writer.write(asdict(error_record))
        
        # CRITICAL: Also write placeholder to results.jsonl
        # This ensures count_jsonl_lines() stays in sync with actual progress
        # Without this, resume would skip failed indices!
        placeholder = {
            "idx": idx,
            "status": "error",
            "error_type": error_record.error_type,
            "error_message": error_record.error_message,
            "_checkpoint_idx": self._current_idx,
            "_checkpoint_time": datetime.now().isoformat(),
        }
        self._writer.write(placeholder)
        
        self._failed += 1
        self._failed_indices.append(idx)
        self._completed += 1
        
        print(f"  ⚠️ Error on item {idx}: {type(error).__name__}: {str(error)[:100]}")
    
    def _sync_progress(self):
        """Sync progress to progress.json with atomic write."""
        elapsed = time.time() - self._start_time
        # Use only this session's items for throughput calculation (Fix B)
        session_items = self._completed - self._resume_from
        items_per_min = session_items / (elapsed / 60) if elapsed > 0 else 0
        remaining = self._total_items - self._completed
        eta_minutes = remaining / items_per_min if items_per_min > 0 else 0
        
        progress = RunProgress(
            run_id=self.run_id,
            started_at=datetime.fromtimestamp(self._start_time).isoformat(),
            last_updated=datetime.now().isoformat(),
            total_items=self._total_items,
            completed_items=self._completed,
            successful_items=self._successful,
            failed_items=self._failed,
            elapsed_seconds=elapsed,
            items_per_minute=items_per_min,
            estimated_remaining_minutes=eta_minutes,
            last_completed_idx=self._current_idx,
            failed_indices=self._failed_indices
        )
        
        # Atomic write: write to temp file then rename (Fix A)
        # This prevents corrupted progress.json if process dies mid-write
        tmp_path = self.progress_path.with_suffix('.tmp')
        with open(tmp_path, 'w') as f:
            json.dump(progress.to_dict(), f, indent=2)
        tmp_path.replace(self.progress_path)
    
    def _save_activations(self):
        """
        Save activations to npz file with their indices for alignment.
        
        Strategy: Load existing activations, append new ones, save all.
        This is safe but requires enough RAM to hold all activations.
        For 200 items × 5 layers × 4096 dim × 4 bytes ≈ 16MB, this is fine.
        
        For very large runs, consider chunked saving to separate files.
        """
        if not self._activations:
            return
        
        try:
            # Load existing activations and idxs if any
            existing_acts = []
            existing_idxs = []
            if self.activations_path.exists():
                try:
                    data = np.load(str(self.activations_path))
                    if 'activations' in data:
                        acts_array = data['activations']
                        existing_acts = [acts_array[i] for i in range(acts_array.shape[0])]
                    if 'idxs' in data:
                        existing_idxs = data['idxs'].tolist()
                    else:
                        existing_idxs = list(range(len(existing_acts)))
                except:
                    pass  # Start fresh if can't load
            
            # Combine existing + new
            all_activations = existing_acts + self._activations
            all_idxs = existing_idxs + self._activation_idxs
            
            if all_activations:
                acts_array = np.stack(all_activations, axis=0)
                idxs_array = np.array(all_idxs, dtype=np.int32)
                
                # Save both activations and their indices (Fix #3)
                np.savez_compressed(
                    str(self.activations_path),
                    activations=acts_array,
                    idxs=idxs_array
                )
                
                # Clear only NEW activations/idxs (keep memory reasonable)
                # Next save will reload from file + append new
                self._activations.clear()
                self._activation_idxs.clear()
                
        except Exception as e:
            print(f"  ⚠️ Could not save activations: {e}")
    
    def finalize(self) -> Dict[str, Any]:
        """
        Finalize the run and return summary.
        
        Call this after the loop completes.
        """
        # Final sync
        self._sync_progress()
        self._save_activations()
        
        # Close writers
        if self._writer:
            self._writer.close()
        if self._error_writer:
            self._error_writer.close()
        
        # Calculate final stats
        elapsed = time.time() - self._start_time
        
        # Session-only metrics (Fix B)
        session_items = self._completed - self._resume_from
        session_throughput = session_items / (elapsed / 60) if elapsed > 0 else 0
        
        summary = {
            'run_id': self.run_id,
            'condition': self.condition,
            'total_items': self._total_items,
            'completed': self._completed,
            'successful': self._successful,
            'failed': self._failed,
            'failed_indices': self._failed_indices,
            'elapsed_seconds': elapsed,
            'elapsed_human': f"{elapsed/60:.1f} minutes",
            # Report session throughput, not historical (Fix B)
            'session_items': session_items,
            'items_per_minute': session_throughput,
            'results_path': str(self.results_path),
            # Check file existence, not memory list (Fix #2)
            'activations_path': str(self.activations_path) if self.activations_path.exists() else None,
            'resumed_from': self._resume_from,
        }
        
        # Print summary
        print(f"\n{'='*60}")
        print(f"✅ {self.condition.upper()} COMPLETE")
        print(f"{'='*60}")
        print(f"  Items: {self._successful}/{self._total_items} successful")
        if self._failed > 0:
            print(f"  Failed: {self._failed} (indices: {self._failed_indices[:10]}{'...' if len(self._failed_indices) > 10 else ''})")
        if self._resume_from > 0:
            print(f"  This session: {session_items} items in {elapsed/60:.1f} min ({session_throughput:.1f}/min)")
        else:
            print(f"  Time: {elapsed/60:.1f} minutes ({session_throughput:.1f} items/min)")
        print(f"  Results: {self.results_path}")
        
        return summary


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================

def run_with_checkpoint(
    run_id: str,
    output_dir: str,
    problems: List[Any],
    conditions: List[str],
    process_fn: Callable,
    model: Any,
    tokenizer: Any,
    extractor: Any,
    config: Dict
) -> Dict[str, Any]:
    """
    High-level function to run all conditions with checkpointing.
    
    NOTE: This convenience function accumulates results in RAM for return value.
    For very large runs (>500 items), use ExperimentRunner directly and read
    results from JSONL files instead.
    
    Args:
        run_id: Base run ID (e.g., "mistral_gsm8k")
        output_dir: Output directory
        problems: List of problems to process
        conditions: List of conditions (e.g., ['baseline', 'enhanced', 'babble'])
        process_fn: Function that processes a single problem
                   Signature: process_fn(model, tokenizer, extractor, problem, condition, config) -> (result, activations)
        model: The loaded model
        tokenizer: The tokenizer
        extractor: Activation extractor
        config: Configuration dict with max_new_tokens etc.
    
    Returns:
        Dict with results per condition
    """
    all_results = {}
    all_activations = {}
    all_summaries = {}
    
    for condition in conditions:
        print(f"\n{'='*60}")
        print(f"CONDITION: {condition.upper()}")
        print(f"{'='*60}")
        
        runner = ExperimentRunner(run_id, output_dir, condition)
        
        results = []
        activations = []
        
        for problem in runner.iterate(problems, condition):
            try:
                result, acts = process_fn(
                    model, tokenizer, extractor, problem, condition, config
                )
                runner.save_item(result, acts)
                results.append(result)
                if acts is not None:
                    activations.append(acts)
                    
            except Exception as e:
                runner.record_error(problem.idx if hasattr(problem, 'idx') else 0, e)
        
        summary = runner.finalize()
        
        all_results[condition] = results
        all_activations[condition] = activations
        all_summaries[condition] = summary
    
    return {
        'results': all_results,
        'activations': all_activations,
        'summaries': all_summaries
    }


def get_run_status(output_dir: str, run_id: str) -> Dict[str, Any]:
    """
    Check status of a run across all conditions.
    
    Returns dict with completion status per condition.
    """
    output_dir = Path(output_dir)
    conditions = ['baseline', 'enhanced', 'babble']
    
    status = {'run_id': run_id, 'conditions': {}}
    
    for condition in conditions:
        results_path = output_dir / f"{run_id}_{condition}_results.jsonl"
        progress_path = output_dir / f"{run_id}_{condition}_progress.json"
        
        cond_status = {
            'completed_items': count_jsonl_lines(str(results_path)),
            'results_exist': results_path.exists(),
            'progress_exists': progress_path.exists(),
        }
        
        if progress_path.exists():
            try:
                with open(progress_path, 'r') as f:
                    progress = json.load(f)
                    cond_status['total_items'] = progress.get('total_items', 0)
                    cond_status['failed_items'] = progress.get('failed_items', 0)
                    cond_status['is_complete'] = (
                        cond_status['completed_items'] >= cond_status['total_items']
                    )
            except:
                cond_status['is_complete'] = False
        else:
            cond_status['is_complete'] = False
        
        status['conditions'][condition] = cond_status
    
    # Overall completion
    status['all_complete'] = all(
        c.get('is_complete', False) for c in status['conditions'].values()
    )
    
    return status


def print_run_status(output_dir: str, run_id: str):
    """Print formatted status of a run."""
    status = get_run_status(output_dir, run_id)
    
    print(f"\n📊 Status: {run_id}")
    print("-" * 40)
    
    for condition, cond_status in status['conditions'].items():
        completed = cond_status['completed_items']
        total = cond_status.get('total_items', '?')
        is_complete = cond_status.get('is_complete', False)
        
        icon = "✅" if is_complete else "🔄" if completed > 0 else "○"
        print(f"  {icon} {condition}: {completed}/{total}")
    
    print("-" * 40)
    if status['all_complete']:
        print("✅ All conditions complete!")
    else:
        print("🔄 Run in progress or incomplete")


# =============================================================================
# SANITY CHECK
# =============================================================================

def run_sanity_test():
    """Quick test of checkpoint functionality."""
    import tempfile
    
    print("Running checkpoint_utils sanity test v1.2...")
    
    with tempfile.TemporaryDirectory() as tmpdir:
        # Test JSONL writer
        writer = JSONLWriter(os.path.join(tmpdir, "test.jsonl"))
        writer.open('w')
        writer.write({"idx": 0, "value": "test"})
        writer.write({"idx": 1, "value": "test2"})
        writer.close()
        
        # Verify
        data = read_jsonl(os.path.join(tmpdir, "test.jsonl"))
        assert len(data) == 2, "JSONL write failed"
        assert data[0]['idx'] == 0, "JSONL content wrong"
        print("  ✓ JSONL writer works")
        
        # Test runner with simulated error
        class FakeProblem:
            def __init__(self, idx):
                self.idx = idx
        
        problems = [FakeProblem(i) for i in range(5)]
        
        runner = ExperimentRunner("test_run", tmpdir, "test_condition")
        
        # Create fake activations
        fake_acts = np.random.randn(3, 100).astype(np.float32)
        
        for problem in runner.iterate(problems):
            if problem.idx == 2:
                # Simulate error on idx=2
                runner.record_error(problem.idx, ValueError("Simulated error"))
            else:
                runner.save_item({'idx': problem.idx, 'result': 'ok'}, fake_acts)
        
        summary = runner.finalize()
        
        assert summary['successful'] == 4, f"Expected 4 successful, got {summary['successful']}"
        assert summary['failed'] == 1, f"Expected 1 failed, got {summary['failed']}"
        assert 2 in summary['failed_indices'], "Failed index 2 not recorded"
        print("  ✓ ExperimentRunner handles errors correctly")
        
        # CRITICAL: Verify results.jsonl has exactly 5 lines (including error placeholder)
        results_lines = count_jsonl_lines(os.path.join(tmpdir, "test_run_test_condition_results.jsonl"))
        assert results_lines == 5, f"BUG: results.jsonl has {results_lines} lines, expected 5"
        print("  ✓ Error placeholder written to results.jsonl")
        
        # Test resume after error - verify counts are accurate
        runner2 = ExperimentRunner("test_run", tmpdir, "test_condition")
        resume_point = runner2.start(5)
        assert resume_point == 5, f"Resume failed: got {resume_point}, expected 5"
        assert runner2._successful == 4, f"Resume successful count wrong: {runner2._successful}"
        assert runner2._failed == 1, f"Resume failed count wrong: {runner2._failed}"
        assert 2 in runner2._failed_indices, "Resume didn't recover failed indices"
        print("  ✓ Resume recovers accurate counts")
        
        # Verify error record exists
        errors = read_jsonl(os.path.join(tmpdir, "test_run_test_condition_errors.jsonl"))
        assert len(errors) == 1, f"Expected 1 error record, got {len(errors)}"
        assert errors[0]['idx'] == 2, "Error index wrong"
        print("  ✓ Error details recorded in errors.jsonl")
        
        # Verify activations have idxs
        acts_path = os.path.join(tmpdir, "test_run_test_condition_activations.npz")
        assert os.path.exists(acts_path), "Activations file not created"
        acts_data = np.load(acts_path)
        assert 'idxs' in acts_data, "Activations missing idxs"
        idxs = acts_data['idxs'].tolist()
        assert 2 not in idxs, "Error idx should not be in activations"
        assert set(idxs) == {0, 1, 3, 4}, f"Wrong idxs in activations: {idxs}"
        n_acts_before = len(acts_data['activations'])
        print("  ✓ Activations include idxs for alignment")
        
        # Verify activations_path in summary
        assert summary['activations_path'] is not None, "activations_path should not be None"
        print("  ✓ activations_path correctly reported")
        
        # Verify atomic progress.json (check no .tmp file left)
        tmp_progress = os.path.join(tmpdir, "test_run_test_condition_progress.tmp")
        assert not os.path.exists(tmp_progress), "Temp progress file should be cleaned up"
        print("  ✓ Atomic progress.json write")
        
        # =====================================================================
        # NEW TEST: Verify NO activation duplication on resume + continue
        # =====================================================================
        problems_extra = [FakeProblem(i) for i in range(7)]  # 7 total, 5 done
        runner3 = ExperimentRunner("test_run", tmpdir, "test_condition")
        
        for problem in runner3.iterate(problems_extra):
            # Only idx 5,6 should be yielded (0-4 already done)
            runner3.save_item({'idx': problem.idx, 'result': 'ok'}, fake_acts)
        
        runner3.finalize()
        
        # Check activations file - should have 4 + 2 = 6 (not 4 + 4 + 2 = 10 from duplication)
        acts_data2 = np.load(acts_path)
        n_acts_after = len(acts_data2['activations'])
        expected_acts = n_acts_before + 2  # 4 original + 2 new (idx 5, 6)
        assert n_acts_after == expected_acts, f"Activation duplication! Got {n_acts_after}, expected {expected_acts}"
        print("  ✓ No activation duplication on resume (v1.2 fix)")
        
        # =====================================================================
        # NEW TEST: Verify truncated lines are ignored
        # =====================================================================
        # Create a new test with a truncated line
        truncated_path = os.path.join(tmpdir, "truncated_test_results.jsonl")
        with open(truncated_path, 'w') as f:
            f.write('{"idx": 0, "status": "ok"}\n')
            f.write('{"idx": 1, "status": "ok"}\n')
            f.write('{"idx": 2, "status": "err')  # Truncated - no newline, incomplete JSON
        
        # Create a runner that would read this
        runner4 = ExperimentRunner("truncated", tmpdir, "test")
        runner4.results_path = Path(truncated_path)
        
        total, successful, failed, _ = runner4._recompute_counts_from_results()
        assert total == 2, f"Truncated line should be ignored, got total={total}"
        print("  ✓ Truncated lines ignored in resume (v1.2 fix)")
    
    print("\n✅ All checkpoint_utils tests passed! (v1.2)")
    return True


if __name__ == "__main__":
    print(f"Checkpoint Utilities v{__version__}")
    print("=" * 50)
    run_sanity_test()
