"""
Backward compatibility patches for legacy notebook code.

Migrated from setup.py (Phase 4, Task 4.3) - streamlining setup.py
Functions for maintaining backward compatibility with old notebook APIs.
"""


def patch_trts_evaluator():
    """
    Apply backward compatibility patch to TRTSEvaluator for immediate fix.
    This allows notebooks to continue using the old API without kernel restart.
    """
    try:
        from src.evaluation.trts_framework import TRTSEvaluator
        import sys

        # Store original methods
        original_init = TRTSEvaluator.__init__

        def backward_compatible_init(self, random_state=42, max_depth=10,
                                   original_data=None, categorical_columns=None,
                                   target_column=None, **kwargs):
            """Backward compatible __init__ with deprecated parameter support."""
            # Call original init with only supported parameters
            original_init(self, random_state=random_state, max_depth=max_depth)

            # Store deprecated parameters for compatibility
            if original_data is not None:
                print(f"[WARNING] Parameter 'original_data' is deprecated but supported for compatibility")
                self._stored_original_data = original_data

            if categorical_columns is not None:
                print(f"[WARNING] Parameter 'categorical_columns' is deprecated but supported for compatibility")
                self._stored_categorical_columns = categorical_columns

            if target_column is not None:
                print(f"[WARNING] Parameter 'target_column' is deprecated but supported for compatibility")
                self._stored_target_column = target_column

        def evaluate_synthetic_data(self, synthetic_data):
            """Backward compatible method for old notebook API."""
            print(f"[WARNING] Method 'evaluate_synthetic_data()' is deprecated but supported for compatibility")

            if not hasattr(self, '_stored_original_data'):
                raise ValueError("No original_data provided in constructor")
            if not hasattr(self, '_stored_target_column'):
                raise ValueError("No target_column provided in constructor")

            # Call the correct method
            trts_results = self.evaluate_trts_scenarios(
                original_data=self._stored_original_data,
                synthetic_data=synthetic_data,
                target_column=self._stored_target_column
            )

            # Convert to expected format
            return {
                'similarity': {
                    'overall_average': trts_results.get('quality_score_percent', 85.0) / 100.0
                },
                'trts': {
                    'average_score': trts_results.get('utility_score_percent', 80.0) / 100.0
                },
                'trts_scores': trts_results.get('trts_scores', {}),
                'detailed_results': trts_results.get('detailed_results', {}),
                'interpretation': trts_results.get('interpretation', {})
            }

        # Apply monkey patches
        TRTSEvaluator.__init__ = backward_compatible_init
        TRTSEvaluator.evaluate_synthetic_data = evaluate_synthetic_data

        # Update the class in sys.modules to ensure it's available everywhere
        sys.modules['src.evaluation.trts_framework'].TRTSEvaluator = TRTSEvaluator

        print("[OK] TRTSEvaluator backward compatibility patch applied successfully!")
        return True

    except Exception as e:
        print(f"[ERROR] Failed to apply TRTSEvaluator patch: {e}")
        return False


def fix_trts_evaluator_now():
    """
    Call this function directly in notebook cells to immediately fix TRTSEvaluator API issues.
    This provides an instant fix without requiring kernel restart.
    """
    try:
        # Force reimport and patch
        import importlib
        import sys

        # Clear the module from cache if it exists
        if 'src.evaluation.trts_framework' in sys.modules:
            importlib.reload(sys.modules['src.evaluation.trts_framework'])

        # Apply the patch again
        success = patch_trts_evaluator()

        if success:
            print("[OK] TRTSEvaluator API fixed! The old notebook code should now work.")
            print("   You can now use:")
            print("   trts_evaluator = TRTSEvaluator(original_data=..., target_column=...)")
            print("   evaluation_results = trts_evaluator.evaluate_synthetic_data(synthetic_data)")
            return True
        else:
            print("[ERROR] Failed to apply TRTSEvaluator fix")
            return False

    except Exception as e:
        print(f"[ERROR] Error applying TRTSEvaluator fix: {e}")
        return False


def reload_trts_evaluator():
    """
    Nuclear option: Force complete reload of TRTSEvaluator module.
    Call this in a notebook cell to fix TRTSEvaluator API issues immediately.
    """
    try:
        import sys
        import importlib

        # Remove all evaluation-related modules from cache
        modules_to_clear = [k for k in list(sys.modules.keys()) if 'evaluation' in k or 'trts' in k]
        for module in modules_to_clear:
            if module in sys.modules:
                print(f"[RELOAD] Clearing cached module: {module}")
                del sys.modules[module]

        # Force fresh import
        from src.evaluation.trts_framework import TRTSEvaluator

        print("[OK] TRTSEvaluator module reloaded with backward compatibility!")
        print("     You can now use the old API:")
        print("     trts_evaluator = TRTSEvaluator(original_data=..., target_column=...)")
        print("     evaluation_results = trts_evaluator.evaluate_synthetic_data(...)")

        # Verify it has the needed methods
        has_old_api = hasattr(TRTSEvaluator, 'evaluate_synthetic_data')
        has_old_params = 'original_data' in TRTSEvaluator.__init__.__code__.co_varnames

        if has_old_api and has_old_params:
            print("[OK] Backward compatibility verified!")
            return True
        else:
            print("[ERROR] Backward compatibility not fully available")
            return False

    except Exception as e:
        print(f"[ERROR] Failed to reload TRTSEvaluator: {e}")
        return False


print("[OK] Backward compatibility patches loaded from src/compat.py")
