"""
Staged Optimization Module for Synthetic Data Models

This module provides group sequential Optuna hyperparameter optimization with:
- Time estimation and trials/hour tracking
- Resume capability via pickle persistence
- Diminishing returns assessment
- Concise trial output

Phase 5 - February 2026
"""

import os
import time
import pickle
import hashlib
import inspect
import logging
from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional, Callable
from pathlib import Path
import pandas as pd
import numpy as np

try:
    import optuna
    from optuna.pruners import MedianPruner
    OPTUNA_AVAILABLE = True
except ImportError:
    OPTUNA_AVAILABLE = False

from .model_factory import ModelFactory
from .registry import resolve_models, get_model_display_name
from .search_spaces import get_search_space, get_pruner_config
import src.models.search_spaces as _search_spaces_module

logger = logging.getLogger(__name__)


def _get_search_space_hash(model_name: str, run_mode: str) -> str:
    """
    Compute a hash of the search space function source for a model + run_mode.

    If the search space code changes (different choices, ranges, etc.) the hash
    changes, which lets us auto-invalidate stale pickled Optuna studies.
    """
    # Get the private helper function for this model (e.g. _get_ctgan_search_space)
    func_name = f"_get_{model_name}_search_space"
    func = getattr(_search_spaces_module, func_name, None)
    if func is None:
        # Fallback: hash the whole module source
        source = inspect.getsource(_search_spaces_module)
    else:
        source = inspect.getsource(func)
    # Include run_mode so debug vs full always gets a different hash
    key = f"{run_mode}:{source}"
    return hashlib.sha256(key.encode()).hexdigest()[:16]


@dataclass
class StagedOptimizationConfig:
    """Configuration for staged optimization."""
    pilot_trials: int = 15
    verbose_level: int = 1  # 0=silent, 1=concise, 2=detailed
    persistence_dir: str = "results/optimization_studies"
    run_mode: str = "full"  # "debug" or "full"
    random_state: int = 42
    continue_on_error: bool = True


@dataclass
class ModelOptimizationState:
    """State of optimization for a single model."""
    model_name: str
    study: Optional['optuna.Study'] = None
    total_trials: int = 0
    best_score: float = 0.0
    best_params: Dict[str, Any] = field(default_factory=dict)
    trial_times: List[float] = field(default_factory=list)
    score_history: List[float] = field(default_factory=list)
    status: str = "pending"  # "pending", "running", "completed", "error"
    error_message: Optional[str] = None
    run_mode: Optional[str] = None  # "debug" or "full" — used to invalidate stale studies
    search_space_hash: Optional[str] = None  # hash of search space source code


class TrialTimeTracker:
    """Track trial durations and estimate trials/hour."""

    def __init__(self):
        self.trial_times: Dict[str, List[float]] = {}
        self.start_times: Dict[str, float] = {}

    def start_trial(self, model_name: str) -> None:
        """Mark the start of a trial."""
        self.start_times[model_name] = time.time()

    def end_trial(self, model_name: str) -> float:
        """Mark the end of a trial and return duration."""
        if model_name not in self.start_times:
            return 0.0

        duration = time.time() - self.start_times[model_name]

        if model_name not in self.trial_times:
            self.trial_times[model_name] = []
        self.trial_times[model_name].append(duration)

        return duration

    def get_trials_per_hour(self, model_name: str) -> float:
        """Calculate trials per hour for a model."""
        if model_name not in self.trial_times or len(self.trial_times[model_name]) == 0:
            return 0.0

        avg_time = np.mean(self.trial_times[model_name])
        if avg_time == 0:
            return float('inf')
        return 3600.0 / avg_time

    def get_avg_trial_time(self, model_name: str) -> float:
        """Get average trial time in seconds."""
        if model_name not in self.trial_times or len(self.trial_times[model_name]) == 0:
            return 0.0
        return np.mean(self.trial_times[model_name])

    def estimate_time_for_trials(self, model_name: str, n_trials: int) -> float:
        """Estimate time in minutes for n additional trials."""
        avg_time = self.get_avg_trial_time(model_name)
        return (avg_time * n_trials) / 60.0

    def get_time_estimates_report(self) -> pd.DataFrame:
        """Generate a report of time estimates for all models."""
        data = []
        for model_name, times in self.trial_times.items():
            if len(times) > 0:
                data.append({
                    'model': model_name,
                    'trials_completed': len(times),
                    'avg_trial_seconds': np.mean(times),
                    'trials_per_hour': self.get_trials_per_hour(model_name),
                    'time_for_10_trials_min': self.estimate_time_for_trials(model_name, 10),
                    'time_for_50_trials_min': self.estimate_time_for_trials(model_name, 50),
                })
        return pd.DataFrame(data)


class ConvergenceAnalyzer:
    """Assess diminishing returns using improvement rate."""

    def __init__(self, window_size: int = 5):
        self.window_size = window_size

    def calculate_improvement_rate(self, scores: List[float]) -> float:
        """
        Calculate the improvement rate over recent trials.

        Returns a value between 0 and 1:
        - 0 means no improvement (plateau reached)
        - 1 means maximum improvement
        """
        if len(scores) < 2:
            return 1.0  # Not enough data, assume still improving

        # Use last window_size scores
        recent_scores = scores[-self.window_size:]

        if len(recent_scores) < 2:
            return 1.0

        # Calculate improvement: difference between best and worst in window
        best_recent = max(recent_scores)
        worst_recent = min(recent_scores)

        # Also consider overall best vs recent best
        overall_best = max(scores)

        # Improvement rate based on how much recent trials improved
        if overall_best == 0:
            return 1.0

        # Rate of improvement in recent window
        window_improvement = (best_recent - worst_recent) / max(overall_best, 0.001)

        return min(1.0, max(0.0, window_improvement))

    def has_plateaued(self, scores: List[float], threshold: float = 0.01) -> bool:
        """Check if optimization has plateaued."""
        if len(scores) < self.window_size:
            return False
        return self.calculate_improvement_rate(scores) < threshold

    def get_convergence_report(
        self,
        model_states: Dict[str, ModelOptimizationState]
    ) -> pd.DataFrame:
        """Generate a diminishing returns report for all models."""
        data = []
        for model_name, state in model_states.items():
            if len(state.score_history) > 0:
                improvement_rate = self.calculate_improvement_rate(state.score_history)
                has_plateau = self.has_plateaued(state.score_history)

                # Calculate score improvement from first to best
                first_score = state.score_history[0]
                best_score = state.best_score
                total_improvement = best_score - first_score

                data.append({
                    'model': model_name,
                    'trials': state.total_trials,
                    'best_score': state.best_score,
                    'improvement_rate': improvement_rate,
                    'total_improvement': total_improvement,
                    'has_plateaued': has_plateau,
                    'recommendation': self._get_recommendation(improvement_rate, state.total_trials)
                })

        return pd.DataFrame(data)

    def _get_recommendation(self, improvement_rate: float, trials: int) -> str:
        """Generate recommendation based on improvement rate and trials."""
        if trials < 10:
            return "Continue - insufficient data"
        elif improvement_rate > 0.05:
            return "Continue - still improving"
        elif improvement_rate > 0.01:
            return "Consider stopping - minor improvements"
        else:
            return "Stop - plateau reached"


class StudyPersistence:
    """Persist and load Optuna studies via pickle."""

    def __init__(self, persistence_dir: str = "results/optimization_studies"):
        self.persistence_dir = Path(persistence_dir)
        self.persistence_dir.mkdir(parents=True, exist_ok=True)

    def _get_study_path(self, model_name: str, dataset_identifier: str) -> Path:
        """Get the path for a study file."""
        return self.persistence_dir / f"{dataset_identifier}_{model_name}_study.pkl"

    def _get_state_path(self, model_name: str, dataset_identifier: str) -> Path:
        """Get the path for a state file."""
        return self.persistence_dir / f"{dataset_identifier}_{model_name}_state.pkl"

    def save_study(
        self,
        model_name: str,
        study: 'optuna.Study',
        state: ModelOptimizationState,
        dataset_identifier: str
    ) -> bool:
        """Save study and state to disk."""
        try:
            study_path = self._get_study_path(model_name, dataset_identifier)
            state_path = self._get_state_path(model_name, dataset_identifier)

            with open(study_path, 'wb') as f:
                pickle.dump(study, f)

            with open(state_path, 'wb') as f:
                pickle.dump(state, f)

            return True
        except Exception as e:
            logger.error(f"Failed to save study for {model_name}: {e}")
            return False

    def load_study(
        self,
        model_name: str,
        dataset_identifier: str
    ) -> Optional[tuple]:
        """Load study and state from disk. Returns (study, state) or None."""
        try:
            study_path = self._get_study_path(model_name, dataset_identifier)
            state_path = self._get_state_path(model_name, dataset_identifier)

            if not study_path.exists() or not state_path.exists():
                return None

            with open(study_path, 'rb') as f:
                study = pickle.load(f)

            with open(state_path, 'rb') as f:
                state = pickle.load(f)

            return study, state
        except Exception as e:
            logger.error(f"Failed to load study for {model_name}: {e}")
            return None

    def study_exists(self, model_name: str, dataset_identifier: str) -> bool:
        """Check if a saved study exists."""
        study_path = self._get_study_path(model_name, dataset_identifier)
        state_path = self._get_state_path(model_name, dataset_identifier)
        return study_path.exists() and state_path.exists()

    def list_saved_studies(self, dataset_identifier: str) -> List[str]:
        """List all saved studies for a dataset."""
        studies = []
        for path in self.persistence_dir.glob(f"{dataset_identifier}_*_study.pkl"):
            model_name = path.stem.replace(f"{dataset_identifier}_", "").replace("_study", "")
            studies.append(model_name)
        return studies

    def flush_studies(self, dataset_identifier: str) -> int:
        """Delete all saved study/state pickle files for a dataset.

        Returns the number of files removed.
        """
        removed = 0
        for pattern in (f"{dataset_identifier}_*_study.pkl", f"{dataset_identifier}_*_state.pkl"):
            for path in self.persistence_dir.glob(pattern):
                path.unlink()
                removed += 1
        return removed


def flush_previous_runs(dataset_identifier: str, persistence_dir: str = None) -> None:
    """Remove all saved Optuna study pickles for *dataset_identifier*.

    Call this at the top of a notebook to guarantee a clean run that starts
    from trial 0 instead of resuming a prior session.

    Parameters
    ----------
    dataset_identifier : str
        The dataset identifier (e.g. ``"breast-cancer-data"``).
    persistence_dir : str, optional
        Path to the optimization_studies directory.  When *None* the default
        ``results/<dataset_identifier>/optimization_studies`` is used.
    """
    if persistence_dir is None:
        persistence_dir = f"results/{dataset_identifier}/optimization_studies"

    store = StudyPersistence(persistence_dir)
    n = store.flush_studies(dataset_identifier)
    if n:
        print(f"[FLUSH] Removed {n} pickle file(s) from {persistence_dir}")
    else:
        print(f"[FLUSH] No previous studies found in {persistence_dir} — starting clean")


class ConciseTrialCallback:
    """Callback for one-line output per trial."""

    def __init__(self, model_name: str, verbose: bool = True):
        self.model_name = model_name
        self.verbose = verbose
        self.cumulative_trial_count = 0

    def __call__(self, study: 'optuna.Study', trial: 'optuna.trial.FrozenTrial') -> None:
        """Called after each trial completes."""
        if not self.verbose:
            return

        self.cumulative_trial_count += 1

        # Get component scores from user attributes
        similarity = trial.user_attrs.get('similarity_score', 0.0)
        accuracy = trial.user_attrs.get('accuracy_score', 0.0)

        # Format output
        combined_score = trial.value if trial.value is not None else 0.0
        best_score = study.best_value if study.best_trial is not None else 0.0

        print(f"[{self.model_name}] Trial {self.cumulative_trial_count}: "
              f"Combined Score: {combined_score:.4f} "
              f"(Similarity: {similarity:.4f}, Accuracy: {accuracy:.4f}) "
              f"Best Score so far: {best_score:.4f}")


class StagedOptimizationManager:
    """Main orchestrator for staged optimization."""

    def __init__(
        self,
        data: pd.DataFrame,
        target_column: str,
        categorical_columns: List[str] = None,
        dataset_identifier: str = "default",
        config: StagedOptimizationConfig = None
    ):
        if not OPTUNA_AVAILABLE:
            raise ImportError("Optuna is required for staged optimization")

        self.data = data
        self.target_column = target_column
        self.categorical_columns = categorical_columns or []
        self.dataset_identifier = dataset_identifier
        self.config = config or StagedOptimizationConfig()

        # Components
        self.time_tracker = TrialTimeTracker()
        self.convergence_analyzer = ConvergenceAnalyzer()
        self.persistence = StudyPersistence(self.config.persistence_dir)

        # State
        self.model_states: Dict[str, ModelOptimizationState] = {}
        self._models_to_run: List[str] = []

    def run_pilot(
        self,
        models_to_run: List[str],
        n_trials: int = None
    ) -> Dict[str, ModelOptimizationState]:
        """
        Run pilot optimization phase to establish time estimates.

        Parameters:
        -----------
        models_to_run : List[str]
            List of model names to optimize
        n_trials : int, optional
            Number of pilot trials (default: config.pilot_trials)

        Returns:
        --------
        Dict[str, ModelOptimizationState] : State for each model
        """
        if n_trials is None:
            n_trials = self.config.pilot_trials

        self._models_to_run = resolve_models(models_to_run)

        print(f"\n{'='*60}")
        print("STAGED OPTIMIZATION - PILOT PHASE")
        print(f"{'='*60}")
        print(f"Models to optimize: {len(self._models_to_run)}")
        print(f"Pilot trials per model: {n_trials}")
        print(f"Dataset identifier: {self.dataset_identifier}")
        print(f"{'='*60}\n")

        for model_name in self._models_to_run:
            self._run_model_optimization(model_name, n_trials, is_pilot=True)

        # Print time estimates
        self._print_time_estimates()

        return self.model_states

    def continue_optimization(
        self,
        additional_trials: int = None,
        trials_per_model: Dict[str, int] = None,
        time_budget_minutes: Dict[str, float] = None
    ) -> Dict[str, ModelOptimizationState]:
        """
        Continue optimization with additional trials.

        Three options for specifying continuation:
        (i) additional_trials: Common number of trials for all models
        (ii) trials_per_model: Specific number of trials per model
        (iii) time_budget_minutes: Time budget per model (estimates trials)

        Parameters:
        -----------
        additional_trials : int, optional
            Common number of additional trials for all models
        trials_per_model : Dict[str, int], optional
            Model-specific trial counts
        time_budget_minutes : Dict[str, float], optional
            Time budget per model in minutes

        Returns:
        --------
        Dict[str, ModelOptimizationState] : Updated state for each model
        """
        if sum([additional_trials is not None,
                trials_per_model is not None,
                time_budget_minutes is not None]) != 1:
            raise ValueError(
                "Exactly one of additional_trials, trials_per_model, or "
                "time_budget_minutes must be specified"
            )

        # Convert time budget to trials
        if time_budget_minutes is not None:
            trials_per_model = {}
            for model_name, minutes in time_budget_minutes.items():
                avg_time = self.time_tracker.get_avg_trial_time(model_name)
                if avg_time > 0:
                    trials_per_model[model_name] = max(1, int((minutes * 60) / avg_time))
                else:
                    trials_per_model[model_name] = 10  # Default if no time data

        # Convert common trials to per-model
        if additional_trials is not None:
            trials_per_model = {
                model_name: additional_trials
                for model_name in self._models_to_run
            }

        print(f"\n{'='*60}")
        print("STAGED OPTIMIZATION - CONTINUATION PHASE")
        print(f"{'='*60}")
        for model_name, trials in trials_per_model.items():
            print(f"  {model_name}: {trials} additional trials")
        print(f"{'='*60}\n")

        for model_name, n_trials in trials_per_model.items():
            if model_name in self._models_to_run or model_name in self.model_states:
                self._run_model_optimization(model_name, n_trials, is_pilot=False)

        # Print time estimates
        self._print_time_estimates()

        return self.model_states

    def get_diminishing_returns_report(self) -> pd.DataFrame:
        """Get a report of diminishing returns for all models."""
        return self.convergence_analyzer.get_convergence_report(self.model_states)

    def get_time_estimates(self) -> pd.DataFrame:
        """Get time estimates for all models."""
        return self.time_tracker.get_time_estimates_report()

    def get_best_params_summary(self) -> pd.DataFrame:
        """Get a summary of best parameters for all models."""
        data = []
        for model_name, state in self.model_states.items():
            if state.status == "completed" or state.total_trials > 0:
                data.append({
                    'model': model_name,
                    'best_score': state.best_score,
                    'total_trials': state.total_trials,
                    'status': state.status,
                    'best_params': str(state.best_params)
                })
        return pd.DataFrame(data)

    def get_smoke_recommendations(self) -> pd.DataFrame:
        """Generate smoke-mode recommendations: 1-hour budget and 20-trial estimates."""
        rows = []
        for model_name in self._models_to_run:
            tph = self.time_tracker.get_trials_per_hour(model_name)
            avg_s = self.time_tracker.get_avg_trial_time(model_name)
            t20 = self.time_tracker.estimate_time_for_trials(model_name, 20)
            trials_in_1hr = int(tph) if tph > 0 else 0
            recommended = min(trials_in_1hr, 50) if trials_in_1hr > 0 else 10
            rows.append({
                'model': model_name,
                'avg_trial_sec': round(avg_s, 1),
                'trials_per_hour': round(tph, 1),
                'trials_in_1hr': trials_in_1hr,
                'time_for_20_trials_min': round(t20, 1),
                'recommended_pilot': recommended,
            })
        return pd.DataFrame(rows)

    def _create_fresh_study(self, model_name: str) -> 'optuna.Study':
        """Create a new Optuna study for a model."""
        pruner_config = get_pruner_config(model_name)
        pruner = MedianPruner(**pruner_config) if pruner_config else None

        return optuna.create_study(
            direction="maximize",
            pruner=pruner,
            study_name=f"{model_name}_hpo_{self.dataset_identifier}"
        )

    def _run_model_optimization(
        self,
        model_name: str,
        n_trials: int,
        is_pilot: bool = True
    ) -> None:
        """Run optimization for a single model."""
        display_name = get_model_display_name(model_name)

        print(f"\n[{'PILOT' if is_pilot else 'CONTINUE'}] Optimizing {display_name}...")
        print(f"{'-'*50}")

        try:
            # Try to load existing study
            loaded = self.persistence.load_study(model_name, self.dataset_identifier)

            # Compute current search space hash for compatibility check
            current_hash = _get_search_space_hash(model_name, self.config.run_mode)

            if loaded is not None:
                study, state = loaded
                saved_hash = getattr(state, 'search_space_hash', None)
                saved_mode = getattr(state, 'run_mode', None)

                # Invalidate if run_mode or search space code changed
                if saved_hash is not None and saved_hash != current_hash:
                    reason = f"run_mode changed ({saved_mode} -> {self.config.run_mode})" \
                        if saved_mode != self.config.run_mode \
                        else "search space code changed"
                    print(f"  [WARN] Incompatible study ({reason}). Starting fresh.")
                    study = self._create_fresh_study(model_name)
                    state = ModelOptimizationState(model_name=model_name)
                elif saved_mode is not None and saved_mode != self.config.run_mode:
                    # Fallback for old pickles without hash
                    print(f"  [WARN] run_mode changed ({saved_mode} -> {self.config.run_mode}). Starting fresh.")
                    study = self._create_fresh_study(model_name)
                    state = ModelOptimizationState(model_name=model_name)
                else:
                    print(f"  Resuming from {state.total_trials} existing trials")
            else:
                # Create new study
                study = self._create_fresh_study(model_name)
                state = ModelOptimizationState(model_name=model_name)

            # Tag the state with current run_mode and hash for future checks
            state.run_mode = self.config.run_mode
            state.search_space_hash = current_hash

            # Create objective and callback
            objective = self._create_objective(model_name)
            callback = ConciseTrialCallback(
                model_name,
                verbose=(self.config.verbose_level >= 1)
            )
            callback.cumulative_trial_count = state.total_trials

            # Suppress Optuna logging for cleaner output
            optuna.logging.set_verbosity(optuna.logging.WARNING)

            # Run optimization
            state.status = "running"
            start_time = time.time()

            for trial_idx in range(n_trials):
                self.time_tracker.start_trial(model_name)

                try:
                    study.optimize(
                        objective,
                        n_trials=1,
                        callbacks=[callback],
                        show_progress_bar=False
                    )
                except optuna.TrialPruned:
                    pass
                except Exception as e:
                    if "dynamic value space" in str(e):
                        # Search space changed mid-study — reset and retry
                        print(f"  [WARN] Incompatible study detected. Starting fresh.")
                        study = self._create_fresh_study(model_name)
                        state = ModelOptimizationState(model_name=model_name)
                        state.run_mode = self.config.run_mode
                        callback.cumulative_trial_count = 0
                        try:
                            study.optimize(objective, n_trials=1, callbacks=[callback], show_progress_bar=False)
                        except Exception:
                            pass
                    elif not self.config.continue_on_error:
                        raise
                    else:
                        logger.warning(f"Trial failed for {model_name}: {e}")

                self.time_tracker.end_trial(model_name)

                # Update state
                state.total_trials += 1
                if study.best_trial is not None:
                    state.best_score = study.best_value
                    state.best_params = study.best_params
                    state.score_history.append(study.best_value)

            state.study = study
            state.status = "completed"

            # Save to disk
            self.persistence.save_study(
                model_name, study, state, self.dataset_identifier
            )

            self.model_states[model_name] = state

            total_time = time.time() - start_time
            print(f"  [OK] {display_name}: {n_trials} trials in {total_time:.1f}s")
            print(f"  Best score: {state.best_score:.4f}")

        except Exception as e:
            state = ModelOptimizationState(
                model_name=model_name,
                status="error",
                error_message=str(e)
            )
            self.model_states[model_name] = state
            print(f"  [ERROR] {display_name} failed: {e}")

            if not self.config.continue_on_error:
                raise

    def _create_objective(self, model_name: str) -> Callable:
        """Create an Optuna objective function for a model."""
        from src.objective.functions import enhanced_objective_function_v2

        def objective(trial: 'optuna.Trial') -> float:
            try:
                # Get hyperparameters from search space
                params = get_search_space(
                    model_name,
                    trial,
                    self.config.run_mode,
                    data_size=len(self.data),
                    n_cols=self.data.shape[1]
                )

                # Create and configure model
                model = ModelFactory.create(
                    model_name,
                    device="cpu",
                    random_state=self.config.random_state
                )

                # Get model-specific training kwargs
                train_kwargs = self._get_train_kwargs(model_name, params)

                # Train model
                model.train(self.data, **train_kwargs)

                # Generate synthetic data
                synthetic_data = model.generate(len(self.data))

                # Coerce target dtype to match real data
                synthetic_data = self._coerce_target_dtype(synthetic_data)

                # Evaluate
                score, _, _ = enhanced_objective_function_v2(
                    self.data,
                    synthetic_data,
                    self.target_column,
                    trial=trial
                )

                return float(score)

            except optuna.TrialPruned:
                raise
            except Exception as e:
                logger.warning(f"Trial {trial.number} for {model_name} failed: {e}")
                return 0.0

        return objective

    def _get_train_kwargs(
        self,
        model_name: str,
        params: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Get training kwargs for a model with column filtering."""
        model_name_lower = model_name.lower()

        # Filter categorical columns to only those that exist
        existing_columns = set(self.data.columns)
        categorical_columns = [
            col for col in self.categorical_columns
            if col in existing_columns
        ]

        kwargs = {}

        # Common parameters
        if "epochs" in params:
            kwargs["epochs"] = params["epochs"]
        if "batch_size" in params:
            kwargs["batch_size"] = params["batch_size"]

        # Model-specific parameter mapping
        if model_name_lower == "ctgan":
            kwargs.update({
                "discrete_columns": categorical_columns,
                "generator_lr": params.get("generator_lr"),
                "discriminator_lr": params.get("discriminator_lr"),
                "generator_dim": params.get("generator_dim"),
                "discriminator_dim": params.get("discriminator_dim"),
                "pac": params.get("pac"),
                "discriminator_steps": params.get("discriminator_steps"),
                "generator_decay": params.get("generator_decay"),
                "discriminator_decay": params.get("discriminator_decay"),
            })
            kwargs = {k: v for k, v in kwargs.items() if v is not None}

        elif model_name_lower in ["ctabgan", "ctabganplus"]:
            kwargs.update({
                "categorical_columns": categorical_columns,
                "target_col": self.target_column,
                "test_ratio": params.get("test_ratio", 0.2),
            })

        elif model_name_lower == "ganeraid":
            kwargs.update({
                "categorical_columns": categorical_columns,
                "nr_of_rows": params.get("nr_of_rows"),
                "hidden_feature_space": params.get("hidden_feature_space"),
                "lr_d": params.get("lr_d"),
                "lr_g": params.get("lr_g"),
                "binary_noise": params.get("binary_noise"),
                "generator_decay": params.get("generator_decay"),
                "discriminator_decay": params.get("discriminator_decay"),
                "dropout_generator": params.get("dropout_generator"),
                "dropout_discriminator": params.get("dropout_discriminator"),
            })
            kwargs = {k: v for k, v in kwargs.items() if v is not None}

        elif model_name_lower == "copulagan":
            kwargs.update({
                "discrete_columns": categorical_columns,
                "generator_lr": params.get("generator_lr"),
                "discriminator_lr": params.get("discriminator_lr"),
                "generator_decay": params.get("generator_decay"),
                "discriminator_decay": params.get("discriminator_decay"),
            })
            kwargs = {k: v for k, v in kwargs.items() if v is not None}

        elif model_name_lower == "tvae":
            kwargs.update({
                "discrete_columns": categorical_columns,
                "learning_rate": params.get("learning_rate"),
                "embedding_dim": params.get("embedding_dim"),
                "l2scale": params.get("l2scale"),
                "compress_dims": params.get("compress_dims"),
                "decompress_dims": params.get("decompress_dims"),
            })
            kwargs = {k: v for k, v in kwargs.items() if v is not None}

        elif model_name_lower == "pategan":
            kwargs.update({
                "discrete_columns": categorical_columns,
                "generator_lr": params.get("generator_lr"),
                "discriminator_lr": params.get("discriminator_lr"),
                "generator_dim": params.get("generator_dim"),
                "discriminator_dim": params.get("discriminator_dim"),
                "generator_decay": params.get("generator_decay"),
                "discriminator_decay": params.get("discriminator_decay"),
                "num_teachers": params.get("num_teachers"),
                "noise_multiplier": params.get("noise_multiplier"),
                "target_epsilon": params.get("target_epsilon"),
                "lap_scale": params.get("lap_scale"),
            })
            kwargs = {k: v for k, v in kwargs.items() if v is not None}

        elif model_name_lower == "medgan":
            kwargs.update({
                "discrete_columns": categorical_columns,
                "pretrain_epochs": params.get("pretrain_epochs"),
                "latent_dim": params.get("latent_dim"),
                "autoencoder_lr": params.get("autoencoder_lr"),
                "generator_lr": params.get("generator_lr"),
                "discriminator_lr": params.get("discriminator_lr"),
                "l2_reg": params.get("l2_reg"),
                "dropout": params.get("dropout"),
                "autoencoder_dim": params.get("autoencoder_dim"),
                "generator_dim": params.get("generator_dim"),
                "discriminator_dim": params.get("discriminator_dim"),
            })
            kwargs = {k: v for k, v in kwargs.items() if v is not None}

        return kwargs

    def _coerce_target_dtype(self, synthetic_data: pd.DataFrame) -> pd.DataFrame:
        """Coerce synthetic target column dtype to match real data."""
        if (self.target_column not in self.data.columns or
            self.target_column not in synthetic_data.columns):
            return synthetic_data

        result = synthetic_data.copy()
        real_dtype = self.data[self.target_column].dtype

        try:
            if pd.api.types.is_numeric_dtype(real_dtype):
                if pd.api.types.is_integer_dtype(real_dtype):
                    result[self.target_column] = pd.to_numeric(
                        result[self.target_column], errors='coerce'
                    ).fillna(0).astype(int)
                else:
                    result[self.target_column] = pd.to_numeric(
                        result[self.target_column], errors='coerce'
                    )
            else:
                result[self.target_column] = result[self.target_column].astype(str)
        except Exception as e:
            logger.warning(f"Could not coerce target dtype: {e}")

        return result

    def _print_time_estimates(self) -> None:
        """Print time estimates for all models."""
        print(f"\n{'='*60}")
        print("TIME ESTIMATES")
        print(f"{'='*60}")

        for model_name in self._models_to_run:
            trials_per_hour = self.time_tracker.get_trials_per_hour(model_name)
            avg_time = self.time_tracker.get_avg_trial_time(model_name)

            if trials_per_hour > 0:
                time_for_50 = self.time_tracker.estimate_time_for_trials(model_name, 50)
                print(f"  {model_name}:")
                print(f"    Avg trial time: {avg_time:.1f}s")
                print(f"    Trials/hour: {trials_per_hour:.1f}")
                print(f"    Time for 50 more trials: {time_for_50:.1f} min")

        print(f"{'='*60}\n")


print("[OK] Staged optimization module loaded from src/models/staged_optimization.py")
