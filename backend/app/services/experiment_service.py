"""
Experiment service — orchestrates training, evaluation, and persistence.
"""

import json
import ast
import logging
import threading
from datetime import datetime, timezone
from typing import Dict, List, Optional, Any, Tuple
from concurrent.futures import ThreadPoolExecutor

from ..models import ExperimentModel, LogEntryModel, ExperimentStatus, get_session_factory
from ..core.config import settings
from ..research.environment import SimulatorConfig, NudgingEnvironment, HealthState, Action
from ..research.policies import create_policy, BasePolicy
from ..research.evaluation import run_evaluation, evaluate_policy, EvaluationResult

logger = logging.getLogger(__name__)


class ExperimentService:
    """Service for managing experiment lifecycle."""
    
    def __init__(self, session_factory):
        self.session_factory = session_factory
        self.executor = ThreadPoolExecutor(max_workers=settings.MAX_CONCURRENT_JOBS)
        self._running_jobs: Dict[int, threading.Event] = {}
        self._trained_policies: Dict[int, BasePolicy] = {}
    
    def create_experiment(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Create and start an experiment."""
        session = self.session_factory()
        try:
            experiment = ExperimentModel(
                name=data['name'],
                description=data.get('description'),
                status=ExperimentStatus.QUEUED,
                algorithm=data['algorithm'],
                fatigue_aware=data.get('fatigue_aware', False),
                simulator_config=json.dumps(data.get('simulator_config', {})),
                algorithm_config=json.dumps(data.get('algorithm_config', {})),
                n_training_episodes=data.get('n_training_episodes'),
                n_eval_episodes=data.get('n_eval_episodes', 1000),
                n_replications=data.get('n_replications', 6),
                base_seed=data.get('base_seed', 42),
            )
            session.add(experiment)
            session.commit()
            session.refresh(experiment)
            
            exp_id = experiment.id
            exp_dict = experiment.to_dict()
            
            # Start background execution
            self.executor.submit(self._run_experiment, exp_id, data)
            
            return exp_dict
        except Exception as e:
            session.rollback()
            raise
        finally:
            session.close()
    
    def _run_experiment(self, experiment_id: int, data: Dict[str, Any]):
        """Run experiment in background thread."""
        session = self.session_factory()
        try:
            # Update status to running
            experiment = session.get(ExperimentModel, experiment_id)
            experiment.status = ExperimentStatus.RUNNING
            experiment.started_at = datetime.now(timezone.utc)
            session.commit()
            
            self._add_log(session, experiment_id, "INFO", "Experiment started")
            
            # Build simulator config
            sim_config_data = data.get('simulator_config', {})
            sim_config = SimulatorConfig(**sim_config_data)
            
            # Build algorithm config
            algo_config = data.get('algorithm_config', {})
            algorithm = data['algorithm']
            fatigue_aware = data.get('fatigue_aware', False)
            
            # Create policy
            self._add_log(session, experiment_id, "INFO", 
                         f"Creating {algorithm} policy (fatigue_aware={fatigue_aware})")
            
            policy = create_policy(algorithm, fatigue_aware=fatigue_aware, **algo_config)
            experiment.fatigue_aware = policy.fatigue_aware
            
            # Train if needed
            training_metrics = None
            if hasattr(policy, 'train') and not policy.trained:
                n_train = data.get('n_training_episodes')
                if n_train and hasattr(policy, 'n_episodes'):
                    policy.n_episodes = n_train
                
                self._add_log(session, experiment_id, "INFO", "Training started")
                training_metrics = policy.train(sim_config, seed=data.get('base_seed', 42))
                self._add_log(session, experiment_id, "INFO", 
                             f"Training completed: {training_metrics.iterations} iterations")
            
            # Evaluate
            self._add_log(session, experiment_id, "INFO", "Evaluation started")
            eval_result = run_evaluation(
                policy=policy,
                config=sim_config,
                n_replications=data.get('n_replications', 6),
                n_episodes_per_replication=data.get('n_eval_episodes', 1000),
                base_seed=data.get('base_seed', 42) + 100000,
            )
            self._add_log(session, experiment_id, "INFO", 
                         f"Evaluation completed: mean return = {eval_result.mean_discounted_return:.4f}")
            
            # Store results
            eval_dict = eval_result.to_dict()
            # Limit sample episode size for storage
            if eval_dict.get('sample_episode'):
                ep = eval_dict['sample_episode']
                # Keep trajectories but limit size
                if len(ep.get('health_trajectory', [])) > 100:
                    ep['health_trajectory'] = ep['health_trajectory'][:100]
            
            experiment = session.get(ExperimentModel, experiment_id)
            experiment.status = ExperimentStatus.COMPLETED
            experiment.completed_at = datetime.now(timezone.utc)
            experiment.evaluation_results = json.dumps(eval_dict, default=str)
            
            if training_metrics:
                tm = training_metrics.to_dict()
                # Subsample training curves for storage
                if len(tm.get('episode_returns', [])) > 1000:
                    step = len(tm['episode_returns']) // 1000
                    tm['episode_returns'] = tm['episode_returns'][::step]
                    if tm.get('exploration_rates'):
                        tm['exploration_rates'] = tm['exploration_rates'][::step]
                experiment.training_metrics = json.dumps(tm, default=str)
            
            # Store policy table
            policy_table = policy.get_policy_table()
            policy_serializable = {
                str(k): int(v) for k, v in policy_table.items()
            }
            experiment.policy_table = json.dumps(policy_serializable)
            
            # Denormalized metrics
            experiment.mean_discounted_return = eval_result.mean_discounted_return
            experiment.ci_95_lower = eval_result.ci_95_lower
            experiment.ci_95_upper = eval_result.ci_95_upper
            experiment.mean_health_score = eval_result.mean_health_score
            experiment.mean_response_rate = eval_result.mean_response_rate
            experiment.mean_habituation = eval_result.mean_habituation
            experiment.mean_nudges = eval_result.mean_nudges_per_episode
            
            session.commit()
            
            # Cache the trained policy
            self._trained_policies[experiment_id] = policy
            
            self._add_log(session, experiment_id, "INFO", "Experiment completed successfully")
            
        except Exception as e:
            logger.exception("Experiment %s failed: %s", experiment_id, e)
            try:
                experiment = session.get(ExperimentModel, experiment_id)
                if experiment:
                    experiment.status = ExperimentStatus.FAILED
                    experiment.error_message = str(e)
                    experiment.completed_at = datetime.now(timezone.utc)
                    session.commit()
                    self._add_log(session, experiment_id, "ERROR", f"Experiment failed: {e}")
            except Exception:
                logger.exception(
                    "Failed to persist failed status for experiment %s",
                    experiment_id,
                )
        finally:
            session.close()
    
    def get_experiment(self, experiment_id: int) -> Optional[Dict[str, Any]]:
        """Get experiment details."""
        session = self.session_factory()
        try:
            experiment = session.get(ExperimentModel, experiment_id)
            if experiment is None:
                return None
            return experiment.to_dict()
        finally:
            session.close()
    
    def get_experiment_results(self, experiment_id: int) -> Optional[Dict[str, Any]]:
        """Get full experiment results."""
        session = self.session_factory()
        try:
            experiment = session.get(ExperimentModel, experiment_id)
            if experiment is None:
                return None
            
            result = experiment.to_dict()
            result['evaluation'] = (
                json.loads(experiment.evaluation_results) 
                if experiment.evaluation_results else None
            )
            result['training_metrics'] = (
                json.loads(experiment.training_metrics)
                if experiment.training_metrics else None
            )
            return result
        finally:
            session.close()
    
    def get_experiment_policy(self, experiment_id: int) -> Optional[Dict[str, Any]]:
        """Get the policy table for an experiment."""
        session = self.session_factory()
        try:
            experiment = session.get(ExperimentModel, experiment_id)
            if experiment is None or experiment.policy_table is None:
                return None
            
            policy_raw = json.loads(experiment.policy_table)
            
            # Build structured policy entries
            config_data = json.loads(experiment.simulator_config)
            sim_config = SimulatorConfig(**config_data)
            env = NudgingEnvironment(sim_config)
            
            entries = []
            for state_str, action in policy_raw.items():
                state = ast.literal_eval(state_str)
                if not isinstance(state, tuple) or not all(
                    isinstance(value, int) for value in state
                ):
                    raise ValueError("Stored policy contains an invalid state key")
                action_int = int(action)
                
                # Compute response probability for this state+action
                hab = state[3] if len(state) > 3 else 0
                resp_prob = env.get_response_probability(action_int, hab)
                
                entries.append({
                    'state': list(state),
                    'action': action_int,
                    'action_name': Action(action_int).name,
                    'response_probability': resp_prob,
                })
            
            return {
                'algorithm': experiment.algorithm,
                'fatigue_aware': experiment.fatigue_aware,
                'total_states': len(entries),
                'entries': entries,
            }
        finally:
            session.close()
    
    def list_experiments(self, limit: int = 50, offset: int = 0) -> List[Dict[str, Any]]:
        """List all experiments."""
        session = self.session_factory()
        try:
            experiments = (
                session.query(ExperimentModel)
                .order_by(ExperimentModel.created_at.desc())
                .offset(offset)
                .limit(limit)
                .all()
            )
            return [e.to_dict() for e in experiments]
        finally:
            session.close()
    
    def get_experiment_count(self) -> int:
        """Get total number of experiments."""
        session = self.session_factory()
        try:
            return session.query(ExperimentModel).count()
        finally:
            session.close()
    
    def get_experiment_logs(self, experiment_id: int) -> List[Dict[str, Any]]:
        """Get log entries for an experiment."""
        session = self.session_factory()
        try:
            logs = (
                session.query(LogEntryModel)
                .filter(LogEntryModel.experiment_id == experiment_id)
                .order_by(LogEntryModel.timestamp.asc())
                .all()
            )
            return [
                {
                    'id': log.id,
                    'timestamp': log.timestamp.isoformat() if log.timestamp else None,
                    'level': log.level,
                    'message': log.message,
                }
                for log in logs
            ]
        finally:
            session.close()
    
    def rerun_experiment(self, experiment_id: int, new_seed: Optional[int] = None) -> Dict[str, Any]:
        """Re-run an experiment with the same or new configuration."""
        session = self.session_factory()
        try:
            original = session.get(ExperimentModel, experiment_id)
            if original is None:
                raise ValueError(f"Experiment {experiment_id} not found")
            
            data = {
                'name': f"{original.name} (rerun)",
                'description': f"Rerun of experiment #{original.id}",
                'algorithm': original.algorithm,
                'fatigue_aware': original.fatigue_aware,
                'simulator_config': json.loads(original.simulator_config),
                'algorithm_config': json.loads(original.algorithm_config) if original.algorithm_config else {},
                'n_training_episodes': original.n_training_episodes,
                'n_eval_episodes': original.n_eval_episodes,
                'n_replications': original.n_replications,
                'base_seed': new_seed if new_seed is not None else original.base_seed,
            }
            
            return self.create_experiment(data)
        finally:
            session.close()
    
    def export_results(self, experiment_id: int, format: str = 'json') -> Optional[Dict[str, Any]]:
        """Export experiment results in the specified format."""
        results = self.get_experiment_results(experiment_id)
        if results is None:
            return None
        
        if format == 'csv':
            # Build CSV-friendly data
            eval_data = results.get('evaluation')
            if eval_data and eval_data.get('replication_results'):
                import csv
                import io
                
                output = io.StringIO()
                writer = csv.DictWriter(output, fieldnames=[
                    'replication_id', 'seed', 'n_episodes',
                    'mean_discounted_return', 'std_discounted_return',
                    'mean_health_score', 'std_health_score',
                    'mean_nudges', 'mean_nudge_fraction',
                    'mean_response_rate', 'mean_habituation',
                ])
                writer.writeheader()
                for rep in eval_data['replication_results']:
                    writer.writerow(rep)
                
                return {
                    'format': 'csv',
                    'content': output.getvalue(),
                    'filename': f'experiment_{experiment_id}_results.csv',
                }
        
        return {
            'format': 'json',
            'content': results,
            'filename': f'experiment_{experiment_id}_results.json',
        }
    
    def _add_log(self, session, experiment_id: int, level: str, message: str):
        """Add a log entry."""
        try:
            log = LogEntryModel(
                experiment_id=experiment_id,
                level=level,
                message=message,
            )
            session.add(log)
            session.commit()
        except Exception:
            logger.exception("Failed to persist experiment log entry")
            session.rollback()
    
    def get_best_experiment(self) -> Optional[Dict[str, Any]]:
        """Get the experiment with the highest mean discounted return."""
        session = self.session_factory()
        try:
            experiment = (
                session.query(ExperimentModel)
                .filter(ExperimentModel.status == ExperimentStatus.COMPLETED)
                .filter(ExperimentModel.mean_discounted_return.isnot(None))
                .order_by(ExperimentModel.mean_discounted_return.desc())
                .first()
            )
            if experiment is None:
                return None
            return experiment.to_dict()
        finally:
            session.close()
    
    def get_dashboard_stats(self) -> Dict[str, Any]:
        """Get dashboard summary statistics."""
        session = self.session_factory()
        try:
            total = session.query(ExperimentModel).count()
            completed = session.query(ExperimentModel).filter(
                ExperimentModel.status == ExperimentStatus.COMPLETED
            ).count()
            running = session.query(ExperimentModel).filter(
                ExperimentModel.status == ExperimentStatus.RUNNING
            ).count()
            failed = session.query(ExperimentModel).filter(
                ExperimentModel.status == ExperimentStatus.FAILED
            ).count()
            
            best = self.get_best_experiment()
            
            latest = (
                session.query(ExperimentModel)
                .order_by(ExperimentModel.created_at.desc())
                .first()
            )
            
            return {
                'total_experiments': total,
                'completed': completed,
                'running': running,
                'failed': failed,
                'best_experiment': best,
                'latest_experiment': latest.to_dict() if latest else None,
            }
        finally:
            session.close()
