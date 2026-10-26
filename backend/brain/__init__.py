"""Phase 06 autonomous brain package."""

from .behavior import BehaviorConfig, BehaviorManager, BehaviorOutcome, BehaviorState
from .executor import ActionExecutor
from .goals import GoalEvaluator, GoalEvaluatorConfig
from .loop import AutonomousBrain, BrainConfig, BrainRuntimeState, BrainTickResult
from .models import (
    ActionRequest,
    ActionResult,
    ActionType,
    BrainStatus,
    Decision,
    ExecutionStatus,
    Goal,
    GoalStatus,
    GoalType,
    PersonalityProfile,
    Plan,
    PlanStep,
    PlanValidation,
    TargetDefinition,
    WorldState,
)
from .personality import DEFAULT_PERSONALITY, load_personality
from .planner import BrainPlanner, BrainPlannerConfig
from .reasoning import BrainReasoner, MockBrainLLMProvider
from .targets import TargetResolver
from .world_state import build_world_state

__all__ = [
    "ActionExecutor",
    "ActionRequest",
    "ActionResult",
    "ActionType",
    "AutonomousBrain",
    "BehaviorConfig",
    "BehaviorManager",
    "BehaviorOutcome",
    "BehaviorState",
    "BrainConfig",
    "BrainPlanner",
    "BrainPlannerConfig",
    "BrainReasoner",
    "BrainRuntimeState",
    "BrainStatus",
    "BrainTickResult",
    "DEFAULT_PERSONALITY",
    "Decision",
    "ExecutionStatus",
    "Goal",
    "GoalEvaluator",
    "GoalEvaluatorConfig",
    "GoalStatus",
    "GoalType",
    "MockBrainLLMProvider",
    "PersonalityProfile",
    "Plan",
    "PlanStep",
    "PlanValidation",
    "TargetDefinition",
    "TargetResolver",
    "WorldState",
    "build_world_state",
    "load_personality",
]
