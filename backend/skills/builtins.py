"""Stable built-in skill definitions backed by Phase 01-08 actions."""

from __future__ import annotations

from ..brain.models import ActionType
from .models import (
    Skill,
    SkillActionStep,
    SkillCategory,
    SkillCompositionStep,
    SkillCondition,
    SkillConditionOperator,
    SkillConditionType,
    SkillParameter,
    SkillParameterType,
)


def _target_parameter(description: str = "world target to use") -> SkillParameter:
    return SkillParameter(
        name="target",
        parameter_type=SkillParameterType.TARGET,
        required=True,
        description=description,
    )


def build_builtin_skills() -> tuple[Skill, ...]:
    walk_to = Skill(
        skill_id="walk_to",
        name="Walk To",
        description="Navigates Lacrimosa to a registered world target.",
        category=SkillCategory.MOVEMENT,
        preconditions=(
            SkillCondition(SkillConditionType.TARGET_EXISTS, SkillConditionOperator.PRESENT, failure_reason="TARGET_NOT_FOUND"),
            SkillCondition(SkillConditionType.TARGET_REACHABLE, SkillConditionOperator.IS_TRUE, failure_reason="TARGET_UNREACHABLE"),
        ),
        parameters=(_target_parameter(),),
        steps=(SkillActionStep(ActionType.MOVE_TO, target_parameter="target"),),
        effects=("character_position_changed",),
        tags=("movement", "navigation"),
    )
    stop = Skill(
        skill_id="stop",
        name="Stop",
        description="Stops current navigation and returns to a controlled stop.",
        category=SkillCategory.MOVEMENT,
        steps=(SkillActionStep(ActionType.STOP),),
        effects=("character_stopped",),
        tags=("movement",),
    )
    turn_to = Skill(
        skill_id="turn_to",
        name="Turn To",
        description="Turns Lacrimosa toward a registered world target.",
        category=SkillCategory.MOVEMENT,
        preconditions=(
            SkillCondition(SkillConditionType.TARGET_EXISTS, SkillConditionOperator.PRESENT, failure_reason="TARGET_NOT_FOUND"),
        ),
        parameters=(_target_parameter(),),
        steps=(SkillActionStep(ActionType.TURN_TO, target_parameter="target"),),
        effects=("character_rotation_changed",),
        tags=("movement",),
    )
    look_at = Skill(
        skill_id="look_at",
        name="Look At",
        description="Directs Lacrimosa's head and upper body toward a target.",
        category=SkillCategory.MOVEMENT,
        preconditions=(
            SkillCondition(SkillConditionType.TARGET_EXISTS, SkillConditionOperator.PRESENT, failure_reason="TARGET_NOT_FOUND"),
        ),
        parameters=(_target_parameter(),),
        steps=(SkillActionStep(ActionType.LOOK_AT, target_parameter="target"),),
        effects=("character_gaze_changed",),
        tags=("movement", "attention"),
    )
    wave = Skill(
        skill_id="wave",
        name="Wave",
        description="Plays Lacrimosa's existing waving animation.",
        category=SkillCategory.SOCIAL,
        preconditions=(
            SkillCondition(SkillConditionType.CHARACTER_STATE, SkillConditionOperator.EQUALS, "standing", failure_reason="CHARACTER_STATE_INCOMPATIBLE"),
        ),
        steps=(SkillActionStep(ActionType.WAVE),),
        animation_reference="Assets/animation/waving",
        estimated_duration_seconds=3.0,
        effects=("entertainment +0.2", "social_engagement +0.1"),
        cooldown_seconds=2.0,
        tags=("social", "animation"),
    )
    dance = Skill(
        skill_id="dance",
        name="Dance",
        description="Dances in a suitably open area.",
        category=SkillCategory.SOCIAL,
        preconditions=(
            SkillCondition(SkillConditionType.CHARACTER_STATE, SkillConditionOperator.EQUALS, "standing", failure_reason="CHARACTER_STATE_INCOMPATIBLE"),
            SkillCondition(SkillConditionType.SPACE_AVAILABLE, SkillConditionOperator.IS_TRUE, failure_reason="SPACE_UNAVAILABLE"),
        ),
        steps=(SkillActionStep(ActionType.DANCE),),
        animation_reference="Assets/animation/dance",
        estimated_duration_seconds=8.0,
        effects=("entertainment +0.3", "social_engagement +0.2"),
        cooldown_seconds=5.0,
        tags=("social", "entertainment", "animation"),
    )
    run = Skill(
        skill_id="run",
        name="Run",
        description="Runs toward a registered navigable target.",
        category=SkillCategory.MOVEMENT,
        preconditions=(
            SkillCondition(SkillConditionType.TARGET_EXISTS, SkillConditionOperator.PRESENT, failure_reason="TARGET_NOT_FOUND"),
            SkillCondition(SkillConditionType.TARGET_REACHABLE, SkillConditionOperator.IS_TRUE, failure_reason="TARGET_UNREACHABLE"),
        ),
        parameters=(_target_parameter(),),
        steps=(SkillActionStep(ActionType.RUN, target_parameter="target"),),
        effects=("character_position_changed",),
        tags=("movement", "navigation"),
    )
    pick_up_phone = Skill(
        skill_id="pick_up_phone",
        name="Pick Up Phone",
        description="Uses the existing smartphone interaction system to pick up the phone.",
        category=SkillCategory.PHONE,
        preconditions=(
            SkillCondition(SkillConditionType.PHONE_STATE, SkillConditionOperator.IN, ("World", "Placed"), failure_reason="PHONE_NOT_PICKABLE"),
        ),
        steps=(SkillActionStep(ActionType.PICK_UP_PHONE),),
        effects=("phone_state = held",),
        cooldown_seconds=1.0,
        tags=("phone", "interaction"),
    )
    place_phone = Skill(
        skill_id="place_phone",
        name="Place Phone",
        description="Places the held phone on table, bench, or floor.",
        category=SkillCategory.PHONE,
        preconditions=(
            SkillCondition(SkillConditionType.PHONE_STATE, SkillConditionOperator.EQUALS, "Held", failure_reason="PHONE_NOT_HELD"),
        ),
        parameters=(
            SkillParameter(
                name="target",
                parameter_type=SkillParameterType.TARGET,
                required=True,
                choices=("TABLE", "BENCH", "FLOOR"),
                description="placement surface",
            ),
        ),
        steps=(SkillActionStep(ActionType.PLACE_PHONE, target_parameter="target"),),
        effects=("phone_state = placed",),
        cooldown_seconds=1.0,
        tags=("phone", "interaction"),
    )
    start_stream = Skill(
        skill_id="start_stream",
        name="Start Stream",
        description="Starts the virtual livestream through the existing stream system.",
        category=SkillCategory.LIVESTREAM,
        preconditions=(
            SkillCondition(SkillConditionType.STREAM_STATE, SkillConditionOperator.IS_FALSE, failure_reason="STREAM_ALREADY_ACTIVE"),
        ),
        steps=(SkillActionStep(ActionType.START_STREAM),),
        effects=("stream_started",),
        cooldown_seconds=2.0,
        tags=("livestream",),
    )
    stop_stream = Skill(
        skill_id="stop_stream",
        name="Stop Stream",
        description="Stops the current virtual livestream.",
        category=SkillCategory.LIVESTREAM,
        preconditions=(
            SkillCondition(SkillConditionType.STREAM_STATE, SkillConditionOperator.IS_TRUE, failure_reason="STREAM_NOT_ACTIVE"),
        ),
        steps=(SkillActionStep(ActionType.STOP_STREAM),),
        effects=("stream_stopped",),
        cooldown_seconds=2.0,
        tags=("livestream",),
    )
    speak = Skill(
        skill_id="speak",
        name="Speak",
        description="Speaks provided dialogue through the existing speech action.",
        category=SkillCategory.SOCIAL,
        parameters=(
            SkillParameter(
                name="text",
                parameter_type=SkillParameterType.STRING,
                required=True,
                description="dialogue to speak",
            ),
        ),
        steps=(SkillActionStep(ActionType.SPEAK, text_parameter="text"),),
        effects=("social_engagement +0.1",),
        tags=("social", "speech"),
    )
    show_table = Skill(
        skill_id="show_table",
        name="Show Table",
        description="Walks to the table, looks at it, and introduces it to a viewer.",
        category=SkillCategory.COMPOSITE,
        composition=(
            SkillCompositionStep("walk_to", {"target": "TABLE"}),
            SkillCompositionStep("look_at", {"target": "TABLE"}),
            SkillCompositionStep("speak", {"text": "This is the table."}),
        ),
        effects=("character_position_changed", "social_engagement +0.2"),
        cooldown_seconds=3.0,
        tags=("social", "composite", "navigation"),
    )
    setup_camera_and_dance = Skill(
        skill_id="setup_camera_and_dance",
        name="Set Up Camera and Dance",
        description="Moves to the placed camera stand, looks at the stream camera, and dances.",
        category=SkillCategory.COMPOSITE,
        preconditions=(
            SkillCondition(
                SkillConditionType.PHONE_STATE,
                SkillConditionOperator.EQUALS,
                "Placed",
                failure_reason="PHONE_NOT_PLACED",
            ),
        ),
        composition=(
            SkillCompositionStep("walk_to", {"target": "CAMERA_STAND"}),
            SkillCompositionStep("look_at", {"target": "CAMERA"}, required=False),
            SkillCompositionStep("dance", {}),
        ),
        effects=("character_position_changed", "entertainment +0.3"),
        cooldown_seconds=8.0,
        tags=("social", "composite", "entertainment", "camera"),
    )
    return (
        walk_to,
        stop,
        turn_to,
        look_at,
        wave,
        dance,
        run,
        pick_up_phone,
        place_phone,
        start_stream,
        stop_stream,
        speak,
        show_table,
        setup_camera_and_dance,
    )
