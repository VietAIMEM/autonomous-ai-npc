"""Memory consolidation, retrieval, and failure-safe service boundary."""

from __future__ import annotations

import json
import logging
from typing import Any, Callable, Iterable, Optional

from ..chat.models import ChatMessage, DialogueResponse
from ..brain.models import ActionResult, ActionType, Goal, PersonalityProfile
from ..brain.personalization import PersonalizedContext, build_personalized_context
from ..brain.personality import (
    PERSONALITY_METADATA_KEY,
    load_personality,
    personality_descriptor,
    personality_from_payload,
    personality_speech_style,
    personality_traits,
)
from ..runtime import instrument
from ..runtime.redaction import safe_error
from .models import (
    MemoryConfig,
    MemoryQuery,
    MemoryType,
    RelationshipLevel,
    RetrievedMemory,
    ViewerMemory,
    Episode,
    InteractionRecord,
    SemanticMemory,
    coerce_relationship_level,
    utc_now,
)
from .repository import MemoryRepository
from .short_term import ShortTermMemory


logger = logging.getLogger(__name__)


def operator_viewer_summary(viewer: ViewerMemory) -> dict[str, Any]:
    """Aggregate, operator-safe view of one viewer (PHASE 23).

    Relationship level, familiarity and learned preferences are operational
    facts; raw interaction text is deliberately not part of this projection.
    """
    level = viewer.relationship_level
    return {
        "viewer_id": viewer.viewer_id,
        "interaction_count": int(viewer.interaction_count),
        "successful_interactions": int(viewer.successful_interactions),
        "failed_interactions": int(viewer.failed_interactions),
        "relationship_level": (
            level.value if isinstance(level, RelationshipLevel) else str(level)
        ),
        "relationship_score": round(float(viewer.relationship_score), 4),
        "engagement_score": round(float(viewer.engagement_score), 4),
        "familiarity": round(float(viewer.familiarity), 4),
        "last_interaction": viewer.last_interaction.isoformat()
        if viewer.last_interaction
        else None,
        "preferences": list(viewer.preferences)[:12],
        "preferred_skills": list(viewer.preferred_skills)[:12],
        "preferred_topics": list(viewer.preferred_topics)[:12],
        "recent_topics": list(viewer.recent_topics)[:12],
    }


#: PHASE 19 bounded relationship thresholds. Each entry must be satisfied in
#: full, and at most one level is gained per recorded interaction, so a single
#: message can never jump a viewer several levels.
_RELATIONSHIP_RULES: tuple[tuple[RelationshipLevel, dict[str, float]], ...] = (
    (
        RelationshipLevel.FAMILIAR,
        {"interactions": 3, "successes": 2, "familiarity": 0.10, "score": 0.0, "engagement": 0.0},
    ),
    (
        RelationshipLevel.REGULAR,
        {"interactions": 8, "successes": 6, "familiarity": 0.30, "score": 0.40, "engagement": 0.55},
    ),
    (
        RelationshipLevel.TRUSTED,
        {"interactions": 20, "successes": 17, "familiarity": 0.75, "score": 0.70, "engagement": 0.75},
    ),
)

#: Topics and skills only become preferences after repeat evidence.
_PREFERENCE_EVIDENCE_THRESHOLD = 2
_MAX_PREFERRED_ENTRIES = 5


class MemoryService:
    """Backend-owned memory facade; never gives repository control to the LLM."""

    def __init__(
        self,
        repository: MemoryRepository,
        config: MemoryConfig | None = None,
        clock: Callable[[], object] = utc_now,
    ) -> None:
        self.config = config or MemoryConfig()
        self.repository = repository
        self.short_term = ShortTermMemory(self.config)
        self._clock = clock
        self._degraded = False
        self._last_error: str | None = None
        self._last_query: MemoryQuery | None = None
        self._last_retrieval_count = 0

    def remember_chat(
        self,
        message: ChatMessage,
        response: DialogueResponse | None = None,
    ) -> None:
        try:
            self.short_term.add_message(message, response)
        except Exception as exc:
            self._mark_degraded(exc, "short_term.add_message")

    def remember_event(
        self,
        text: str,
        *,
        importance: float = 0.2,
        viewer_id: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        try:
            self.short_term.add_event(
                text,
                importance=importance,
                viewer_id=viewer_id,
                metadata=metadata,
            )
        except Exception as exc:
            self._mark_degraded(exc, "short_term.add_event")

    def observe_context(
        self,
        *,
        goal: str | None = None,
        action: str | None = None,
        world_context: dict[str, Any] | None = None,
    ) -> None:
        try:
            if goal is not None:
                self.short_term.set_current_goal(goal)
            if action is not None:
                self.short_term.set_current_action(action)
            if world_context is not None:
                self.short_term.set_world_context(world_context)
        except Exception as exc:
            self._mark_degraded(exc, "short_term.observe_context")

    def record_outcome(
        self,
        *,
        viewer_id: str | None,
        chat: ChatMessage | None,
        goal: Goal | None,
        action: ActionType | str | None,
        success: bool,
        importance: float | None = None,
        response: str | None = None,
        context: dict[str, Any] | None = None,
        skill_id: str | None = None,
    ) -> None:
        action_name = action.value if isinstance(action, ActionType) else action
        action_lower = action_name.lower() if action_name else "action"
        outcome_importance = self._importance(chat, success) if importance is None else importance
        outcome_importance = max(0.0, min(1.0, outcome_importance))
        try:
            self.short_term.add_action(
                f"{action_name or 'action'} {'succeeded' if success else 'failed'}",
                importance=outcome_importance,
                viewer_id=viewer_id,
                metadata={"result": "SUCCESS" if success else "FAILED"},
            )
        except Exception as exc:
            self._mark_degraded(exc, "short_term.add_action")

        if outcome_importance < self.config.importance_threshold:
            return

        try:
            episode_id = self.repository.next_id("episode")
            interaction_id = self.repository.next_id("interaction")
        except Exception as exc:
            self._mark_degraded(exc, "next_id")
            return

        now = self._clock()
        tags = self._tags(action_name, chat, skill_id)
        episode = Episode(
            episode_id=episode_id,
            timestamp=now,
            event=chat.raw_text if chat else f"Lacrimosa completed {action_lower}",
            context=self._episode_context(chat, goal, context),
            action=action_name,
            result="SUCCESS" if success else "FAILED",
            importance=outcome_importance,
            viewer_id=viewer_id,
            goal=goal.goal_type.value if goal else None,
            location=self._location(context),
            tags=tags,
        )
        self._save_episode(episode)

        record = InteractionRecord(
            interaction_id=interaction_id,
            timestamp=now,
            viewer_id=viewer_id,
            message_id=chat.message_id if chat else None,
            intent=chat.intent.value if chat else None,
            requested_action=chat.target if chat else None,
            action=action_name,
            response=response,
            result="SUCCESS" if success else "FAILED",
            importance=outcome_importance,
        )
        self._save_interaction(record)

        if viewer_id:
            self._derive_semantic_memory(
                viewer_id=viewer_id,
                action=action_lower,
                success=success,
                importance=outcome_importance,
            )
            self._update_viewer_memory(
                viewer_id=viewer_id,
                success=success,
                topic=chat.target if chat and chat.target else action_lower,
                preference=action_lower,
                skill_id=skill_id,
            )

    def retrieve(self, query: MemoryQuery) -> list[RetrievedMemory]:
        self._last_query = query
        try:
            viewer_memories: list[ViewerMemory] = []
            if query.viewer_id:
                viewer_memory = self.repository.get_viewer_memory(query.viewer_id)
                if viewer_memory is not None:
                    viewer_memories = [viewer_memory]

            episodes = self.repository.search_episodes(
                viewer_id=query.viewer_id,
                limit=100,
            )
            semantic = self.repository.search_semantic_memory(
                subject=query.viewer_id,
                limit=100,
            )
            if not query.viewer_id:
                episodes = self.repository.search_episodes(limit=100)
                semantic = self.repository.search_semantic_memory(limit=100)

            candidates: list[RetrievedMemory] = []
            candidates.extend(self._episode_candidates(episodes))
            candidates.extend(self._semantic_candidates(semantic))
            candidates.extend(self._viewer_candidates(viewer_memories))
            ranked = sorted(
                enumerate(candidates),
                key=lambda indexed: (-indexed[1].score, indexed[0]),
            )
            ranked = [item for _, item in ranked]
            result = ranked[: self.config.retrieval_top_k]
            self._last_retrieval_count = len(result)
            logger.info(
                "MEMORY: retrieved %d relevant memories%s",
                len(result),
                f" for {query.viewer_id}" if query.viewer_id else "",
            )
            return result
        except Exception as exc:
            self._mark_degraded(exc, "retrieve")
            self._last_retrieval_count = 0
            return []

    def get_state(self) -> dict[str, Any]:
        try:
            counts = self.repository.counts()
            last_episode = self.repository.last_episode()
            degraded = False
            error = None
        except Exception as exc:
            self._mark_degraded(exc, "get_state")
            counts = {}
            last_episode = None
            degraded = True
            error = self._last_error
        return {
            "schema_version": 2,
            "degraded": degraded or self._degraded,
            "last_error": error or self._last_error,
            "counts": counts,
            "short_term": self.short_term.snapshot(),
            "last_episode": last_episode.to_dict() if last_episode else None,
            "last_query": {
                "viewer_id": self._last_query.viewer_id if self._last_query else None,
                "goal": self._last_query.goal if self._last_query else None,
                "intent": self._last_query.intent if self._last_query else None,
                "action": self._last_query.action if self._last_query else None,
                "text": self._last_query.text if self._last_query else None,
            },
            "last_retrieval_count": self._last_retrieval_count,
        }

    def get_viewer(self, viewer_id: str) -> dict[str, Any]:
        try:
            viewer = self.repository.get_viewer_memory(viewer_id)
            interactions = self.repository.search_interactions(viewer_id=viewer_id, limit=20)
            viewer_payload = viewer.to_dict() if viewer else None
            personalization = self.personalize(
                viewer_id,
                viewer_state=viewer_payload,
                personality=self.load_personality_state(),
            )
            return {
                "viewer": viewer_payload,
                "recent_interactions": [record.to_dict() for record in interactions],
                "personalization": personalization.to_dict(),
            }
        except Exception as exc:
            self._mark_degraded(exc, "get_viewer")
            return {
                "viewer": None,
                "recent_interactions": [],
                "personalization": build_personalized_context(
                    viewer_id,
                    None,
                    greeting_allowed=False,
                ).to_dict(),
                "degraded": True,
            }

    def list_viewers(self, limit: int = 50) -> dict[str, Any]:
        """Bounded operational viewer list for the PHASE 23 admin console.

        Only relationship-level, aggregate fields leave this method; raw
        interaction text stays behind :meth:`get_viewer`.
        """
        bounded = max(1, min(int(limit), 200))
        try:
            viewers = self.repository.list_viewer_memories(bounded)
        except Exception as exc:
            self._mark_degraded(exc, "list_viewers")
            return {
                "viewers": [],
                "count": 0,
                "degraded": True,
                "last_error": self._last_error,
            }
        return {
            "viewers": [operator_viewer_summary(viewer) for viewer in viewers],
            "count": len(viewers),
            "limit": bounded,
            "degraded": self._degraded,
        }

    def personalize(
        self,
        viewer_id: str | None,
        *,
        viewer_state: dict[str, Any] | None = None,
        message_text: str | None = None,
        intent: str | None = None,
        greeting_allowed: bool = True,
        personality: PersonalityProfile | None = None,
    ) -> PersonalizedContext:
        """One bounded personalization block for a real decision.

        The viewer state is read from the existing viewer memory table unless
        the caller already has it. Any retrieval failure degrades to generic
        context; it never fabricates history and never stops the Brain.
        """
        if viewer_state is None and viewer_id:
            try:
                viewer = self.repository.get_viewer_memory(viewer_id)
                viewer_state = viewer.to_dict() if viewer else None
            except Exception as exc:
                self._mark_degraded(exc, "personalize")
                viewer_state = None
        if personality is None:
            personality = self.load_personality_state()
        return build_personalized_context(
            viewer_id,
            viewer_state,
            message_text=message_text,
            intent=intent,
            greeting_allowed=greeting_allowed,
            personality=personality,
        )

    def load_personality_state(
        self,
        overrides: dict[str, float] | None = None,
    ) -> PersonalityProfile:
        """Load the persisted stable profile, falling back to configuration.

        A missing row, corrupt payload or unavailable repository returns the
        configured profile; a broken store never invents personality drift.
        """
        try:
            raw = self.repository.get_metadata(PERSONALITY_METADATA_KEY)
        except Exception:
            raw = None
        if isinstance(raw, str) and raw.strip():
            try:
                payload = json.loads(raw)
            except (TypeError, ValueError):
                payload = None
            profile = personality_from_payload(payload)
            if profile is not None:
                if overrides:
                    merged = personality_traits(profile)
                    merged.update(overrides)
                    return load_personality(merged)
                return profile
        return load_personality(overrides)

    def save_personality_state(self, profile: PersonalityProfile) -> bool:
        """Persist the stable profile through the existing metadata table."""
        payload = {"traits": personality_traits(profile)}
        try:
            self.repository.set_metadata(
                PERSONALITY_METADATA_KEY,
                json.dumps(payload, sort_keys=True),
            )
            instrument.count("memory.personality_writes")
            return True
        except Exception as exc:
            self._mark_degraded(exc, "save_personality_state")
            return False

    def set_personality(
        self,
        profile: PersonalityProfile | None = None,
        *,
        overrides: dict[str, float] | None = None,
    ) -> PersonalityProfile:
        """Explicit, one-shot personality configuration update."""
        resolved = profile or load_personality(overrides)
        self.save_personality_state(resolved)
        return resolved

    def personality_diagnostics(self) -> dict[str, Any]:
        """Inspectable, non-sensitive view of the personality layer."""
        profile = self.load_personality_state()
        return {
            "profile": personality_traits(profile),
            "speech_style": personality_speech_style(profile),
            "descriptor": personality_descriptor(profile),
            "source": "persisted" if self._personality_persisted() else "configuration",
            "rule": "stable configuration; never drifts from runtime events",
        }

    def _personality_persisted(self) -> bool:
        """True only when a decodable profile is actually stored.

        A corrupt row still loads the configured fallback profile, so
        diagnostics must not claim a persisted personality that could not be
        decoded.
        """
        try:
            raw = self.repository.get_metadata(PERSONALITY_METADATA_KEY)
        except Exception:
            return False
        if not isinstance(raw, str) or not raw.strip():
            return False
        try:
            payload = json.loads(raw)
        except (TypeError, ValueError):
            return False
        return personality_from_payload(payload) is not None

    def close(self) -> None:
        try:
            self.repository.close()
        except Exception as exc:
            self._mark_degraded(exc, "close")

    def _save_episode(self, episode: Episode) -> None:
        try:
            self.repository.save_episode(episode)
            instrument.count("memory.episode_writes")
            logger.info(
                "MEMORY: stored episode %s importance=%.2f",
                episode.episode_id,
                episode.importance,
            )
        except Exception as exc:
            self._mark_degraded(exc, "save_episode")

    def _save_interaction(self, record: InteractionRecord) -> None:
        try:
            self.repository.save_interaction(record)
            instrument.count("memory.interaction_writes")
        except Exception as exc:
            self._mark_degraded(exc, "save_interaction")

    def _derive_semantic_memory(
        self,
        *,
        viewer_id: str,
        action: str,
        success: bool,
        importance: float,
    ) -> None:
        try:
            episodes = self.repository.search_episodes(
                viewer_id=viewer_id,
                topic=action,
                limit=100,
            )
            same_action = [
                episode
                for episode in episodes
                if episode.action and episode.action.lower() == action
            ]
            enough_evidence = len(same_action) >= self.config.semantic_evidence_threshold
            strong_single_evidence = importance >= 0.85
            if not (enough_evidence or strong_single_evidence):
                return

            predicate = "likes" if success else "dislikes"
            existing = self.repository.find_semantic_memory(viewer_id, predicate, action)
            if existing is not None:
                confidence = 1.0 - (1.0 - existing.confidence) * 0.9
                updated = SemanticMemory(
                    memory_id=existing.memory_id,
                    subject=existing.subject,
                    predicate=existing.predicate,
                    value=existing.value,
                    confidence=confidence,
                    importance=max(existing.importance, 0.65),
                    source=existing.source,
                    evidence_count=min(existing.evidence_count + 1, 1000),
                    created_at=existing.created_at,
                    updated_at=self._clock(),
                )
                self.repository.save_semantic_memory(updated)
                instrument.count("memory.semantic_writes")
                logger.info(
                    "MEMORY: updated semantic memory %s evidence=%d",
                    updated.memory_id,
                    updated.evidence_count,
                )
                return

            conflicting = self.repository.find_semantic_memory(viewer_id, predicate, None)
            if conflicting is not None:
                # Supporting evidence updates the matching value above. A different
                # value is allowed as an additional preference; direct predicate
                # conflicts are not blindly overwritten.
                logger.info(
                    "MEMORY: kept existing semantic memory %s",
                    conflicting.memory_id,
                )

            if predicate == "dislikes":
                opposing = self.repository.find_semantic_memory(viewer_id, "likes", action)
                if opposing is not None and opposing.evidence_count >= 1:
                    logger.info(
                        "MEMORY: semantic conflict for %s %s; kept higher-evidence record %s",
                        viewer_id,
                        action,
                        opposing.memory_id,
                    )
                    return

            memory = SemanticMemory(
                memory_id=self.repository.next_id("semantic"),
                subject=viewer_id,
                predicate=predicate,
                value=action,
                confidence=0.75 if strong_single_evidence else self.config.semantic_confidence_threshold,
                importance=0.65,
                source="interaction_outcome",
                evidence_count=len(same_action),
                created_at=self._clock(),
                updated_at=self._clock(),
            )
            self.repository.save_semantic_memory(memory)
            instrument.count("memory.semantic_writes")
        except Exception as exc:
            self._mark_degraded(exc, "derive_semantic_memory")

    def _update_viewer_memory(
        self,
        *,
        viewer_id: str,
        success: bool,
        topic: str,
        preference: str,
        skill_id: str | None = None,
    ) -> None:
        try:
            existing = self.repository.get_viewer_memory(viewer_id)
            if existing is None:
                existing = ViewerMemory(viewer_id=viewer_id)
            preferences = list(existing.preferences)
            if preference not in preferences:
                preferences.append(preference)
            preferences = preferences[-5:]
            topics = [value for value in existing.recent_topics if value != topic]
            topics.append(topic)
            topics = topics[-5:]
            delta = 0.05 if success else -0.05
            interaction_count = existing.interaction_count + 1
            successful_interactions = existing.successful_interactions + (1 if success else 0)
            failed_interactions = existing.failed_interactions + (0 if success else 1)
            relationship_score = max(0.0, min(1.0, existing.relationship_score + delta))
            # Familiarity accumulates with every exchange and faster with
            # successful ones; engagement is a bounded running average of
            # outcomes. Both are evidence, never a single-event jump.
            familiarity = min(1.0, existing.familiarity + 0.03 + (0.02 if success else 0.0))
            engagement = max(
                0.0,
                min(1.0, existing.engagement_score * 0.75 + (0.25 if success else 0.0)),
            )
            preferred_topics = self._preferred_topics(viewer_id, existing)
            preferred_skills = self._preferred_skills(viewer_id, existing, skill_id)
            level = self._progress_relationship(
                existing.relationship_level,
                interactions=interaction_count,
                successes=successful_interactions,
                familiarity=familiarity,
                score=relationship_score,
                engagement=engagement,
            )
            updated = ViewerMemory(
                viewer_id=viewer_id,
                interaction_count=interaction_count,
                successful_interactions=successful_interactions,
                failed_interactions=failed_interactions,
                relationship_score=relationship_score,
                last_interaction=self._clock(),
                preferences=tuple(preferences),
                recent_topics=tuple(topics),
                relationship_level=level.value,
                engagement_score=engagement,
                familiarity=familiarity,
                preferred_skills=preferred_skills,
                preferred_topics=preferred_topics,
            )
            self.repository.save_viewer_memory(updated)
            instrument.count("memory.viewer_writes")
            logger.info(
                "MEMORY: updated %s interaction_count=%d level=%s",
                viewer_id,
                updated.interaction_count,
                updated.relationship_level,
            )
        except Exception as exc:
            self._mark_degraded(exc, "update_viewer_memory")

    def _progress_relationship(
        self,
        current: str | RelationshipLevel,
        *,
        interactions: int,
        successes: int,
        familiarity: float,
        score: float,
        engagement: float,
    ) -> RelationshipLevel:
        """At most one promotion per interaction, only on full evidence."""
        level = coerce_relationship_level(current)
        rule = next((item for item in _RELATIONSHIP_RULES if item[0].rank == level.rank + 1), None)
        if rule is None:
            return level
        candidate, threshold = rule
        satisfied = (
            interactions >= threshold["interactions"]
            and successes >= threshold["successes"]
            and familiarity >= threshold["familiarity"]
            and score >= threshold["score"]
            and engagement >= threshold["engagement"]
        )
        return candidate if satisfied else level

    def _preferred_topics(
        self,
        viewer_id: str,
        existing: ViewerMemory,
    ) -> tuple[str, ...]:
        """Topics with repeat evidence, derived from the existing interactions table."""
        try:
            counts: dict[str, int] = {}
            for record in self.repository.search_interactions(viewer_id=viewer_id, limit=200):
                value = record.requested_action or record.action
                if not isinstance(value, str) or not value.strip():
                    continue
                normalized = value.strip().lower()
                counts[normalized] = counts.get(normalized, 0) + 1
            repeated = [
                value
                for value, count in sorted(counts.items(), key=lambda item: (-item[1], item[0]))
                if count >= _PREFERENCE_EVIDENCE_THRESHOLD
            ]
            merged = list(dict.fromkeys([*repeated, *existing.preferred_topics]))
            return tuple(merged[:_MAX_PREFERRED_ENTRIES])
        except Exception as exc:
            self._mark_degraded(exc, "preferred_topics")
            return existing.preferred_topics

    def _preferred_skills(
        self,
        viewer_id: str,
        existing: ViewerMemory,
        skill_id: str | None,
    ) -> tuple[str, ...]:
        """Skill preferences require repeated *successful* episodes for that viewer."""
        try:
            if not skill_id:
                return existing.preferred_skills
            candidate = skill_id.strip().lower()
            if not candidate:
                return existing.preferred_skills
            counts: dict[str, int] = {}
            for episode in self.repository.search_episodes(viewer_id=viewer_id, limit=100):
                if episode.result != "SUCCESS" or episode.viewer_id != viewer_id:
                    continue
                for tag in episode.tags:
                    normalized = tag.strip().lower()
                    if normalized:
                        counts[normalized] = counts.get(normalized, 0) + 1
            repeated = [
                value
                for value, count in sorted(counts.items(), key=lambda item: (-item[1], item[0]))
                if count >= _PREFERENCE_EVIDENCE_THRESHOLD
            ]
            # One successful use stays weak evidence; only repeats are promoted.
            merged = list(dict.fromkeys([*repeated, *existing.preferred_skills]))
            return tuple(merged[:_MAX_PREFERRED_ENTRIES])
        except Exception as exc:
            self._mark_degraded(exc, "preferred_skills")
            return existing.preferred_skills

    def _episode_candidates(self, episodes: Iterable[Episode]) -> list[RetrievedMemory]:
        now = self._clock()
        candidates = []
        for episode in episodes:
            terms = set(self._memory_terms(episode.event, episode.goal, episode.action))
            candidates.append(
                RetrievedMemory(
                    memory_type=MemoryType.EPISODE,
                    memory_id=episode.episode_id,
                    summary=self._summary(episode.event, episode.action, episode.result),
                    score=self._score(
                        viewer_id=episode.viewer_id,
                        terms=terms,
                        goal=episode.goal,
                        timestamp=episode.timestamp,
                        importance=episode.importance,
                        now=now,
                    ),
                    importance=episode.importance,
                    timestamp=episode.timestamp,
                    viewer_id=episode.viewer_id,
                )
            )
        return candidates

    def _semantic_candidates(
        self,
        memories: Iterable[SemanticMemory],
    ) -> list[RetrievedMemory]:
        now = self._clock()
        candidates = []
        for memory in memories:
            terms = set(self._memory_terms(memory.subject, memory.predicate, memory.value))
            timestamp = memory.updated_at or memory.created_at or now
            candidates.append(
                RetrievedMemory(
                    memory_type=MemoryType.SEMANTIC,
                    memory_id=memory.memory_id,
                    summary=memory.statement,
                    score=self._score(
                        viewer_id=memory.subject,
                        terms=terms,
                        goal=None,
                        timestamp=timestamp,
                        importance=memory.importance,
                        now=now,
                    ),
                    importance=memory.importance,
                    timestamp=timestamp,
                    viewer_id=memory.subject,
                )
            )
        return candidates

    def _viewer_candidates(
        self,
        memories: Iterable[ViewerMemory],
    ) -> list[RetrievedMemory]:
        now = self._clock()
        candidates = []
        for memory in memories:
            terms = set(self._memory_terms(*memory.preferences, *memory.recent_topics))
            timestamp = memory.last_interaction or now
            summary = (
                f"interaction_count={memory.interaction_count}; "
                f"relationship_score={memory.relationship_score:.2f}; "
                f"preferences={', '.join(memory.preferences) or 'none'}; "
                f"recent_topics={', '.join(memory.recent_topics) or 'none'}"
            )
            candidates.append(
                RetrievedMemory(
                    memory_type=MemoryType.VIEWER,
                    memory_id=memory.viewer_id,
                    summary=summary,
                    score=self._score(
                        viewer_id=memory.viewer_id,
                        terms=terms,
                        goal=None,
                        timestamp=timestamp,
                        importance=0.6,
                        now=now,
                    ),
                    importance=0.6,
                    timestamp=timestamp,
                    viewer_id=memory.viewer_id,
                )
            )
        return candidates

    def _score(
        self,
        *,
        viewer_id: str | None,
        terms: set[str],
        goal: str | None,
        timestamp,
        importance: float,
        now,
    ) -> float:
        query = self._last_query
        if query is None:
            return max(0.0, min(1.0, importance))

        viewer_match = 0.45 if query.viewer_id and viewer_id == query.viewer_id else 0.0
        query_terms = set(query.topic_terms())
        topic_match = 0.35 if query_terms and terms.intersection(query_terms) else 0.0
        goal_match = 0.1 if query.goal and goal and query.goal.lower() == goal.lower() else 0.0
        elapsed = max(0.0, (now - timestamp).total_seconds())
        recency = 0.15 * max(0.0, 1.0 - elapsed / self.config.recency_window_seconds)
        return max(0.0, min(1.0, viewer_match + topic_match + goal_match + recency + 0.2 * importance))

    @staticmethod
    def _memory_terms(*values: str | None) -> list[str]:
        terms: list[str] = []
        for value in values:
            if not value:
                continue
            terms.extend(part for part in value.lower().replace("_", " ").split() if len(part) > 2)
        return terms

    def _summary(self, event: str, action: str | None, result: str) -> str:
        value = event if len(event) <= self.config.max_summary_length else event[: self.config.max_summary_length]
        return f"{value}; action={action or 'none'}; result={result}"

    @staticmethod
    def _episode_context(
        chat: ChatMessage | None,
        goal: Goal | None,
        context: dict[str, Any] | None,
    ) -> dict[str, Any]:
        value = dict(context or {})
        if chat:
            value.update(
                {
                    "message_id": chat.message_id,
                    "intent": chat.intent.value,
                    "requested_action": chat.target,
                }
            )
        if goal:
            value["goal"] = goal.goal_type.value
        return value

    @staticmethod
    def _location(context: dict[str, Any] | None) -> str | None:
        value = (context or {}).get("location")
        return value if isinstance(value, str) and value else None

    @staticmethod
    def _tags(
        action: str | None,
        chat: ChatMessage | None,
        skill_id: str | None = None,
    ) -> tuple[str, ...]:
        values = []
        if action:
            values.append(action.lower())
        if chat and chat.target:
            values.append(chat.target.lower())
        if skill_id and skill_id.strip():
            values.append(skill_id.strip().lower())
        return tuple(dict.fromkeys(values))

    def _importance(self, chat: ChatMessage | None, success: bool) -> float:
        if chat is None:
            return 0.65 if success else 0.45
        base = 0.72 if chat.intent.value.startswith("REQUEST") else 0.6
        return base if success else max(0.4, base - 0.15)

    def _mark_degraded(self, exc: Exception, operation: str) -> None:
        self._degraded = True
        # Redacted and bounded: this string is surfaced by ``get_state`` and by
        # the unified diagnostics payload, so it must never carry key material.
        self._last_error = f"{operation}:{safe_error(exc, limit=200)}"
        instrument.count("memory.degraded")
        instrument.record_error("memory", f"{operation}:{type(exc).__name__}")
        logger.error("MEMORY: %s failed; degraded mode enabled", operation)
