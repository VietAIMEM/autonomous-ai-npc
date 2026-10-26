"""Phase 08 social autonomy subsystem."""

from .attention import AttentionAssessment, SocialAttention, assess_camera_framing
from .context import SocialContext, SocialReaction
from .conversation import ConversationManager
from .models import (
    AttentionDecision,
    ConversationState,
    PendingInteraction,
    SocialConfig,
    SocialDecision,
    SocialSpeakingPolicy,
    TopicState,
    ViewerSocialContext,
)
from .service import SocialAutonomyService

__all__ = [
    "AttentionAssessment",
    "AttentionDecision",
    "ConversationManager",
    "ConversationState",
    "PendingInteraction",
    "SocialAttention",
    "SocialAutonomyService",
    "SocialConfig",
    "SocialContext",
    "SocialDecision",
    "SocialReaction",
    "SocialSpeakingPolicy",
    "TopicState",
    "ViewerSocialContext",
    "assess_camera_framing",
]
