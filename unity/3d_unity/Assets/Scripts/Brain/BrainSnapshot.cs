using System.Collections.Generic;

namespace RoomAgent.Brain
{
    [System.Serializable]
    public sealed class WorldSnapshot
    {
        public CharacterSnapshot character;
        public PhoneSnapshot phone;
        public StreamSnapshot stream;
        [System.NonSerialized]
        public Dictionary<string, TargetSnapshot> targets;
    }

    [System.Serializable]
    public sealed class CharacterSnapshot
    {
        public List<float> position;
        public List<float> forward;
        public float rotation_y;
        public string posture;
        public string current_action;
        // PHASE 26B: observed gesture evidence. Unity reports what it actually
        // played, so a backend "gesture = WAVE" record alone is never treated
        // as proof that the animation ran.
        public string last_gesture;
        public string last_gesture_at;
        public bool gesture_active;
        // PHASE 28 runtime motion diagnostics. Every value is read back from the
        // live Animator / NavMeshAgent / IK components on this frame, so a
        // requested transition can never be reported as an observed one.
        public string animator_state;
        public string animator_previous_state;
        public string animator_requested_action;
        public bool animator_transition_active;
        public float animator_transition_progress;
        public string animator_transition_target;
        public float animator_state_normalized_time;
        public float animator_base_layer_weight;
        public bool root_motion_apply;
        public string locomotion_owner;
        public float agent_velocity;
        public float agent_desired_speed;
        public bool agent_is_stopped;
        public List<float> agent_destination;
        public string orientation_state;
        public List<float> orientation_target;
        public float orientation_angle_remaining;
        public string rotation_owner;
        public bool look_at_active;
        public List<float> look_at_target;
        public string ik_left_hand_target;
        public string ik_right_hand_target;
        public float ik_left_hand_weight;
        public float ik_right_hand_weight;
        public string motion_reference;
        public string animation_completion_reason;
        public string animation_last_result;
        public string last_action_result;
        public string execution_id;
    }

    [System.Serializable]
    public sealed class PhoneSnapshot
    {
        public string state;
        public List<float> position;
        public string holder;
        public string placement_surface;
    }

    [System.Serializable]
    public sealed class StreamSnapshot
    {
        public bool is_streaming;
        public bool camera_available;
        public bool camera_active;
        public float duration_seconds;
        public int viewer_count;
        public List<float> camera_position;
        public List<float> camera_rotation;
        public List<float> camera_forward;
        public float camera_field_of_view;
    }

    /// <summary>Reply contract for the backend's ``/stream/state`` route.</summary>
    [System.Serializable]
    public sealed class StreamTransportState
    {
        public bool enabled;
        public bool transport_enabled;
        public bool live;
        public string state;
    }

    [System.Serializable]
    public sealed class TargetSnapshot
    {
        public List<float> position;
        public bool navigation_target;
        public bool look_at_target;
        public bool reachable;
    }
}
