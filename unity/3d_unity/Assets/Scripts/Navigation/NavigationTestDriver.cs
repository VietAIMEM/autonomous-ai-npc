using UnityEngine;
using RoomAgent.Navigation;

namespace RoomAgent.Navigation
{
    /// <summary>
    /// PHASE 01 validation harness only. It walks a list of waypoints so navigation can be
    /// exercised before the backend action system exists, and optionally moves on mouse clicks.
    /// Later phases replace this with MoveTo actions issued by the AI brain.
    /// </summary>
    [DisallowMultipleComponent]
    public sealed class NavigationTestDriver : MonoBehaviour
    {
        [SerializeField] NavMeshWalker walker;
        [SerializeField] Transform[] waypoints;
        [SerializeField] bool autoCycle = true;
        [SerializeField] float dwellSeconds = 1.5f;
        [SerializeField] float stoppingDistance = 0.3f;
        [SerializeField] bool clickToMove = true;
        [SerializeField] LayerMask clickLayerMask = ~0;

        int nextIndex;
        float dwellTimer;
        string label = "idle";

        /// <summary>What the harness last asked the walker to do. Handy for the Inspector and tests.</summary>
        public string CurrentCommand => label;

        void Awake()
        {
            if (walker == null)
                walker = GetComponent<NavMeshWalker>();
        }

        void OnEnable()
        {
            nextIndex = 0;
            dwellTimer = 0f;
        }

        void Update()
        {
            if (walker == null)
                return;

            if (autoCycle && waypoints != null && waypoints.Length > 0)
                RunCycle();

            HandleClick();
        }

        void RunCycle()
        {
            if (!walker.IsReady)
                return;

            if (walker.IsMoving)
            {
                dwellTimer = 0f;
                return;
            }

            dwellTimer += Time.deltaTime;
            if (dwellTimer < dwellSeconds)
                return;

            dwellTimer = 0f;
            var target = waypoints[nextIndex % waypoints.Length];
            var name = target != null ? target.name : "null";
            if (target != null && walker.MoveTo(target.position, stoppingDistance))
                label = "moving to " + name;
            nextIndex = (nextIndex + 1) % waypoints.Length;
        }

        void HandleClick()
        {
            if (!clickToMove)
                return;

#if ENABLE_INPUT_SYSTEM
            var mouse = UnityEngine.InputSystem.Mouse.current;
            if (mouse == null || !mouse.leftButton.wasPressedThisFrame)
                return;
            var screenPosition = mouse.position.ReadValue();
#else
            if (!Input.GetMouseButtonDown(0))
                return;
            var screenPosition = Input.mousePosition;
#endif

            var camera = Camera.main;
            if (camera == null)
                return;

            var ray = camera.ScreenPointToRay(screenPosition);
            if (!Physics.Raycast(ray, out var hit, 200f, clickLayerMask, QueryTriggerInteraction.Ignore))
                return;

            if (walker.MoveTo(hit.point, stoppingDistance))
                label = "moving to click " + hit.point;
        }

        void OnValidate()
        {
            if (walker == null)
                walker = GetComponent<NavMeshWalker>();
        }
    }
}
