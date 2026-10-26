using UnityEngine;

namespace RoomAgent.Interaction
{
    [DisallowMultipleComponent]
    public sealed class InteractionSurface : MonoBehaviour
    {
        [SerializeField] private bool acceptsPlacement = true;

        public bool AcceptsPlacement => acceptsPlacement;
    }
}
