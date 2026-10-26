using UnityEngine;

namespace RoomAgent.Interaction
{
    public interface IInteractable
    {
        bool CanInteract(GameObject actor);
        bool Interact(GameObject actor);
    }
}
