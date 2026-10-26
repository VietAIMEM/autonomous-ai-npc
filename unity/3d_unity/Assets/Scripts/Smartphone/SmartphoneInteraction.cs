using RoomAgent.Interaction;
using RoomAgent.Lacrimosa;
using UnityEngine;

namespace RoomAgent.Smartphone
{
    [DisallowMultipleComponent]
    [RequireComponent(typeof(SmartphoneController))]
    public sealed class SmartphoneInteraction : MonoBehaviour, IInteractable
    {
        [SerializeField] private Transform handAnchorOverride;
        [SerializeField] private float maxInteractionDistance = 1.8f;

        private SmartphoneController controller;

        public bool CanInteract(GameObject actor)
        {
            if (actor == null || GetComponent<SmartphoneController>().CurrentState == SmartphoneState.Held)
                return false;

            var character = actor.GetComponent<LacrimosaController>();
            var flatCharacterPosition = new Vector3(
                actor.transform.position.x,
                0f,
                actor.transform.position.z);
            var flatPhonePosition = new Vector3(
                transform.position.x,
                0f,
                transform.position.z);
            return character != null
                && Vector3.Distance(flatCharacterPosition, flatPhonePosition) <= maxInteractionDistance;
        }

        public bool Interact(GameObject actor)
        {
            if (!CanInteract(actor))
                return false;

            return GetComponent<SmartphoneController>().PickUp(ResolveHandAnchor(actor.transform), actor);
        }

        private Transform ResolveHandAnchor(Transform actor)
        {
            if (handAnchorOverride != null)
                return handAnchorOverride;

            var animator = actor.GetComponentInChildren<Animator>();
            return animator != null ? animator.GetBoneTransform(HumanBodyBones.RightHand) ?? actor.transform : actor.transform;
        }
    }
}
