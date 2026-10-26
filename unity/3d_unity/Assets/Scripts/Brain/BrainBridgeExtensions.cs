using RoomAgent.Brain;
using RoomAgent.Lacrimosa;

namespace RoomAgent.Brain
{
    public static class BrainBridgeExtensions
    {
        public static LacrimosaMovement Movement(this LacrimosaController controller) =>
            controller != null ? controller.GetComponent<LacrimosaMovement>() : null;
    }
}
