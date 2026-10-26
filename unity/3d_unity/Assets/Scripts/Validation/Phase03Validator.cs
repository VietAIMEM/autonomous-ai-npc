using System.Collections;
using System.Text;
using UnityEngine;
using RoomAgent.Lacrimosa;
using RoomAgent.Smartphone;

namespace RoomAgent.Validation
{
    public sealed class Phase03Validator : MonoBehaviour
    {
        private const int MoveFrameLimit = 3000;
        private const int TurnFrameLimit = 1000;

        private StringBuilder log;
        private bool running;
        private LacrimosaMovement movement;

        public bool IsRunning => running;
        public string Report { get; private set; } = string.Empty;

        public void Run()
        {
            if (!running)
                StartCoroutine(Validate());
        }

        private IEnumerator Validate()
        {
            running = true;
            log = new StringBuilder();
            var character = FindObjectOfType<LacrimosaController>();
            var phone = FindObjectOfType<SmartphoneController>();
            movement = character != null ? character.GetComponent<LacrimosaMovement>() : null;

            if (phone == null || character == null)
            {
                Report = "FAIL: missing smartphone or character";
                running = false;
                yield break;
            }

            var tableTop = GameObject.Find("Environment/Table/Top");
            var benchSeat = GameObject.Find("Environment/Bench/Seat");
            if (tableTop == null || benchSeat == null)
            {
                Report = "FAIL: missing table or bench surface";
                running = false;
                yield break;
            }

            var tablePoint = new Vector3(-3.1f, tableTop.transform.position.y, 0f);
            var benchPoint = new Vector3(-4.7f, benchSeat.transform.position.y, 0.0f);
            var floorPoint = new Vector3(0f, -0.05f, 0f);

            Log("=== PHASE 03 SMARTPHONE VALIDATION ===");

            // Sequence 1: table pickup, carry, turn, return, place back on table.
            ResetPhone(phone, tablePoint);
            yield return MoveTo(character, new Vector3(-1.8f, 0f, 0f));
            yield return TryPickUp(phone, character, "table");
            yield return MoveTo(character, Vector3.zero);
            yield return TurnTo(character, tablePoint);
            yield return MoveTo(character, new Vector3(-1.8f, 0f, 0f));
            yield return TryPlace(phone, character, tablePoint, "table");

            // Sequence 2: bench pickup, carry away, return, place on bench.
            ResetPhone(phone, benchPoint);
            yield return MoveTo(character, new Vector3(-3.8f, 0f, -1.0f));
            yield return TryPickUp(phone, character, "bench");
            yield return MoveTo(character, Vector3.zero);
            yield return MoveTo(character, new Vector3(-3.8f, 0f, -1.0f));
            yield return TryPlace(phone, character, benchPoint, "bench");

            // Sequence 3: floor pickup, carry away, return, place on floor.
            ResetPhone(phone, floorPoint);
            yield return MoveTo(character, new Vector3(0.8f, 0f, 0f));
            yield return TryPickUp(phone, character, "floor");
            yield return MoveTo(character, new Vector3(2.0f, 0f, 0f));
            yield return MoveTo(character, new Vector3(0.8f, 0f, 0f));
            yield return TryPlace(phone, character, floorPoint, "floor");

            // Sequence 4: repeated transfer across all surfaces using the same instance.
            yield return RepeatedTransfer(phone, character, tablePoint, benchPoint, floorPoint);

            Report = log.ToString();
            running = false;
        }

        private IEnumerator RepeatedTransfer(
            SmartphoneController phone,
            LacrimosaController character,
            Vector3 tablePoint,
            Vector3 benchPoint,
            Vector3 floorPoint)
        {
            Log("--- REPEATED TRANSFER ---");
            for (var i = 0; i < 2; i++)
            {
                yield return MoveTo(character, new Vector3(0.8f, 0f, 0f));
                yield return TryPickUp(phone, character, "floor");
                yield return MoveTo(character, new Vector3(-1.8f, 0f, 0f));
                yield return TryPlace(phone, character, tablePoint, "table");

                yield return MoveTo(character, new Vector3(-1.8f, 0f, 0f));
                yield return TryPickUp(phone, character, "table");
                yield return MoveTo(character, Vector3.zero);
                yield return MoveTo(character, new Vector3(-3.8f, 0f, -1.0f));
                yield return TryPlace(phone, character, benchPoint, "bench");

                yield return MoveTo(character, new Vector3(-3.8f, 0f, -1.0f));
                yield return TryPickUp(phone, character, "bench");
                yield return MoveTo(character, Vector3.zero);
                yield return TryPlace(phone, character, floorPoint, "floor");
            }
        }

        private IEnumerator MoveTo(LacrimosaController character, Vector3 target)
        {
            if (!character.MoveTo(target))
            {
                Log("FAIL: MoveTo rejected target " + target.ToString("F2"));
                yield break;
            }

            var agent = character.GetComponent<LacrimosaMovement>().Agent;
            var frames = 0;
            while (agent.pathPending && frames < MoveFrameLimit)
            {
                frames++;
                yield return null;
            }

            while (movement.IsMoving && frames < MoveFrameLimit)
            {
                frames++;
                yield return null;
            }

            var reached = Vector3.Distance(character.transform.position, target) < 0.9f;
            Log("MoveTo " + target.ToString("F2") + " frames=" + frames + " reached=" + reached + " pos=" + character.transform.position.ToString("F3"));
        }

        private IEnumerator TurnTo(LacrimosaController character, Vector3 target)
        {
            character.TurnTo(target);
            var frames = 0;
            while (character.IsTurning && frames < TurnFrameLimit)
            {
                frames++;
                yield return null;
            }

            Log("TurnTo " + target.ToString("F2") + " frames=" + frames + " yaw=" + character.transform.eulerAngles.y.ToString("F1"));
        }

        private IEnumerator TryPickUp(SmartphoneController phone, LacrimosaController character, string label)
        {
            var pickedUp = character.PickUp(phone);
            Log("PickUp[" + label + "]=" + pickedUp + " state=" + phone.CurrentState + " parent=" + (phone.transform.parent == null ? "null" : phone.transform.parent.name));
            yield return null;
        }

        private IEnumerator TryPlace(SmartphoneController phone, LacrimosaController character, Vector3 target, string label)
        {
            var placed = character.PlaceAt(target);
            Log("Place[" + label + "]=" + placed + " state=" + phone.CurrentState + " pos=" + phone.transform.position.ToString("F3") + " surface=" + (phone.PlacementSurface == null ? "null" : phone.PlacementSurface.name));
            yield return null;
        }

        private static void ResetPhone(SmartphoneController phone, Vector3 position)
        {
            if (Physics.Raycast(new Vector3(position.x, 2.5f, position.z), Vector3.down, out var hit, 4f))
                position.y = hit.point.y + phone.GetComponent<Collider>().bounds.extents.y + 0.001f;

            phone.transform.SetParent(null, true);
            phone.Release(position, Quaternion.identity, null);
        }

        private void Log(string message)
        {
            if (log == null)
                log = new StringBuilder();
            log.AppendLine(message);
        }
    }
}
