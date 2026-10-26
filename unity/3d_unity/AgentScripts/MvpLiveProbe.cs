using System;
using System.Collections;
using System.Text;
using NUnit.Framework;
using RoomAgent.Brain;
using RoomAgent.Livestream;
using RoomAgent.Smartphone;
using UnityEngine;
using UnityEngine.Networking;
using UnityEngine.TestTools;

public static class MvpLiveProbe
{
    private const string ChatUrl = "http://127.0.0.1:8765/chat/message";
    private static string lastStatus = "not_started";

    public static IEnumerator PrepareSeededDance()
    {
        var loop = FindOne<BrainLoopController>();
        Assert.That(loop, Is.Not.Null, "BrainLoopController missing");
        loop.StopLoop();

        var phone = FindOne<SmartphoneController>();
        Assert.That(phone, Is.Not.Null, "SmartphoneController missing");

        var stream = FindOne<LivestreamSystem>();
        Assert.That(stream, Is.Not.Null, "LivestreamSystem missing");
        stream.StopStream();

        var bridge = FindOne<BrainActionBridge>();
        Assert.That(bridge, Is.Not.Null, "BrainActionBridge missing");

        // Normalize the setup state: if the phone is already held, place it
        // directly; otherwise exercise pickup first. Either way the final
        // setup state is Placed on the table through production components.
        if (phone.CurrentState != SmartphoneState.Held)
        {
            bool pickupDone = false;
            bool pickupSuccess = false;
            Assert.That(bridge.Execute("PICK_UP_PHONE", null, null, success =>
            {
                pickupSuccess = success;
                pickupDone = true;
            }), Is.True, "pickup execution was not accepted");
            yield return new WaitUntil(() => pickupDone);
            Assert.That(pickupSuccess, Is.True, "phone pickup failed");
            Assert.That(phone.CurrentState, Is.EqualTo(SmartphoneState.Held), "phone was not held");
        }

        bool placementDone = false;
        bool placementSuccess = false;
        Assert.That(bridge.Execute("PLACE_PHONE", "TABLE", null, success =>
        {
            placementSuccess = success;
            placementDone = true;
        }), Is.True, "placement execution was not accepted");
        yield return new WaitUntil(() => placementDone);
        Assert.That(placementSuccess, Is.True, "phone placement failed");
        Assert.That(phone.CurrentState, Is.EqualTo(SmartphoneState.Placed), "phone was not placed");
        Assert.That(phone.PlacementSurface, Is.Not.Null, "phone had no placement surface");

        Assert.That(stream.StartStream(), Is.True, "stream did not start");
        Assert.That(stream.IsStreaming, Is.True, "stream was not active");

        var viewers = stream.Viewers;
        Assert.That(viewers, Is.Not.Null, "viewer manager missing");
        if (viewers.GetViewer("viewer_mvp") == null)
            Assert.That(viewers.AddViewer("viewer_mvp"), Is.True, "viewer was not added");

        Debug.Log("[MvpLiveProbe] seeded context ready");
    }

    public static string PrepareSeededDanceStart()
    {
        if (lastStatus == "preparing")
            return lastStatus;

        lastStatus = "preparing";
        var host = new GameObject("MvpLiveProbeHost");
        UnityEngine.Object.DontDestroyOnLoad(host);
        host.AddComponent<MvpProbeHost>().Run(PrepareSeededDance(), () =>
        {
            lastStatus = "ready";
            UnityEngine.Object.Destroy(host);
        });
        return "started";
    }

    public static string PrepareStatus()
    {
        return lastStatus;
    }

    public static string SubmitDance()
    {
        var loop = FindOne<BrainLoopController>();
        Assert.That(loop, Is.Not.Null, "BrainLoopController missing");
        Assert.That(loop.SetViewerResponse("positive"), Is.True, "positive viewer response rejected");

        Assert.That(BrainChatSubmitter.Submit("viewer_mvp", "Lacrimosa, dance!"), Is.True, "chat submission failed");
        return "submitted";
    }

    public static string DescribeRuntime()
    {
        var loop = FindOne<BrainLoopController>();
        var bridge = FindOne<BrainActionBridge>();
        var phone = FindOne<SmartphoneController>();
        var stream = FindOne<LivestreamSystem>();
        if (loop == null || bridge == null || phone == null || stream == null)
            return "missing_runtime_component";

        return $"running={loop.IsRunning};busy={bridge.IsBusy};phone={phone.CurrentState};surface={(phone.PlacementSurface ? phone.PlacementSurface.name : "none")};stream={stream.IsStreaming};viewers={stream.Viewers.ViewerCount}";
    }


    public static string DescribePositions()
    {
        var bridge = FindOne<BrainActionBridge>();
        var phone = FindOne<SmartphoneController>();
        var character = FindOne<RoomAgent.Lacrimosa.LacrimosaController>();
        var table = GameObject.Find("Table");
        if (bridge == null || phone == null || character == null || table == null)
            return "missing_position_component";

        var collider = table.GetComponentInChildren<Collider>();
        return $"character={character.transform.position};phone={phone.transform.position};table={(collider ? collider.bounds.center : table.transform.position)};canInteract={character.CanInteract(phone)}";
    }

    public static string TryDirectPickup()
    {
        var character = FindOne<RoomAgent.Lacrimosa.LacrimosaController>();
        var phone = FindOne<SmartphoneController>();
        if (character == null || phone == null)
            return "missing_pickup_component";

        var canInteract = character.CanInteract(phone);
        var pickedUp = canInteract && character.PickUp(phone);
        return $"canInteract={canInteract};pickedUp={pickedUp};state={phone.CurrentState}";
    }

    public static string TryDirectPlacementOnTable()
    {
        var character = FindOne<RoomAgent.Lacrimosa.LacrimosaController>();
        var phone = FindOne<SmartphoneController>();
        var table = GameObject.Find("Table");
        var tableCollider = table != null ? table.GetComponentInChildren<Collider>() : null;
        if (character == null || phone == null || tableCollider == null)
            return "missing_placement_component";

        var placed = character.HeldPhone != null && character.PlaceAt(tableCollider.bounds.center);
        return $"placed={placed};state={phone.CurrentState};surface={(phone.PlacementSurface ? phone.PlacementSurface.name : "none")}";
    }

    public static string StartSeededStream()
    {
        var loop = FindOne<BrainLoopController>();
        var stream = FindOne<LivestreamSystem>();
        if (loop == null || stream == null)
            return "missing_stream_component";

        loop.StopLoop();
        var started = stream.StartStream();
        var viewers = stream.Viewers;
        var viewerAdded = viewers.GetViewer("viewer_mvp") != null || viewers.AddViewer("viewer_mvp");
        return $"started={started};viewerAdded={viewerAdded};stream={stream.IsStreaming};viewers={viewers.ViewerCount};brainRunning={loop.IsRunning}";
    }

    public static string PrepareDanceReaction()
    {
        var loop = FindOne<BrainLoopController>();
        if (loop == null)
            return "missing_brain_loop";

        return $"responseSet={loop.SetViewerResponse("positive")};running={loop.IsRunning}";
    }

    public static string StartDanceTrial()
    {
        var loop = FindOne<BrainLoopController>();
        if (loop == null)
            return "missing_brain_loop";

        var responseSet = loop.SetViewerResponse("positive");
        if (!loop.IsRunning)
            loop.StartLoop();
        return $"responseSet={responseSet};running={loop.IsRunning}";
    }

    public static string StartDanceReactionWatcher()
    {
        var loop = FindOne<BrainLoopController>();
        if (loop == null)
            return "missing_brain_loop";

        var host = new GameObject("MvpDanceReactionWatcherHost");
        UnityEngine.Object.DontDestroyOnLoad(host);
        host.AddComponent<MvpProbeHost>().Run(WatchForDance(loop), () =>
        {
            UnityEngine.Object.Destroy(host);
        });
        return "watcher_started";
    }

    private static IEnumerator WatchForDance(BrainLoopController loop)
    {
        const string stateUrl = "http://127.0.0.1:8766/brain/state";
        const float timeoutSeconds = 30f;
        var elapsed = 0f;
        while (elapsed < timeoutSeconds)
        {
            using var request = UnityWebRequest.Get(stateUrl);
            request.timeout = 2;
            yield return request.SendWebRequest();
            if (request.result == UnityWebRequest.Result.Success)
            {
                var body = request.downloadHandler.text;
                var state = JsonUtility.FromJson<BrainStateResponse>(body);
                if (state != null && state.pending_action != null && state.pending_action.action == "DANCE")
                {
                    loop.SetViewerResponse("positive");
                    Debug.Log("[MvpLiveProbe] positive reaction staged for pending DANCE");
                    yield break;
                }
            }

            elapsed += Time.unscaledDeltaTime;
            yield return new WaitForSeconds(0.25f);
        }

        Debug.LogWarning("[MvpLiveProbe] pending DANCE action was not observed");
    }

    [Serializable]
    private sealed class BrainStateResponse
    {
        public BrainActionRequest pending_action;
    }

    private static T FindOne<T>() where T : UnityEngine.Object
    {
        return UnityEngine.Object.FindAnyObjectByType<T>();
    }

    private sealed class MvpProbeHost : MonoBehaviour
    {
        private static MvpProbeHost active;

        public void Run(IEnumerator coroutine, Action completed)
        {
            if (active != null && active != this)
            {
                Destroy(gameObject);
                return;
            }

            active = this;
            StartCoroutine(RunAndComplete(coroutine, completed));
        }

        private IEnumerator RunAndComplete(IEnumerator coroutine, Action completed)
        {
            yield return coroutine;
            completed?.Invoke();
        }

        private void OnDestroy()
        {
            if (active == this)
                active = null;
            StopAllCoroutines();
        }
    }
}
