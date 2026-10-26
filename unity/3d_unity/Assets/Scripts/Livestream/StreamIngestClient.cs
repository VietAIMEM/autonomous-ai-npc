using System;
using System.Collections.Generic;
using System.Globalization;
using System.IO;
using System.Net.Sockets;
using System.Text;
using System.Threading;
using UnityEngine;

namespace RoomAgent.Livestream
{
    /// <summary>
    /// PHASE 13 capture pump: hands the real stream its picture and sound.
    ///
    /// The main thread only renders a JPEG and reads back pixels; one sender
    /// thread owns every socket, including the start/stop control calls, so a
    /// slow provider can never stall a frame. LivestreamSystem still decides
    /// whether the stream is logically on; this class only moves bytes.
    /// </summary>
    [DisallowMultipleComponent]
    public sealed class StreamIngestClient : MonoBehaviour
    {
        public const string DefaultBaseUrl = "http://127.0.0.1:8766";

        private const int ConnectTimeoutMs = 1500;
        private const int ResponseTimeoutMs = 2000;
        private const int IdleSleepMs = 5;

        /// Capture outruns the socket on a congested link; two frames is enough
        /// to hide jitter and small enough that the stream never lags the scene.
        private const int MaxPendingFrames = 2;
        private const int MaxRingSeconds = 2;
        private const int AudioChunkMs = 100;
        private const int MaxReplyBytes = 262_144;

        private static StreamIngestClient instance;

        private readonly object videoGate = new object();
        private readonly Queue<byte[]> pendingFrames = new Queue<byte[]>();
        private readonly object audioGate = new object();

        private short[] ring;
        private int ringWrite;
        private int ringFilled;

        private Camera source;
        /// <summary>
        /// Cached on the main thread. Reading Camera.name off the sender thread is
        /// illegal in Unity and made every video post throw.
        /// </summary>
        private string cameraName = string.Empty;
        private RenderTexture frame;
        private Texture2D scratch;
        private Thread worker;
        private ManualResetEventSlim wake = new ManualResetEventSlim(false);

        private string host = "127.0.0.1";
        private int port = 8766;

        private int width = 640;
        private int height = 360;
        private int jpegQuality = 70;
        private float fps = 15f;
        private int sampleRate = 48000;
        private int ringChannels = 1;

        private long videoSequence;
        private long audioSequence;
        private long framesCaptured;
        private long framesPosted;
        private long audioPosted;
        private long framesDropped;
        private long audioDropped;
        private long postFailures;
        private long workerLoops;

        private long nextCaptureAtTicks;
        private long nextAudioAtTicks;

        /// <summary>
        /// Monotonic identity of the current pump. It only advances in
        /// BeginPump/StopPump, so a test or operator can prove that a camera
        /// source change happened inside one continuous pump session instead of
        /// a pump restart.
        /// </summary>
        private long pumpSessionId;
        /// <summary>Advances every time the active capture camera changes.</summary>
        private long sourceGeneration;
        /// <summary>Test-only injection point for a camera with no pump bound.</summary>
        private Camera simulatedSource;

        private volatile bool capturing;
        private volatile bool stopWorker;
        private volatile bool stopRequested;
        private volatile bool profileReady;
        private volatile bool transportConfigured;

        private volatile int control;
        private float controlWaitSeconds = 15f;
        private volatile bool controlPending;
        private volatile bool controlOk;
        private volatile string controlState = string.Empty;
        private volatile string controlReason = string.Empty;
        /// <summary>Last stage the sender thread entered, for liveness diagnostics.</summary>
        private volatile string workerStage = "idle";
        private volatile string workerError = string.Empty;

        private const int ControlNone = 0;
        private const int ControlProfile = 1;
        private const int ControlStart = 2;
        private const int ControlStop = 3;

        public static StreamIngestClient Instance => instance;
        public bool IsCapturing => capturing;
        public bool ProfileReady => profileReady;
        public bool TransportConfigured => transportConfigured;
        public bool ControlPending => controlPending;
        public bool ControlOk => controlOk;
        public string ControlState => controlState;
        public string ControlReason => controlReason;
        public string WorkerStage => workerStage;
        public string WorkerError => workerError;
        public int Width => width;
        public int Height => height;
        public float Fps => fps;
        public int SampleRate => sampleRate;
        public long FramesCaptured => Interlocked.Read(ref framesCaptured);
        public long FramesPosted => Interlocked.Read(ref framesPosted);
        public long AudioPosted => Interlocked.Read(ref audioPosted);
        public long FramesDropped => Interlocked.Read(ref framesDropped);
        public long AudioDropped => Interlocked.Read(ref audioDropped);
        public long PostFailures => Interlocked.Read(ref postFailures);
        /// <summary>Sender-thread heartbeat: proves the pump is alive and not parked.</summary>
        public long WorkerLoops => Interlocked.Read(ref workerLoops);
        public long PumpSessionId => Interlocked.Read(ref pumpSessionId);
        public long SourceGeneration => Interlocked.Read(ref sourceGeneration);
        public bool IsPumping => capturing;
        public bool HasSource => source != null;
        public string CameraName => cameraName;

        public static StreamIngestClient GetOrCreate()
        {
            if (instance != null)
                return instance;

            var existing = FindFirstObjectByType<StreamIngestClient>();
            if (existing != null)
            {
                instance = existing;
                return existing;
            }

            var host = new GameObject("StreamIngestClient");
            // Editor probes drive the same class, and DontDestroyOnLoad is illegal
            // outside play mode.
            if (Application.isPlaying)
                UnityEngine.Object.DontDestroyOnLoad(host);
            host.hideFlags = HideFlags.DontSave;
            var client = host.AddComponent<StreamIngestClient>();
            instance = client;
            return client;
        }

        public static void DestroyInstance()
        {
            if (instance == null)
                return;

            var client = instance;
            instance = null;
            client.Shutdown();
        }

        private void Awake()
        {
            if (instance != null && instance != this)
            {
                DestroyImmediate(gameObject);
                return;
            }

            instance = this;
            EnsureWake();
        }

        private void Update()
        {
            if (capturing)
                CaptureNow();
        }

        private void OnDestroy()
        {
            StopPump();
            if (instance == this)
                instance = null;
        }

        /// <summary>
        /// Starts the pump against a running backend. Returns as soon as the
        /// capture clock is armed; the profile and the transport are negotiated
        /// by the sender thread so the caller never blocks on the network.
        /// </summary>
        public bool BeginPump(Camera camera, string baseUrl)
        {
            if (camera == null)
                return false;

            if (!TryHost(baseUrl, out host, out port))
                return false;

            source = camera;
            cameraName = camera.name;
            Interlocked.Increment(ref pumpSessionId);
            Interlocked.Increment(ref sourceGeneration);
            profileReady = false;
            transportConfigured = false;
            stopRequested = false;
            controlOk = false;
            controlState = string.Empty;
            controlReason = string.Empty;
            Interlocked.Exchange(ref videoSequence, 0);
            Interlocked.Exchange(ref audioSequence, 0);
            Interlocked.Exchange(ref framesCaptured, 0);
            Interlocked.Exchange(ref framesPosted, 0);
            Interlocked.Exchange(ref audioPosted, 0);
            Interlocked.Exchange(ref framesDropped, 0);
            Interlocked.Exchange(ref audioDropped, 0);
            Interlocked.Exchange(ref postFailures, 0);
            lock (videoGate)
                pendingFrames.Clear();
            lock (audioGate)
            {
                ringWrite = 0;
                ringFilled = 0;
            }
            nextAudioAtTicks = DateTime.UtcNow.Ticks;

            EnsureTap();
            capturing = true;
            nextCaptureAtTicks = DateTime.UtcNow.Ticks;

            EnsureWake();
            if (worker == null || !worker.IsAlive)
            {
                stopWorker = false;
                worker = new Thread(Pump)
                {
                    IsBackground = true,
                    Name = "stream-ingest"
                };
                worker.Start();
            }

            EnqueueControl(ControlProfile);
            return true;
        }

        /// <summary>
        /// Swaps the capture camera inside a running pump session. Called on the
        /// Unity main thread only: CaptureNow/RenderFrame run from Update, and
        /// the sender thread never dereferences the Camera object, so no pump
        /// restart, render-target teardown, or counter reset is required.
        /// </summary>
        public bool SetSourceCamera(Camera camera)
        {
            if (camera == null)
                return false;
            if (source == camera && simulatedSource == null)
                return true;

            source = camera;
            simulatedSource = null;
            // Cached here because reading Camera.name off the sender thread is
            // illegal in Unity.
            cameraName = camera.name;
            Interlocked.Increment(ref sourceGeneration);
            return true;
        }

        /// <summary>
        /// Test-only equivalent of BeginPump's camera binding. It never touches
        /// the network, worker thread, or capture clock, so a Play-Mode test can
        /// prove same-session switching without starting an encoder.
        /// </summary>
        public bool SetSimulatedSource(Camera camera)
        {
            if (camera == null)
                return false;

            simulatedSource = camera;
            source = camera;
            cameraName = camera.name;
            Interlocked.Increment(ref sourceGeneration);
            return true;
        }

        public void ClearSimulatedSource()
        {
            simulatedSource = null;
            source = null;
            cameraName = string.Empty;
        }

        /// <summary>
        /// Test-only: establishes pump-session identity so a test can prove a
        /// camera swap happened without a pump restart, and without touching
        /// the network or starting an encoder.
        /// </summary>
        public void ResetPumpSessionIdentityForTest()
        {
            Interlocked.Exchange(ref pumpSessionId, 1);
            Interlocked.Exchange(ref sourceGeneration, 0);
            capturing = true;
        }

        public void EndSimulatedSession()
        {
            capturing = false;
            ClearSimulatedSource();
        }

        /// <summary>
        /// Test-only render through the real RenderFrame path. Returns the JPEG
        /// byte count and payload hash so a test can prove the rendered image
        /// actually changed with the camera source.
        /// </summary>
        public bool SimulateCaptureOnce(out int byteCount, out int payloadHash)
        {
            byteCount = 0;
            payloadHash = 0;
            if (source == null)
                return false;

            if (!SimulateCapturePayload(out var payload))
                return false;

            byteCount = payload.Length;
            payloadHash = ComputePayloadHash(payload);
            return true;
        }

        /// <summary>
        /// Test-only: renders one frame through the real RenderFrame path and
        /// returns the JPEG payload so evidence can be saved from the same
        /// simulated pump session.
        /// </summary>
        public bool SimulateCapturePayload(out byte[] payload)
        {
            payload = null;
            if (source == null)
                return false;

            payload = RenderFrame();
            return payload != null && payload.Length > 0;
        }

        private static int ComputePayloadHash(byte[] payload)
        {
            unchecked
            {
                var hash = 17;
                for (var index = 0; index < payload.Length; index += 7)
                    hash = hash * 31 + payload[index];
                hash = hash * 31 + payload.Length;
                return hash;
            }
        }

        /// <summary>Stops capture and releases every GPU handle the pump owns.</summary>
        public void StopPump()
        {
            capturing = false;
            lock (videoGate)
                pendingFrames.Clear();
            ReleaseRenderTargets();
            source = null;
            simulatedSource = null;
            Interlocked.Increment(ref pumpSessionId);
        }

        /// <summary>Full teardown: thread, buffers and the host object.</summary>
        public void Shutdown()
        {
            StopPump();
            stopWorker = true;
            Wake();
            if (worker != null && worker.IsAlive)
                worker.Join(2000);
            worker = null;
            ring = null;
            wake = null;

            if (instance == this)
                instance = null;
            if (this != null && gameObject != null)
                DestroyImmediate(gameObject);
        }

        public void RequestProfile() => EnqueueControl(ControlProfile);

        public void RequestStart(float waitSeconds)
        {
            controlWaitSeconds = Mathf.Clamp(waitSeconds, 0f, 30f);
            stopRequested = false;
            // Stopping tears capture down, and an encoder only ever becomes LIVE
            // once pictures reach it, so a restart has to re-arm the capture clock
            // here rather than wait for someone to call BeginPump again.
            if (source != null)
            {
                nextCaptureAtTicks = DateTime.UtcNow.Ticks;
                capturing = true;
            }
            EnqueueControl(ControlStart);
        }

        public void RequestStop(float waitSeconds)
        {
            controlWaitSeconds = Mathf.Clamp(waitSeconds, 0f, 30f);
            EnqueueControl(ControlStop);
        }

        private void EnqueueControl(int kind)
        {
            controlOk = false;
            controlReason = string.Empty;
            controlPending = true;
            // A stop asked for mid-wait has to be able to cut the wait short,
            // otherwise STOP_STREAM would queue behind a live start attempt.
            if (kind == ControlStop)
                stopRequested = true;
            control = kind;
            Wake();
        }

        /// <summary>
        /// Renders one picture from the stream camera and queues it. Safe to call
        /// from Update or from an editor probe; it is a no-op while the pump is off.
        /// </summary>
        public bool CaptureNow()
        {
            if (!capturing || source == null)
                return false;
            var nowTicks = DateTime.UtcNow.Ticks;
            if (nowTicks < Interlocked.Read(ref nextCaptureAtTicks))
                return false;

            // Wall clock, not Time: the same pump has to pace correctly when an
            // editor probe drives it without a play loop.
            var frameTicks = (long)(TimeSpan.TicksPerSecond / Math.Max(1.0, (double)fps));
            var scheduled = Math.Max(Interlocked.Read(ref nextCaptureAtTicks), nowTicks) + frameTicks;
            Interlocked.Exchange(ref nextCaptureAtTicks, scheduled);

            var payload = RenderFrame();
            if (payload == null || payload.Length == 0)
                return false;

            Interlocked.Increment(ref framesCaptured);
            lock (videoGate)
            {
                while (pendingFrames.Count >= MaxPendingFrames)
                {
                    pendingFrames.Dequeue();
                    Interlocked.Increment(ref framesDropped);
                }
                pendingFrames.Enqueue(payload);
            }
            Wake();
            return true;
        }

        /// <summary>
        /// Entry point for Unity's audio output. OnAudioFilterRead calls this on
        /// the audio thread; an editor probe can call it with the same samples.
        /// </summary>
        public void PushAudio(float[] data, int incomingChannels)
        {
            if (data == null || data.Length == 0)
                return;

            var channelCount = Mathf.Max(1, incomingChannels);
            lock (audioGate)
            {
                var buffer = ring;
                if (buffer == null || buffer.Length == 0)
                    return;

                // The ring is always mono: the stream carries a single mixed track,
                // and a stereo encoder input is filled by repeating each sample.
                for (var index = 0; index + channelCount <= data.Length; index += channelCount)
                {
                    var mixed = 0f;
                    for (var channel = 0; channel < channelCount; channel++)
                        mixed += data[index + channel];
                    mixed /= channelCount;

                    var sample = (short)Mathf.Clamp(
                        Mathf.RoundToInt(mixed * 32767f),
                        short.MinValue,
                        short.MaxValue);

                    buffer[ringWrite] = sample;
                    ringWrite = (ringWrite + 1) % buffer.Length;
                    if (ringFilled < buffer.Length)
                        ringFilled++;
                    else
                        Interlocked.Increment(ref audioDropped);
                }
            }
        }

        private void EnsureWake()
        {
            // A domain reload rebuilds the managed side of an editor-created
            // component, so the handle is treated as something that can go missing.
            if (wake == null)
                wake = new ManualResetEventSlim(false);
        }

        private void Wake()
        {
            EnsureWake();
            wake.Set();
        }

        private void EnsureTap()
        {
            // Adding a component to a scene object would dirty the editor scene,
            // so the tap is only ever attached while the game is actually running.
            if (!Application.isPlaying)
                return;

            var player = FindFirstObjectByType<Speech.SpeechAudioPlayer>();
            if (player == null || player.gameObject.GetComponent<StreamAudioTap>() != null)
                return;

            var tap = player.gameObject.AddComponent<StreamAudioTap>();
            tap.hideFlags = HideFlags.DontSave;
        }

        private byte[] RenderFrame()
        {
            if (frame == null ||
                frame.width != width ||
                frame.height != height)
            {
                ReleaseRenderTargets();
                frame = new RenderTexture(width, height, 24, RenderTextureFormat.ARGB32)
                {
                    name = "StreamFrame",
                    hideFlags = HideFlags.DontSave
                };
                frame.Create();
                scratch = new Texture2D(width, height, TextureFormat.RGB24, false)
                {
                    name = "StreamScratch",
                    hideFlags = HideFlags.DontSave
                };
            }

            var previousTarget = source.targetTexture;
            var previousActive = RenderTexture.active;
            source.targetTexture = frame;
            source.Render();
            source.targetTexture = previousTarget;

            RenderTexture.active = frame;
            scratch.ReadPixels(new Rect(0, 0, width, height), 0, 0);
            scratch.Apply();
            RenderTexture.active = previousActive;

            try
            {
                return scratch.EncodeToJPG(jpegQuality);
            }
            catch (Exception error)
            {
                Debug.Log($"[StreamIngestClient] JPEG encode failed: {error.Message}");
                return null;
            }
        }

        private void ReleaseRenderTargets()
        {
            if (frame != null)
            {
                frame.Release();
                DestroyImmediate(frame);
                frame = null;
            }
            if (scratch != null)
            {
                DestroyImmediate(scratch);
                scratch = null;
            }
        }

        private void Pump()
        {
            // An exception on this thread would silently stop the stream, so every
            // pass is guarded and reported instead of escaping into the void.
            while (!stopWorker)
            {
                Interlocked.Increment(ref workerLoops);
                try
                {
                    var kind = Interlocked.Exchange(ref control, ControlNone);
                    if (kind != ControlNone)
                        RunControl(kind);

                    workerStage = "media";
                    var worked = FlushMedia();
                    if (!worked)
                    {
                        workerStage = "idle";
                        wake.Reset();
                        wake.Wait(IdleSleepMs * 4);
                    }
                }
                catch (Exception error)
                {
                    Interlocked.Increment(ref postFailures);
                    workerStage = "error";
                    if (string.IsNullOrEmpty(workerError))
                        workerError = $"{error.GetType().Name}: {error.Message}";
                    Debug.Log($"[StreamIngestClient] pump iteration failed: {error.GetType().Name}: {error.Message}");
                }
            }

            // A stop queued while this thread was waiting for LIVE still has to
            // reach the backend, or a torn-down pump would leave the encoder running.
            workerStage = "drain";
            if (Interlocked.Exchange(ref control, ControlNone) == ControlStop)
                RunControl(ControlStop);
            workerStage = "stopped";
        }

        /// <summary>One pass over the video queue and the wall-clock audio track.</summary>
        public void FlushMediaNow() => FlushMedia();

        private bool FlushMedia()
        {
            var worked = SendOneFrame();
            worked |= SendAudio();
            return worked;
        }

        private void RunControl(int kind)
        {
            switch (kind)
            {
                case ControlProfile:
                    workerStage = "profile";
                    FetchProfile();
                    break;
                case ControlStart:
                    workerStage = "start";
                    // The control mailbox holds one request, so a start asked for in
                    // the same frame as BeginPump can overwrite the pending profile
                    // fetch. Without that profile the audio ring never exists and the
                    // encoder dies on a video-only stream, so guarantee it here.
                    if (!profileReady)
                        FetchProfile();
                    StartTransport();
                    break;
                case ControlStop:
                    workerStage = "stop";
                    StopTransport();
                    break;
            }
            controlPending = false;
            workerStage = "idle";
        }

        private void FetchProfile()
        {
            var reply = Send("GET", "/stream/state", null);
            if (!reply.Ok)
            {
                Interlocked.Increment(ref postFailures);
                transportConfigured = false;
                controlOk = false;
                controlReason = reply.Failure ?? "PROFILE_UNAVAILABLE";
                return;
            }

            StateResponse state = Parse<StateResponse>(reply.Body);
            if (state == null)
            {
                controlOk = false;
                controlReason = "PROFILE_MALFORMED";
                return;
            }

            transportConfigured = state.enabled && state.transport_enabled;
            controlState = state.state ?? string.Empty;
            if (state.ingest == null)
            {
                profileReady = true;
                controlOk = transportConfigured;
                controlReason = transportConfigured ? string.Empty : "TRANSPORT_DISABLED";
                return;
            }

            ApplyProfile(state.ingest);
            profileReady = true;
            controlOk = transportConfigured;
            controlReason = transportConfigured ? string.Empty : "TRANSPORT_DISABLED";
        }

        private void ApplyProfile(IngestProfile profile)
        {
            width = Mathf.Max(16, profile.width > 0 ? profile.width : width);
            height = Mathf.Max(16, profile.height > 0 ? profile.height : height);
            fps = profile.fps > 0f ? profile.fps : fps;
            jpegQuality = Mathf.Clamp(profile.jpeg_quality > 0 ? profile.jpeg_quality : jpegQuality, 1, 100);
            sampleRate = profile.audio_sample_rate > 0 ? profile.audio_sample_rate : sampleRate;
            ringChannels = Mathf.Clamp(profile.audio_channels > 0 ? profile.audio_channels : ringChannels, 1, 2);

            var needed = sampleRate * MaxRingSeconds;
            lock (audioGate)
            {
                if (ring == null || ring.Length != needed)
                    ring = new short[needed];
                ringWrite = 0;
                ringFilled = 0;
            }
        }

        private void StartTransport()
        {
            // The encoder only becomes LIVE once real pictures reach it, and this
            // is the thread that sends them, so the start request gets a short
            // budget and the rest of the wait is spent feeding media and polling
            // state. Blocking here would starve the very gate it waits on.
            workerStage = "start:post";
            var reply = Send("POST", "/stream/start?wait=0.5", null, 6000);
            CommandResponse result = reply.Ok ? Parse<CommandResponse>(reply.Body) : null;
            if (result == null)
            {
                Interlocked.Increment(ref postFailures);
                controlOk = false;
                controlState = reply.Ok ? "UNKNOWN" : "DISCONNECTED";
                controlReason = reply.Failure ?? "START_UNREACHABLE";
                return;
            }

            controlState = result.state ?? string.Empty;
            controlReason = string.IsNullOrEmpty(result.reason) ? string.Empty : result.reason;
            controlOk = result.ok;
            if (result.ok)
                return;

            // A short wait routinely ends in TIMEOUT/BACKOFF while ffmpeg is still
            // coming up, so confirm the settled state before declaring failure.
            controlOk = WaitForLive(controlWaitSeconds);
        }

        private bool WaitForLive(float budgetSeconds)
        {
            var deadline = DateTime.UtcNow.AddSeconds(Mathf.Max(0f, budgetSeconds));
            while (DateTime.UtcNow < deadline && !stopRequested && !stopWorker)
            {
                workerStage = "wait-live";
                FlushMedia();
                Thread.Sleep(40);
                FlushMedia();

                var reply = Send("GET", "/stream/state", null);
                if (!reply.Ok)
                {
                    Interlocked.Increment(ref postFailures);
                    continue;
                }

                StateResponse state = Parse<StateResponse>(reply.Body);
                if (state == null)
                    continue;
                controlState = state.state ?? string.Empty;
                if (controlState == "LIVE")
                {
                    controlReason = string.Empty;
                    return true;
                }
                if (state.state == "STOPPED" || state.state == "IDLE" || state.state == "ERROR")
                {
                    controlReason = state.reason ?? "NOT_LIVE";
                    return false;
                }
            }

            if (string.IsNullOrEmpty(controlReason))
                controlReason = stopRequested ? "STOP_PENDING" : "NOT_LIVE";
            return false;
        }

        private void StopTransport()
        {
            capturing = false;
            stopRequested = true;
            lock (videoGate)
                pendingFrames.Clear();
            var path = $"/stream/stop?wait={controlWaitSeconds.ToString("0.###", CultureInfo.InvariantCulture)}";
            var reply = Send("POST", path, null, Mathf.CeilToInt(controlWaitSeconds * 1000f) + 4000);
            CommandResponse result = reply.Ok ? Parse<CommandResponse>(reply.Body) : null;
            if (result == null)
            {
                Interlocked.Increment(ref postFailures);
                controlOk = false;
                controlReason = reply.Failure ?? "STOP_UNREACHABLE";
                return;
            }

            controlOk = result.ok;
            controlState = result.state ?? string.Empty;
            controlReason = string.IsNullOrEmpty(result.reason) ? string.Empty : result.reason;
        }

        private bool SendOneFrame()
        {
            byte[] payload;
            lock (videoGate)
            {
                if (pendingFrames.Count == 0)
                    return false;
                payload = pendingFrames.Dequeue();
            }

            var sequence = Interlocked.Increment(ref videoSequence);
            var path = $"/stream/media/video?seq={sequence}&camera={Uri.EscapeDataString(CameraName)}";
            var reply = Send("POST", path, payload);
            if (reply.Ok)
            {
                Interlocked.Increment(ref framesPosted);
                if (reply.Body != null && reply.Body.Contains("\"NOT_STREAMING\""))
                {
                    // The transport is not up yet: keep pumping quietly instead of
                    // treating the first frames of a start attempt as failures.
                    Interlocked.Decrement(ref videoSequence);
                    Interlocked.Decrement(ref framesPosted);
                }
            }
            else
            {
                Interlocked.Increment(ref postFailures);
            }
            return true;
        }

        private bool SendAudio()
        {
            if (!capturing || ring == null)
                return false;

            var nowTicks = DateTime.UtcNow.Ticks;
            var elapsedMs = (nowTicks - Interlocked.Read(ref nextAudioAtTicks)) / TimeSpan.TicksPerMillisecond;
            if (elapsedMs < AudioChunkMs)
                return false;

            var wanted = (int)(elapsedMs * sampleRate / 1000L);
            if (wanted <= 0)
                return false;
            wanted = Math.Min(wanted, sampleRate);
            Interlocked.Add(ref nextAudioAtTicks, (long)(wanted * 1000L / sampleRate) * TimeSpan.TicksPerMillisecond);

            var payload = ReadAudio(wanted);
            if (payload == null)
                return false;

            var sequence = Interlocked.Increment(ref audioSequence);
            var path = $"/stream/media/audio?seq={sequence}&rate={sampleRate}&channels={ringChannels}";
            var reply = Send("POST", path, payload);
            if (reply.Ok)
            {
                Interlocked.Increment(ref audioPosted);
                if (reply.Body != null && reply.Body.Contains("\"NOT_STREAMING\""))
                    Interlocked.Decrement(ref audioSequence);
            }
            else
            {
                Interlocked.Increment(ref postFailures);
            }
            return true;
        }

        /// <summary>
        /// Pulls the requested number of samples as little-endian s16le, filling
        /// gaps with silence so the encoder's audio clock always advances.
        /// </summary>
        private byte[] ReadAudio(int samples)
        {
            var buffer = ring;
            var bytes = new byte[samples * ringChannels * 2];
            var cursor = 0;
            lock (audioGate)
            {
                var start = ringWrite - ringFilled;
                if (start < 0)
                    start += buffer.Length;

                for (var index = 0; index < samples; index++)
                {
                    short value = 0;
                    if (ringFilled > 0)
                    {
                        value = buffer[start];
                        start = (start + 1) % buffer.Length;
                        ringFilled--;
                    }
                    for (var channel = 0; channel < ringChannels; channel++)
                    {
                        bytes[cursor++] = (byte)(value & 0xFF);
                        bytes[cursor++] = (byte)((value >> 8) & 0xFF);
                    }
                }
            }
            return bytes;
        }

        private T Parse<T>(string body) where T : class
        {
            try
            {
                return JsonUtility.FromJson<T>(body);
            }
            catch (Exception)
            {
                return null;
            }
        }

        private HttpReply Send(string method, string path, byte[] body)
            => Send(method, path, body, ResponseTimeoutMs);

        private HttpReply Send(string method, string path, byte[] body, int timeoutMs)
        {
            var client = new TcpClient();
            try
            {
                var connect = client.BeginConnect(host, port, null, null);
                if (!connect.AsyncWaitHandle.WaitOne(ConnectTimeoutMs))
                {
                    client.Close();
                    return HttpReply.Fail("CONNECT_TIMEOUT");
                }
                client.EndConnect(connect);

                using (var stream = client.GetStream())
                {
                    stream.ReadTimeout = timeoutMs;
                    stream.WriteTimeout = timeoutMs;

                    var header = new StringBuilder();
                    header.Append(method).Append(' ').Append(path).Append(" HTTP/1.1\r\n");
                    header.Append("Host: ").Append(host).Append(':').Append(port).Append("\r\n");
                    header.Append("Accept: application/json\r\n");
                    header.Append("Content-Type: application/octet-stream\r\n");
                    header.Append("Connection: close\r\n");
                    header.Append("Content-Length: ")
                        .Append(body == null ? 0 : body.Length)
                        .Append("\r\n\r\n");

                    var head = Encoding.ASCII.GetBytes(header.ToString());
                    stream.Write(head, 0, head.Length);
                    if (body != null && body.Length > 0)
                        stream.Write(body, 0, body.Length);
                    stream.Flush();

                    return ReadReply(stream);
                }
            }
            catch (SocketException)
            {
                return HttpReply.Fail("SOCKET_ERROR");
            }
            catch (IOException)
            {
                return HttpReply.Fail("IO_ERROR");
            }
            catch (Exception)
            {
                return HttpReply.Fail("TRANSPORT_ERROR");
            }
            finally
            {
                client.Close();
            }
        }

        private static HttpReply ReadReply(Stream stream)
        {
            var buffer = new byte[16 * 1024];
            var raw = new MemoryStream();
            while (true)
            {
                int read;
                try
                {
                    read = stream.Read(buffer, 0, buffer.Length);
                }
                catch (IOException)
                {
                    break;
                }
                if (read <= 0)
                    break;
                raw.Write(buffer, 0, read);
                if (raw.Length > MaxReplyBytes)
                    break;
            }

            var bytes = raw.ToArray();
            var boundary = IndexOf(bytes, new byte[] { 13, 10, 13, 10 });
            if (boundary < 0)
                return HttpReply.Fail("MALFORMED_REPLY");

            var head = Encoding.ASCII.GetString(bytes, 0, boundary);
            var parts = head.Split(' ');
            if (parts.Length < 2 || !int.TryParse(parts[1], out var status))
                return HttpReply.Fail("MALFORMED_STATUS");

            var body = new byte[bytes.Length - boundary - 4];
            Array.Copy(bytes, boundary + 4, body, 0, body.Length);
            return new HttpReply
            {
                Status = status,
                Body = Encoding.UTF8.GetString(body)
            };
        }

        private static int IndexOf(byte[] haystack, byte[] needle)
        {
            for (var index = 0; index <= haystack.Length - needle.Length; index++)
            {
                var matched = true;
                for (var offset = 0; offset < needle.Length; offset++)
                {
                    if (haystack[index + offset] == needle[offset])
                        continue;
                    matched = false;
                    break;
                }
                if (matched)
                    return index;
            }
            return -1;
        }

        public static bool TryHost(string baseUrl, out string hostName, out int portNumber)
        {
            hostName = "127.0.0.1";
            portNumber = 8766;
            if (string.IsNullOrWhiteSpace(baseUrl))
                return false;

            if (!Uri.TryCreate(baseUrl.Trim(), UriKind.Absolute, out var uri))
                return false;
            if (uri.Scheme != Uri.UriSchemeHttp)
                return false;

            hostName = uri.Host;
            portNumber = uri.Port;
            return true;
        }

        [Serializable]
        private sealed class IngestProfile
        {
            public int width;
            public int height;
            public float fps;
            public int jpeg_quality;
            public int audio_sample_rate;
            public int audio_channels;
            public int video_queue_capacity;
            public int audio_queue_capacity;
            public int max_frame_bytes;
            public int max_audio_bytes;
        }

        [Serializable]
        private sealed class StateResponse
        {
            public string provider;
            public string state;
            public bool enabled;
            public bool transport_enabled;
            public bool real_transport;
            public string reason;
            public IngestProfile ingest;
        }

        [Serializable]
        private sealed class CommandResponse
        {
            public bool ok;
            public string state;
            public string reason;
            public bool live;
            public int waited_ms;
        }

        private sealed class HttpReply
        {
            public int Status;
            public string Body = string.Empty;
            public string Failure;

            public bool Ok => Status >= 200 && Status < 300;

            public static HttpReply Fail(string reason)
                => new HttpReply { Status = 0, Failure = reason };
        }
    }
}
