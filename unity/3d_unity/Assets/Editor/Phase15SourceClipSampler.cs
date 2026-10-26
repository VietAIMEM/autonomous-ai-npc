using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using UnityEditor;
using UnityEngine;

namespace RoomAgent.Editor
{
    public static class Phase15SourceClipSampler
    {
        private const string FbxPath = "Assets/animation/phase15_motius/lacrimosa_production_motius_animated.fbx";
        private const string ClipName = "MotiusPhase15Wave";
        private const string ReportPath = "../../Logs/phase15_source_clip_sample.json";
        private static readonly float[] SampleTimes = { 0f, 0.5f, 1f };

        public static void Run()
        {
            var report = new SourceSampleReport();
            try
            {
                var sourceModel = AssetDatabase.LoadAssetAtPath<GameObject>(FbxPath);
                var sourceClip = AssetDatabase.LoadAllAssetsAtPath(FbxPath)
                    .OfType<AnimationClip>()
                    .FirstOrDefault(clip => clip.name == ClipName);
                if (sourceModel == null || sourceClip == null)
                    throw new InvalidOperationException("Generated source model or named clip did not load.");

                using var instance = new PrefabScope(sourceModel);
                var animator = instance.Object.GetComponent<Animator>();
                if (animator == null || animator.avatar == null || !animator.avatar.isValid || !animator.avatar.isHuman)
                    throw new InvalidOperationException("Generated source model does not have a valid Humanoid Animator.");

                animator.enabled = false;
                report.success = true;
                report.clipName = sourceClip.name;
                report.frameRate = sourceClip.frameRate;
                report.length = sourceClip.length;
                report.bindingCount = AnimationUtility.GetCurveBindings(sourceClip).Length;

                foreach (var time in SampleTimes)
                {
                    sourceClip.SampleAnimation(instance.Object, time);
                    report.samples.Add(new SourceSample
                    {
                        time = time,
                        rootPosition = instance.Object.transform.position,
                        hipsPosition = Position(animator, HumanBodyBones.Hips),
                        rightUpperArmPosition = Position(animator, HumanBodyBones.RightUpperArm),
                        rightLowerArmPosition = Position(animator, HumanBodyBones.RightLowerArm),
                        rightHandPosition = Position(animator, HumanBodyBones.RightHand),
                        rightFootPosition = Position(animator, HumanBodyBones.RightFoot)
                    });
                }

                report.rightHandDisplacement = MaxDistance(
                    report.samples.Select(sample => sample.rightHandPosition).ToArray());
                report.rightLowerArmDisplacement = MaxDistance(
                    report.samples.Select(sample => sample.rightLowerArmPosition).ToArray());
                report.rightUpperArmDisplacement = MaxDistance(
                    report.samples.Select(sample => sample.rightUpperArmPosition).ToArray());
                report.rootDisplacement = MaxDistance(
                    report.samples.Select(sample => sample.rootPosition).ToArray());
                report.nonTrivial = report.rightHandDisplacement > 0.01f ||
                    report.rightLowerArmDisplacement > 0.01f ||
                    report.rightUpperArmDisplacement > 0.01f;
            }
            catch (Exception exception)
            {
                report.success = false;
                report.error = exception.ToString();
            }

            WriteReport(JsonUtility.ToJson(report, true));
            Debug.Log("[Phase15SourceClipSampler] " + JsonUtility.ToJson(report));
            EditorApplication.Exit(report.success && report.nonTrivial ? 0 : 1);
        }

        private static Vector3 Position(Animator animator, HumanBodyBones bone)
        {
            var transform = animator.GetBoneTransform(bone);
            if (transform == null)
                throw new InvalidOperationException("Source Avatar is missing " + bone);
            return transform.position;
        }

        private static float MaxDistance(Vector3[] values)
        {
            var maximum = 0f;
            for (var first = 0; first < values.Length; first++)
            for (var second = first + 1; second < values.Length; second++)
                maximum = Mathf.Max(maximum, Vector3.Distance(values[first], values[second]));
            return maximum;
        }

        private static void WriteReport(string json)
        {
            var path = Path.GetFullPath(Path.Combine(Directory.GetCurrentDirectory(), ReportPath));
            Directory.CreateDirectory(Path.GetDirectoryName(path));
            File.WriteAllText(path, json);
        }

        private sealed class PrefabScope : IDisposable
        {
            public PrefabScope(GameObject original)
            {
                Object = UnityEngine.Object.Instantiate(original);
            }

            public GameObject Object { get; }

            public void Dispose()
            {
                if (Object != null)
                    UnityEngine.Object.DestroyImmediate(Object);
            }
        }

        [Serializable]
        private sealed class SourceSampleReport
        {
            public bool success;
            public bool nonTrivial;
            public string error;
            public string clipName;
            public float frameRate;
            public float length;
            public int bindingCount;
            public float rootDisplacement;
            public float rightUpperArmDisplacement;
            public float rightLowerArmDisplacement;
            public float rightHandDisplacement;
            public List<SourceSample> samples = new List<SourceSample>();
        }

        [Serializable]
        private sealed class SourceSample
        {
            public float time;
            public Vector3 rootPosition;
            public Vector3 hipsPosition;
            public Vector3 rightUpperArmPosition;
            public Vector3 rightLowerArmPosition;
            public Vector3 rightHandPosition;
            public Vector3 rightFootPosition;
        }
    }
}
