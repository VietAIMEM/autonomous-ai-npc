using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using RoomAgent.Lacrimosa;
using UnityEditor;
using UnityEditor.SceneManagement;
using UnityEngine;

namespace RoomAgent.Editor
{
    [InitializeOnLoad]
    public static class Phase17RigSamplingDiagnostic
    {
        private const string ScenePath = "Assets/Scenes/SampleScene.unity";
        private const string RequestPath = "Temp/Phase17RigSampling.request";
        private const string ReportPath = "Temp/Phase17RigSampling.json";

        static Phase17RigSamplingDiagnostic()
        {
            EditorApplication.delayCall += RunRequested;
        }

        private static void RunRequested()
        {
            var path = Path.Combine(Directory.GetCurrentDirectory(), RequestPath);
            if (!File.Exists(path))
                return;

            var request = JsonUtility.FromJson<RigSamplingRequest>(File.ReadAllText(path));
            File.Delete(path);
            Run(request.sourceReference ?? "");
        }

        private static void Run(string sourceReference)
        {
            var report = new RigSamplingReport
            {
                sourceReference = sourceReference ?? "",
                success = false
            };

            GameObject sourceClone = null;
            GameObject targetClone = null;
            try
            {
                if (string.IsNullOrWhiteSpace(sourceReference) ||
                    !sourceReference.StartsWith("Assets/", StringComparison.Ordinal))
                    throw new InvalidOperationException("INVALID_SOURCE_REFERENCE");

                var scene = EditorSceneManager.OpenScene(ScenePath, OpenSceneMode.Single);
                if (!scene.isLoaded)
                    throw new InvalidOperationException("PRODUCTION_SCENE_MISSING");

                var character = UnityEngine.Object.FindFirstObjectByType<LacrimosaController>();
                if (character == null)
                    throw new InvalidOperationException("PRODUCTION_RIG_MISSING");

                var sourceModel = AssetDatabase.LoadAssetAtPath<GameObject>(sourceReference);
                if (sourceModel == null)
                    throw new InvalidOperationException("SOURCE_MODEL_UNRESOLVED");

                var sourceAvatar = AssetDatabase.LoadAllAssetsAtPath(sourceReference)
                    .OfType<Avatar>()
                    .FirstOrDefault();
                if (sourceAvatar == null || !sourceAvatar.isValid || !sourceAvatar.isHuman)
                    throw new InvalidOperationException("INVALID_SOURCE_AVATAR");

                var sourceClip = AssetDatabase.LoadAllAssetsAtPath(sourceReference)
                    .OfType<AnimationClip>()
                    .FirstOrDefault();
                if (sourceClip == null || sourceClip.length <= 0f)
                    throw new InvalidOperationException("SOURCE_CLIP_UNRESOLVED");

                sourceClone = UnityEngine.Object.Instantiate(sourceModel);
                sourceClone.name = "Phase17_Source_Rig_Sample";
                sourceClone.transform.position = Vector3.zero;
                sourceClone.transform.rotation = Quaternion.identity;
                var sourceAnimator = sourceClone.GetComponent<Animator>();
                if (sourceAnimator == null)
                    sourceAnimator = sourceClone.AddComponent<Animator>();
                sourceAnimator.avatar = sourceAvatar;
                sourceAnimator.applyRootMotion = false;
                sourceAnimator.enabled = true;

                targetClone = UnityEngine.Object.Instantiate(character.gameObject);
                targetClone.name = "Phase17_Target_Rig_Sample";
                targetClone.transform.position = Vector3.zero;
                targetClone.transform.rotation = Quaternion.identity;
                var targetAnimator = targetClone.GetComponent<Animator>();
                if (targetAnimator == null || targetAnimator.avatar == null ||
                    !targetAnimator.avatar.isValid || !targetAnimator.avatar.isHuman)
                    throw new InvalidOperationException("INVALID_TARGET_RIG");
                targetAnimator.applyRootMotion = false;
                targetAnimator.enabled = true;

                var sampleTimes = new[]
                {
                    0f,
                    sourceClip.length * 0.25f,
                    sourceClip.length * 0.5f,
                    sourceClip.length * 0.75f,
                    sourceClip.length
                };
                report.sampleTimes = sampleTimes;
                report.sourceSamples = SampleRig(sourceClone, sourceClip, sampleTimes);
                report.targetSamples = SampleRig(targetClone, sourceClip, sampleTimes);
                report.sourceFootMinimumY = report.sourceSamples
                    .SelectMany(sample => sample.bones)
                    .Where(sample => sample.bone == "LeftFoot" || sample.bone == "RightFoot")
                    .Min(sample => sample.position.y);
                report.targetFootMinimumY = report.targetSamples
                    .SelectMany(sample => sample.bones)
                    .Where(sample => sample.bone == "LeftFoot" || sample.bone == "RightFoot")
                    .Min(sample => sample.position.y);
                report.success = true;
            }
            catch (Exception exception)
            {
                report.error = exception.ToString();
            }
            finally
            {
                if (sourceClone != null)
                    UnityEngine.Object.DestroyImmediate(sourceClone);
                if (targetClone != null)
                    UnityEngine.Object.DestroyImmediate(targetClone);
                WriteReport(report);
            }
        }

        private static RigSample[] SampleRig(
            GameObject rig,
            AnimationClip clip,
            float[] sampleTimes)
        {
            var animator = rig.GetComponent<Animator>();
            var samples = new List<RigSample>();
            AnimationMode.StartAnimationMode();
            try
            {
                foreach (var time in sampleTimes)
                {
                    AnimationMode.SampleAnimationClip(rig, clip, time);
                    samples.Add(new RigSample
                    {
                        time = time,
                        bones = CaptureBones(animator, rig.transform)
                    });
                }
            }
            finally
            {
                AnimationMode.StopAnimationMode();
            }
            return samples.ToArray();
        }

        private static BoneSample[] CaptureBones(Animator animator, Transform root)
        {
            return new[]
            {
                CaptureBone(animator, HumanBodyBones.Hips, root),
                CaptureBone(animator, HumanBodyBones.Head, root),
                CaptureBone(animator, HumanBodyBones.LeftHand, root),
                CaptureBone(animator, HumanBodyBones.RightHand, root),
                CaptureBone(animator, HumanBodyBones.LeftFoot, root),
                CaptureBone(animator, HumanBodyBones.RightFoot, root)
            };
        }

        private static BoneSample CaptureBone(
            Animator animator,
            HumanBodyBones bone,
            Transform root)
        {
            var transform = animator.GetBoneTransform(bone);
            if (transform == null)
                throw new InvalidOperationException("SAMPLE_BONE_MISSING:" + bone);
            return new BoneSample
            {
                bone = bone.ToString(),
                position = transform.position - root.position,
                rotation = transform.rotation
            };
        }

        private static void WriteReport(RigSamplingReport report)
        {
            var path = Path.Combine(Directory.GetCurrentDirectory(), ReportPath);
            Directory.CreateDirectory(Path.GetDirectoryName(path));
            File.WriteAllText(path, JsonUtility.ToJson(report, true));
            Debug.Log("[Phase17RigSamplingDiagnostic] " + JsonUtility.ToJson(report));
        }

        [Serializable]
        private sealed class RigSamplingRequest
        {
            public string sourceReference;
        }

        [Serializable]
        private sealed class RigSamplingReport
        {
            public bool success;
            public string error;
            public string sourceReference;
            public float[] sampleTimes;
            public RigSample[] sourceSamples;
            public RigSample[] targetSamples;
            public float sourceFootMinimumY;
            public float targetFootMinimumY;
        }

        [Serializable]
        private sealed class RigSample
        {
            public float time;
            public BoneSample[] bones;
        }

        [Serializable]
        private sealed class BoneSample
        {
            public string bone;
            public Vector3 position;
            public Quaternion rotation;
        }
    }
}
