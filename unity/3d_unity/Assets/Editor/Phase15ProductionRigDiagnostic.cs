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
    public static class Phase15ProductionRigDiagnostic
    {
        private const string ScenePath = "Assets/Scenes/SampleScene.unity";
        private const string ReportPath = "Temp/Phase15ProductionRig.json";

        public static void Run()
        {
            try
            {
                var scene = EditorSceneManager.OpenScene(ScenePath, OpenSceneMode.Single);
                if (!scene.isLoaded)
                    throw new InvalidOperationException("Production scene did not load.");

                var character = UnityEngine.Object.FindFirstObjectByType<LacrimosaController>();
                if (character == null)
                    throw new InvalidOperationException("Production LacrimosaController is missing.");

                var animator = character.GetComponent<Animator>();
                if (animator == null)
                    throw new InvalidOperationException("Production Lacrimosa has no Animator.");
                if (animator.avatar == null)
                    throw new InvalidOperationException("Production Lacrimosa has no Avatar.");

                var description = animator.avatar.humanDescription;
                var report = new RigReport
                {
                    success = true,
                    characterName = character.name,
                    avatarValid = animator.avatar.isValid,
                    avatarHuman = animator.avatar.isHuman,
                    rootMotionBoneName = description.rootMotionBoneName,
                    rootBone = animator.GetBoneTransform(HumanBodyBones.Hips) != null
                        ? GetTransformPath(animator.GetBoneTransform(HumanBodyBones.Hips).parent, character.transform)
                        : null,
                    controller = animator.runtimeAnimatorController != null
                        ? animator.runtimeAnimatorController.name
                        : null,
                    clips = animator.runtimeAnimatorController != null
                        ? animator.runtimeAnimatorController.animationClips
                            .Select(clip => clip.name)
                            .ToArray()
                        : Array.Empty<string>(),
                    mappings = description.human
                        .Select(bone => new BoneMapping
                        {
                            humanName = bone.humanName,
                            boneName = bone.boneName,
                            transformPath = GetTransformPath(
                                Enum.TryParse(bone.humanName, out HumanBodyBones humanBone)
                                    ? animator.GetBoneTransform(humanBone)
                                    : null,
                                character.transform)
                        })
                        .ToArray(),
                    hierarchy = BuildMappedHierarchy(animator, character.transform)
                };

                WriteReport(JsonUtility.ToJson(report, true));
            }
            catch (Exception exception)
            {
                WriteReport(JsonUtility.ToJson(new RigReport
                {
                    success = false,
                    error = exception.ToString()
                }, true));
                if (Application.isBatchMode)
                    EditorApplication.Exit(1);
            }
        }

        private static string GetTransformPath(Transform transform, Transform root)
        {
            if (transform == null)
                return null;
            var names = new List<string>();
            var current = transform;
            while (current != null && current != root)
            {
                names.Add(current.name);
                current = current.parent;
            }
            names.Reverse();
            return string.Join("/", names);
        }

        private static HierarchyNode[] BuildMappedHierarchy(Animator animator, Transform root)
        {
            var hierarchy = new Dictionary<Transform, HierarchyNode>();
            foreach (HumanBodyBones bone in Enum.GetValues(typeof(HumanBodyBones)))
            {
                if (bone == HumanBodyBones.LastBone)
                    continue;

                var transform = animator.GetBoneTransform(bone);
                while (transform != null && transform != root)
                {
                    if (!hierarchy.ContainsKey(transform))
                    {
                        hierarchy[transform] = new HierarchyNode
                        {
                            name = transform.name,
                            parentName = transform.parent != null && transform.parent != root
                                ? transform.parent.name
                                : null,
                            path = GetTransformPath(transform, root)
                        };
                    }
                    transform = transform.parent;
                }
            }
            return hierarchy.Values.OrderBy(node => node.path).ToArray();
        }

        private static void WriteReport(string json)
        {
            var path = Path.Combine(Directory.GetCurrentDirectory(), ReportPath);
            Directory.CreateDirectory(Path.GetDirectoryName(path));
            File.WriteAllText(path, json);
            Debug.Log("[Phase15ProductionRigDiagnostic] " + json);
        }

        [Serializable]
        private sealed class RigReport
        {
            public bool success;
            public string error;
            public string characterName;
            public bool avatarValid;
            public bool avatarHuman;
            public string rootMotionBoneName;
            public string rootBone;
            public string controller;
            public string[] clips;
            public BoneMapping[] mappings;
            public HierarchyNode[] hierarchy;
        }

        [Serializable]
        private sealed class BoneMapping
        {
            public string humanName;
            public string boneName;
            public string transformPath;
        }

        [Serializable]
        private sealed class HierarchyNode
        {
            public string name;
            public string parentName;
            public string path;
        }
    }
}
