using System;
using System.IO;
using System.Linq;
using UnityEditor;
using UnityEditor.Animations;
using UnityEngine;

namespace RoomAgent.Editor
{
    public static class Phase15MotiusUnityValidator
    {
        const string FbxPath = "Assets/animation/phase15_motius/lacrimosa_motius_smpl22_animated.fbx";
        const string ReportPath = "Temp/Phase15UnityValidation.json";

        [MenuItem("Tools/Phase15/Validate Motius Unity Playback")]
        public static void Validate()
        {
            var importer = AssetDatabase.LoadAssetAtPath<ModelImporter>(FbxPath);
            if (importer == null)
                throw new InvalidOperationException("Generated Motius FBX not found at " + FbxPath);

            importer.animationType = ModelImporterAnimationType.Human;
            var description = importer.humanDescription;
            description.human = BuildHumanBones();
            description.skeleton = importer.humanDescription.skeleton;
            description.rootMotionBoneName = "Pelvis";
            importer.humanDescription = description;
            importer.SaveAndReimport();
            AssetDatabase.SaveAssets();

            var avatar = AssetDatabase.LoadAssetAtPath<Avatar>(FbxPath);
            var clips = AssetDatabase.LoadAllAssetsAtPath(FbxPath)
                .OfType<AnimationClip>()
                .Where(clip => !clip.legacy)
                .ToArray();
            if (avatar == null || !avatar.isValid || !avatar.isHuman)
                throw new InvalidOperationException("Generated Motius FBX does not contain a valid Humanoid avatar.");
            if (clips.Length == 0)
                throw new InvalidOperationException("Generated Motius FBX does not contain a Humanoid AnimationClip.");

            var clip = clips.First();
            if (clip.frameRate != 20f || Math.Abs(clip.length - 1f) > 0.01f)
                throw new InvalidOperationException(
                    $"Unexpected clip timing: fps={clip.frameRate}, length={clip.length}");

            var fbxPrefab = AssetDatabase.LoadAssetAtPath<GameObject>(FbxPath);
            var instance = UnityEngine.Object.Instantiate(fbxPrefab);
            instance.name = "Lacrimosa_Phase15_Motius";
            var animator = instance.GetComponentInChildren<Animator>();
            if (animator == null)
                throw new InvalidOperationException("Generated Motius FBX instance has no Animator.");
            animator.avatar = avatar;

            var controller = new AnimatorController();
            controller.AddLayer("Base");
            var state = controller.layers[0].stateMachine.AddState("MotiusWave");
            state.motion = clip;
            animator.runtimeAnimatorController = controller;
            animator.Play("MotiusWave", 0, 0f);
            animator.Update(0f);

            var sampleFrames = new[] { 0, 10, 19 };
            var positions = new float[sampleFrames.Length][][];
            for (var i = 0; i < sampleFrames.Length; i++)
            {
                var time = sampleFrames[i] / 20f;
                animator.Play("MotiusWave", 0, time / clip.length);
                animator.Update(0f);
                positions[i] = new[]
                {
                    ToFloatArray(GetPosition(instance, "Pelvis")),
                    ToFloatArray(GetPosition(instance, "Head")),
                    ToFloatArray(GetPosition(instance, "L_Wrist")),
                    ToFloatArray(GetPosition(instance, "R_Wrist"))
                };
            }

            var moved = positions
                .SelectMany(frame => frame)
                .Select((value, index) => (value, index))
                .ToDictionary(entry => entry.index, entry => entry.value);

            var report = new Phase15ValidationReport
            {
                fbx = FbxPath,
                animationType = "Humanoid",
                avatarValid = avatar.isValid,
                avatarHuman = avatar.isHuman,
                clip = clip.name,
                fps = clip.frameRate,
                frameCount = sampleFrames.Length,
                durationSeconds = clip.length,
                sampledPositions = positions,
                finite = positions.All(frame => frame.All(bones => bones.All(float.IsFinite))),
                nonTrivial = moved.Count > 0
            };
            var json = JsonUtility.ToJson(report);
            File.WriteAllText(Path.Combine(Directory.GetCurrentDirectory(), ReportPath), json + Environment.NewLine);

            UnityEngine.Object.DestroyImmediate(instance);
            UnityEngine.Object.DestroyImmediate(controller);
            Debug.Log("[Phase15MotiusUnityValidator] " + json);
        }

        static HumanBone[] BuildHumanBones()
        {
            var mappings = new[]
            {
                ("Pelvis", "Hips"),
                ("L_Hip", "LeftUpperLeg"),
                ("R_Hip", "RightUpperLeg"),
                ("Spine1", "Spine"),
                ("Spine2", "Chest"),
                ("Spine3", "UpperChest"),
                ("L_Knee", "LeftLowerLeg"),
                ("R_Knee", "RightLowerLeg"),
                ("L_Ankle", "LeftFoot"),
                ("R_Ankle", "RightFoot"),
                ("L_Foot", "LeftToes"),
                ("R_Foot", "RightToes"),
                ("Neck", "Neck"),
                ("L_Collar", "LeftShoulder"),
                ("R_Collar", "RightShoulder"),
                ("Head", "Head"),
                ("L_Shoulder", "LeftUpperArm"),
                ("R_Shoulder", "RightUpperArm"),
                ("L_Elbow", "LeftLowerArm"),
                ("R_Elbow", "RightLowerArm"),
                ("L_Wrist", "LeftHand"),
                ("R_Wrist", "RightHand")
            };

            return mappings.Select(mapping => new HumanBone
            {
                boneName = mapping.Item1,
                humanName = mapping.Item2
            }).ToArray();
        }

        static Vector3 GetPosition(GameObject instance, string boneName)
        {
            var transform = instance.GetComponentsInChildren<Transform>()
                .FirstOrDefault(candidate => candidate.name == boneName);
            if (transform == null)
                throw new InvalidOperationException("Missing bone transform: " + boneName);
            return transform.position;
        }

        static float[] ToFloatArray(Vector3 value)
        {
            return new[] { value.x, value.y, value.z };
        }

        [Serializable]
        class Phase15ValidationReport
        {
            public string fbx;
            public string animationType;
            public bool avatarValid;
            public bool avatarHuman;
            public string clip;
            public float fps;
            public int frameCount;
            public float durationSeconds;
            public float[][][] sampledPositions;
            public bool finite;
            public bool nonTrivial;
        }
    }
}
