using System;
using System.Collections;
using System.Collections.Generic;
using System.Text;
using UnityEditor;
using UnityEditor.SceneManagement;
using UnityEngine;
using UnityEngine.AI;
using Unity.AI.Navigation;
using RoomAgent.Navigation;
using Object = UnityEngine.Object;

namespace RoomAgent.EditorTools
{
    public static class Phase01WorldBuilder
    {
        const string ScenePath = "Assets/Scenes/SampleScene.unity";
        const string FbxPath = "Assets/model/lacrimosa-animated-neverness-to-everness/source/lacrimosa final.fbx";
        const string MaterialsPath = "Assets/Materials";
        const string NavMeshDataPath = "Assets/Scenes/NavMesh-RoomNavMeshSurface.asset";
        const string ValidationRootName = "Navigation Validation";

        [MenuItem("Room Agent/Phase 01/Build Room Foundation")]
        public static void BuildRoomFoundation()
        {
            if (EditorApplication.isPlaying)
            {
                Debug.LogError("Stop Play Mode before building the Phase 01 scene.");
                return;
            }

            try
            {
                EnsureFolders();
                var scene = EditorSceneManager.OpenScene(ScenePath, OpenSceneMode.Single);
                var environment = GetOrCreateRoot("Environment");
                var characters = GetOrCreateRoot("Characters");
                var navigationRoot = GetOrCreateRoot("Navigation");

                var floor = BuildFloor(environment);
                var walls = BuildWalls(environment);
                var bench = BuildBench(environment);
                var table = BuildTable(environment);
                var character = BuildCharacter(characters, floor);
                var navigation = BuildNavigation(navigationRoot, character);

                PositionMainCamera(character);
                EditorSceneManager.SaveScene(scene);
                AssetDatabase.SaveAssets();
                AssetDatabase.Refresh();

                var summary = new StringBuilder()
                    .AppendLine("PHASE 01 scene built and saved.")
                    .AppendLine("Character renderer bounds: " + GetRendererBounds(character).ToString("F3"))
                    .AppendLine("Character spawn: " + character.transform.position.ToString("F3"));
                Debug.Log(summary, character);
            }
            catch (Exception exception)
            {
                Debug.LogException(exception);
            }
        }

        [MenuItem("Room Agent/Phase 01/Validate Room Foundation")]
        public static void ValidateRoomFoundation()
        {
            if (EditorApplication.isPlaying)
            {
                Debug.LogError("Stop Play Mode before running editor validation.");
                return;
            }

            var scene = EditorSceneManager.OpenScene(ScenePath, OpenSceneMode.Single);
            var environment = GameObject.Find("Environment");
            var characters = GameObject.Find("Characters");
            var navigationRoot = GameObject.Find("Navigation");

            var checks = new List<string>();
            var pass = true;

            if (environment == null || characters == null || navigationRoot == null)
            {
                pass = false;
                checks.Add("Missing one of Environment/Characters/Navigation roots.");
            }

            var requiredNames = new[] { "Floor", "Wall_North", "Wall_South", "Wall_East", "Wall_West", "Bench", "Table", "Lacrimosa" };
            foreach (var name in requiredNames)
            {
                var found = FindByName(name, environment, characters);
                if (found == null)
                {
                    pass = false;
                    checks.Add("Missing " + name + ".");
                }
            }

            var floor = FindByName("Floor", environment, null);
            if (floor != null)
            {
                var collider = floor.GetComponent<Collider>();
                var valid = collider != null && floor.transform.position == Vector3.zero;
                EditorUtility.SetDirty(floor);
                pass &= valid;
                checks.Add((valid ? "PASS" : "FAIL") + " Floor collider/placement: " + (collider != null ? collider.ToString() : "none"));
            }

            foreach (var direction in new[] { "North", "South", "East", "West" })
            {
                var wall = FindByName("Wall_" + direction, environment, null);
                var valid = wall != null && wall.GetComponent<Collider>() != null;
                if (wall != null)
                    EditorUtility.SetDirty(wall);
                pass &= valid;
                checks.Add((valid ? "PASS" : "FAIL") + " Wall_" + direction + " collider.");
            }

            foreach (var furniture in new[] { "Bench", "Table" })
            {
                var item = FindByName(furniture, environment, null);
                var colliders = item != null ? item.GetComponentsInChildren<Collider>() : Array.Empty<Collider>();
                var valid = colliders.Length > 0;
                if (item != null)
                    EditorUtility.SetDirty(item);
                pass &= valid;
                checks.Add((valid ? "PASS" : "FAIL") + " " + furniture + " colliders: " + colliders.Length + ".");
            }

            var character = FindByName("Lacrimosa", environment, characters);
            if (character != null)
            {
                var agent = character.GetComponentInChildren<NavMeshAgent>();
                var walker = character.GetComponent<NavMeshWalker>();
                var renderers = character.GetComponentsInChildren<Renderer>();
                var bounds = GetRendererBounds(character);
                var valid = agent != null && walker != null && renderers.Length > 0;
                pass &= valid;
                checks.Add((valid ? "PASS" : "FAIL") + " Lacrimosa agent/walker/renderers (" + renderers.Length + ").");
                checks.Add("Lacrimosa bounds: " + bounds.ToString("F3"));
            }

            var surface = navigationRoot != null ? navigationRoot.GetComponentInChildren<NavMeshSurface>() : null;
            if (surface != null)
            {
                var valid = surface.navMeshData != null;
                pass &= valid;
                checks.Add((valid ? "PASS" : "FAIL") + " NavMesh data asset.");
            }

            var message = "PHASE 01 editor validation: " + (pass ? "PASS" : "FAIL") + Environment.NewLine + string.Join(Environment.NewLine, checks);
            if (pass)
                Debug.Log(message);
            else
                Debug.LogError(message);
        }

        static void EnsureFolders()
        {
            if (!AssetDatabase.IsValidFolder("Assets/Editor"))
                AssetDatabase.CreateFolder("Assets", "Editor");
            if (!AssetDatabase.IsValidFolder(MaterialsPath))
                AssetDatabase.CreateFolder("Assets", "Materials");
        }

        static GameObject GetOrCreateRoot(string name)
        {
            var existing = GameObject.Find(name);
            if (existing != null)
                return existing;

            var root = new GameObject(name);
            Undo.RegisterCreatedObjectUndo(root, "Create " + name);
            return root;
        }

        static GameObject CreatePrimitive(string name, PrimitiveType type, Transform parent, Vector3 localPosition, Vector3 localSize, Material material)
        {
            var existing = FindDirectChild(parent, name);
            var primitive = existing != null ? existing : GameObject.CreatePrimitive(type);
            if (existing == null)
            {
                primitive.transform.SetParent(parent, false);
                Undo.RegisterCreatedObjectUndo(primitive, "Create " + name);
            }
            primitive.name = name;
            primitive.transform.localPosition = localPosition;
            primitive.transform.localRotation = Quaternion.identity;
            primitive.transform.localScale = localSize;

            foreach (var collider in primitive.GetComponentsInChildren<Collider>())
                Object.DestroyImmediate(collider);

            var boxCollider = primitive.AddComponent<BoxCollider>();
            boxCollider.size = Vector3.one;
            boxCollider.center = Vector3.zero;

            var meshRenderer = primitive.GetComponent<MeshRenderer>();
            meshRenderer.sharedMaterial = material;
            meshRenderer.shadowCastingMode = UnityEngine.Rendering.ShadowCastingMode.On;
            meshRenderer.receiveShadows = true;

            return primitive;
        }

        static GameObject BuildFloor(GameObject environment)
        {
            var floor = CreatePrimitive("Floor", PrimitiveType.Cube, environment.transform,
                Vector3.zero, new Vector3(12f, 0.1f, 12f), LoadMaterial("RoomFloor", Color.gray));
            floor.isStatic = true;
            return floor;
        }

        static GameObject BuildWalls(GameObject environment)
        {
            var glass = LoadGlassMaterial();
            var existing = FindDirectChild(environment.transform, "GlassWalls");
            if (existing == null)
                existing = new GameObject("GlassWalls");
            existing.transform.SetParent(environment.transform, false);
            existing.transform.localPosition = Vector3.zero;

            CreatePrimitive("Wall_North", PrimitiveType.Cube, existing.transform,
                new Vector3(0f, 1.6f, -6f), new Vector3(12f, 3.2f, 0.1f), glass);
            CreatePrimitive("Wall_South", PrimitiveType.Cube, existing.transform,
                new Vector3(0f, 1.6f, 6f), new Vector3(12f, 3.2f, 0.1f), glass);
            CreatePrimitive("Wall_East", PrimitiveType.Cube, existing.transform,
                new Vector3(6f, 1.6f, 0f), new Vector3(0.1f, 3.2f, 12f), glass);
            CreatePrimitive("Wall_West", PrimitiveType.Cube, existing.transform,
                new Vector3(-6f, 1.6f, 0f), new Vector3(0.1f, 3.2f, 12f), glass);

            return existing;
        }

        static GameObject BuildBench(GameObject environment)
        {
            var bench = FindDirectChild(environment.transform, "Bench") ?? new GameObject("Bench");
            bench.transform.SetParent(environment.transform, false);
            bench.transform.localPosition = Vector3.zero;
            MarkFurnitureNotWalkable(bench);

            var wood = LoadMaterial("FurnitureWood", new Color(0.45f, 0.35f, 0.28f));
            var metal = LoadMaterial("FurnitureMetal", new Color(0.65f, 0.67f, 0.70f));

            CreatePrimitive("Seat", PrimitiveType.Cube, bench.transform,
                new Vector3(-4.7f, 0.44f, 0f), new Vector3(0.6f, 0.08f, 8f), wood);
            CreatePrimitive("Backrest", PrimitiveType.Cube, bench.transform,
                new Vector3(-4.98f, 0.78f, 0f), new Vector3(0.06f, 0.68f, 8f), wood);
            CreatePrimitive("Leg_North", PrimitiveType.Cube, bench.transform,
                new Vector3(-4.7f, 0.20f, -3.5f), new Vector3(0.5f, 0.4f, 0.08f), metal);
            CreatePrimitive("Leg_South", PrimitiveType.Cube, bench.transform,
                new Vector3(-4.7f, 0.20f, 3.5f), new Vector3(0.5f, 0.4f, 0.08f), metal);

            foreach (Transform child in bench.transform)
                child.gameObject.isStatic = true;

            return bench;
        }

        static GameObject BuildTable(GameObject environment)
        {
            var table = FindDirectChild(environment.transform, "Table") ?? new GameObject("Table");
            table.transform.SetParent(environment.transform, false);
            table.transform.localPosition = Vector3.zero;
            MarkFurnitureNotWalkable(table);

            var topMaterial = LoadMaterial("TableTop", new Color(0.78f, 0.79f, 0.81f));
            var legMaterial = LoadMaterial("FurnitureMetal", new Color(0.65f, 0.67f, 0.70f));

            CreatePrimitive("Top", PrimitiveType.Cube, table.transform,
                new Vector3(-3.1f, 0.735f, 0f), new Vector3(1.4f, 0.05f, 0.8f), topMaterial);
            CreatePrimitive("Leg_NorthWest", PrimitiveType.Cube, table.transform,
                new Vector3(-3.68f, 0.355f, -0.31f), new Vector3(0.06f, 0.71f, 0.06f), legMaterial);
            CreatePrimitive("Leg_NorthEast", PrimitiveType.Cube, table.transform,
                new Vector3(-2.52f, 0.355f, -0.31f), new Vector3(0.06f, 0.71f, 0.06f), legMaterial);
            CreatePrimitive("Leg_SouthWest", PrimitiveType.Cube, table.transform,
                new Vector3(-3.68f, 0.355f, 0.31f), new Vector3(0.06f, 0.71f, 0.06f), legMaterial);
            CreatePrimitive("Leg_SouthEast", PrimitiveType.Cube, table.transform,
                new Vector3(-2.52f, 0.355f, 0.31f), new Vector3(0.06f, 0.71f, 0.06f), legMaterial);

            foreach (Transform child in table.transform)
                child.gameObject.isStatic = true;

            return table;
        }

        static void MarkFurnitureNotWalkable(GameObject furniture)
        {
            var modifier = furniture.GetComponent<NavMeshModifier>();
            if (modifier == null)
                modifier = furniture.AddComponent<NavMeshModifier>();

            modifier.overrideArea = true;
            modifier.area = 1; // Built-in Not Walkable area.
            EditorUtility.SetDirty(modifier);
        }

        static GameObject BuildCharacter(GameObject characters, GameObject floor)
        {
            var fbx = AssetDatabase.LoadAssetAtPath<GameObject>(FbxPath);
            if (fbx == null)
                throw new InvalidOperationException("Lacrimosa FBX not found at " + FbxPath);

            var existing = FindDirectChild(characters.transform, "Lacrimosa");
            if (existing == null)
            {
                existing = (GameObject)PrefabUtility.InstantiatePrefab(fbx, characters.transform);
                existing.name = "Lacrimosa";
            }

            var bounds = GetRendererBounds(existing);
            existing.transform.localScale = Vector3.one;
            bounds = GetRendererBounds(existing);
            var characterHeight = 1.65f;
            var scale = characterHeight / bounds.size.y;
            existing.transform.localRotation = Quaternion.identity;
            existing.transform.localScale = Vector3.one * scale;

            bounds = GetRendererBounds(existing);
            var floorTop = floor.transform.position.y + floor.transform.localScale.y * 0.5f;
            var spawn = new Vector3(2f, 0f, 3.5f);
            existing.transform.position += new Vector3(
                spawn.x - bounds.center.x,
                floorTop - bounds.min.y,
                spawn.z - bounds.center.z);

            foreach (var rigidbody in existing.GetComponentsInChildren<Rigidbody>())
                Object.DestroyImmediate(rigidbody);
            foreach (var collider in existing.GetComponentsInChildren<Collider>())
                Object.DestroyImmediate(collider);

            var agent = existing.GetComponent<NavMeshAgent>();
            if (agent == null)
                agent = existing.AddComponent<NavMeshAgent>();
            agent.radius = Mathf.Max(0.22f, bounds.extents.x * 0.25f);
            agent.height = characterHeight;
            agent.baseOffset = 0f;
            agent.speed = 1.5f;
            agent.angularSpeed = 240f;
            agent.acceleration = 6f;
            agent.stoppingDistance = 0.3f;
            agent.autoBraking = true;
            agent.autoRepath = true;
            agent.updatePosition = true;
            agent.updateRotation = true;

            if (existing.GetComponent<NavMeshWalker>() == null)
                existing.AddComponent<NavMeshWalker>();

            var capsule = existing.GetComponent<CapsuleCollider>();
            if (capsule == null)
                capsule = existing.AddComponent<CapsuleCollider>();
            capsule.radius = agent.radius;
            capsule.height = characterHeight;
            capsule.center = new Vector3(0f, characterHeight * 0.5f, 0f);
            capsule.direction = 1;

            return existing;
        }

        static GameObject BuildNavigation(GameObject navigationRoot, GameObject character)
        {
            var surfaceObject = FindDirectChild(navigationRoot.transform, "RoomNavMeshSurface");
            if (surfaceObject == null)
            {
                surfaceObject = new GameObject("RoomNavMeshSurface");
                surfaceObject.transform.SetParent(navigationRoot.transform, false);
                Undo.RegisterCreatedObjectUndo(surfaceObject, "Create RoomNavMeshSurface");
            }

            var surface = surfaceObject.GetComponent<NavMeshSurface>();
            if (surface == null)
                surface = surfaceObject.AddComponent<NavMeshSurface>();

            surface.collectObjects = CollectObjects.Volume;
            surface.size = new Vector3(12f, 2f, 12f);
            surface.center = new Vector3(0f, 0.4f, 0f);
            surface.layerMask = ~0;
            surface.useGeometry = NavMeshCollectGeometry.RenderMeshes;
            surface.defaultArea = 0;
            surface.ignoreNavMeshAgent = true;
            surface.ignoreNavMeshObstacle = true;
            surface.overrideVoxelSize = false;
            surface.overrideTileSize = false;
            surface.minRegionArea = 1f;
            surface.buildHeightMesh = false;

            if (AssetDatabase.LoadAssetAtPath<NavMeshData>(NavMeshDataPath) != null)
                AssetDatabase.DeleteAsset(NavMeshDataPath);

            EditorUtility.SetDirty(surface);
            surface.BuildNavMesh();

            AssetDatabase.CreateAsset(surface.navMeshData, NavMeshDataPath);
            EditorUtility.SetDirty(surface);
            AssetDatabase.SaveAssets();

            var validation = FindDirectChild(navigationRoot.transform, ValidationRootName);
            if (validation == null)
            {
                validation = new GameObject(ValidationRootName);
                validation.transform.SetParent(navigationRoot.transform, false);
                Undo.RegisterCreatedObjectUndo(validation, "Create " + ValidationRootName);
            }

            validation.transform.localPosition = Vector3.zero;
            CreateWaypoint(validation.transform, "Spawn", character.transform.position);
            CreateWaypoint(validation.transform, "Center", new Vector3(0f, 0.05f, 0f));
            CreateWaypoint(validation.transform, "NearTable", new Vector3(-1.6f, 0.05f, 0f));
            CreateWaypoint(validation.transform, "NearBench", new Vector3(-3.5f, 0.05f, 1.2f));
            CreateWaypoint(validation.transform, "OppositeSide", new Vector3(3.5f, 0.05f, -3.5f));
            CreateWaypoint(validation.transform, "NearGlassWall", new Vector3(4.5f, 0.05f, 4.5f));

            var driver = navigationRoot.GetComponent<NavigationTestDriver>();
            if (driver == null)
                driver = navigationRoot.AddComponent<NavigationTestDriver>();
            var walker = character.GetComponent<NavMeshWalker>();

            var serializedDriver = new SerializedObject(driver);
            serializedDriver.FindProperty("walker").objectReferenceValue = walker;
            serializedDriver.FindProperty("autoCycle").boolValue = true;
            serializedDriver.FindProperty("dwellSeconds").floatValue = 1f;
            serializedDriver.FindProperty("stoppingDistance").floatValue = 0.3f;
            serializedDriver.FindProperty("clickToMove").boolValue = true;
            var waypoints = GetWaypoints(validation.transform);
            var waypointProperty = serializedDriver.FindProperty("waypoints");
            waypointProperty.arraySize = waypoints.Length;
            for (var i = 0; i < waypoints.Length; i++)
                waypointProperty.GetArrayElementAtIndex(i).objectReferenceValue = waypoints[i];
            serializedDriver.ApplyModifiedProperties();
            EditorUtility.SetDirty(driver);

            return surfaceObject;
        }

        static void CreateWaypoint(Transform parent, string name, Vector3 position)
        {
            var waypoint = FindDirectChild(parent, name);
            if (waypoint == null)
                waypoint = new GameObject(name);
            waypoint.transform.SetParent(parent, false);
            waypoint.transform.localPosition = position;
        }

        static Transform[] GetWaypoints(Transform parent)
        {
            var result = new List<Transform>();
            foreach (Transform child in parent)
                result.Add(child);
            return result.ToArray();
        }

        static void PositionMainCamera(GameObject character)
        {
            var camera = Camera.main;
            if (camera == null)
                return;

            var transform = camera.transform;
            var characterPosition = character != null ? character.transform.position : Vector3.zero;
            transform.position = characterPosition + new Vector3(4.5f, 1.8f, 4.5f);
            transform.rotation = Quaternion.identity;
            transform.LookAt(characterPosition + Vector3.up * 0.8f);
            camera.farClipPlane = 100f;
            Selection.activeGameObject = camera.gameObject;
            SceneView.lastActiveSceneView?.FrameSelected();
        }

        static Material LoadMaterial(string name, Color color)
        {
            var path = MaterialsPath + "/" + name + ".mat";
            var material = AssetDatabase.LoadAssetAtPath<Material>(path);
            if (material == null)
            {
                material = new Material(Shader.Find("Universal Render Pipeline/Lit"));
                AssetDatabase.CreateAsset(material, path);
            }

            if (material.HasProperty("_BaseColor"))
                material.SetColor("_BaseColor", color);
            if (material.HasProperty("_Smoothness"))
                material.SetFloat("_Smoothness", 0.25f);
            EditorUtility.SetDirty(material);
            return material;
        }

        static Material LoadGlassMaterial()
        {
            const string path = MaterialsPath + "/RoomGlass.mat";
            var material = AssetDatabase.LoadAssetAtPath<Material>(path);
            if (material == null)
            {
                material = new Material(Shader.Find("Universal Render Pipeline/Lit"));
                AssetDatabase.CreateAsset(material, path);
            }

            if (material.HasProperty("_BaseColor"))
                material.SetColor("_BaseColor", new Color(0.62f, 0.85f, 0.92f, 0.32f));
            if (material.HasProperty("_Smoothness"))
                material.SetFloat("_Smoothness", 0.75f);
            if (material.HasProperty("_Surface"))
                material.SetFloat("_Surface", 1f);
            if (material.HasProperty("_Blend"))
                material.SetFloat("_Blend", 0f);
            if (material.HasProperty("_SrcBlend"))
                material.SetFloat("_SrcBlend", (float)UnityEngine.Rendering.BlendMode.SrcAlpha);
            if (material.HasProperty("_DstBlend"))
                material.SetFloat("_DstBlend", (float)UnityEngine.Rendering.BlendMode.OneMinusSrcAlpha);
            if (material.HasProperty("_ZWrite"))
                material.SetFloat("_ZWrite", 0f);
            material.renderQueue = (int)UnityEngine.Rendering.RenderQueue.Transparent;
            material.SetOverrideTag("RenderType", "Transparent");
            material.EnableKeyword("_SURFACE_TYPE_TRANSPARENT");
            material.EnableKeyword("_BLENDMODE_ALPHA");
            EditorUtility.SetDirty(material);
            return material;
        }

        static GameObject FindByName(string name, GameObject first, GameObject second)
        {
            return FindByName(name, first) ?? FindByName(name, second);
        }

        static GameObject FindByName(string name, GameObject root)
        {
            if (root == null)
                return GameObject.Find(name);

            foreach (Transform transform in root.GetComponentsInChildren<Transform>(true))
                if (transform.name == name)
                    return transform.gameObject;
            return null;
        }

        static GameObject FindDirectChild(Transform parent, string name)
        {
            foreach (Transform child in parent)
                if (child.name == name)
                    return child.gameObject;
            return null;
        }

        static Bounds GetRendererBounds(GameObject target)
        {
            var bounds = new Bounds();
            foreach (var renderer in target.GetComponentsInChildren<Renderer>())
            {
                if (!renderer.enabled)
                    continue;
                if (bounds.size == Vector3.zero)
                    bounds = renderer.bounds;
                else
                    bounds.Encapsulate(renderer.bounds);
            }
            return bounds;
        }

    }
}
