using NUnit.Framework;
using RoomAgent.Brain;

namespace RoomAgent.EditorTests
{
    [TestFixture]
    public sealed class BrainIsolationPolicyTests
    {
        [Test]
        public void ProductionModeKeepsTheExistingPath()
        {
            Assert.IsTrue(BrainIsolationPolicy.TryAuthorize(
                false,
                "http://127.0.0.1:8766",
                "START_STREAM",
                out _));
        }

        [Test]
        public void TestModeWithoutConfigurationFailsClosed()
        {
            Assert.IsFalse(BrainIsolationPolicy.TryAuthorize(
                true,
                "",
                "LOOK_AT",
                out var reason));
            Assert.AreEqual("e2e_isolation_config_missing", reason);
        }

        [Test]
        public void InvalidTestConfigurationFailsClosed()
        {
            Assert.IsFalse(BrainIsolationPolicy.TryAuthorize(
                true,
                "not-a-url",
                "LOOK_AT",
                out var reason));
            Assert.AreEqual("e2e_isolation_config_invalid", reason);
        }

        [TestCase("http://127.0.0.1:8766")]
        [TestCase("http://localhost:8766")]
        [TestCase("http://127.0.0.1:8766/")]
        public void ProductionEquivalentTestConfigurationFailsClosed(string streamBaseUrl)
        {
            Assert.IsFalse(BrainIsolationPolicy.TryAuthorize(
                true,
                streamBaseUrl,
                "LOOK_AT",
                out var reason));
            Assert.AreEqual("e2e_isolation_config_production_equivalent", reason);
        }

        [TestCase("START_STREAM")]
        [TestCase("start_stream")]
        [TestCase("STOP_STREAM")]
        [TestCase("stop_stream")]
        public void StreamControlActionsAreDeniedInTestMode(string action)
        {
            Assert.IsFalse(BrainIsolationPolicy.TryAuthorize(
                true,
                "http://127.0.0.1:18766",
                action,
                out var reason));
            Assert.AreEqual("e2e_stream_action_denied", reason);
        }

        [TestCase("MOVE_TO")]
        [TestCase("LOOK_AT")]
        [TestCase("SPEAK")]
        [TestCase("PICK_UP_PHONE")]
        public void SafeNonStreamingActionsUseTheIntendedTestPath(string action)
        {
            Assert.IsTrue(BrainIsolationPolicy.TryAuthorize(
                true,
                "http://127.0.0.1:18766",
                action,
                out _));
        }

        [Test]
        public void MaintainStreamStartCannotReachProduction()
        {
            Assert.IsFalse(BrainIsolationPolicy.TryAuthorize(
                true,
                "http://127.0.0.1:18766",
                "START_STREAM",
                out var reason));
            Assert.AreEqual("e2e_stream_action_denied", reason);
        }
    }
}
