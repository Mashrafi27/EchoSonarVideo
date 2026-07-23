from echo_rl.config import Config
from echo_rl.data.studies import Clip
from echo_rl.data.trajectory import overview_turn, build_trajectory


def _clips():
    return [
        Clip("di-1", "PLAX Standard", "/d/di-1_PLAX Standard", [f"{i}.png" for i in range(10)]),
        Clip("di-2", "A4C", "/d/di-2_A4C", [f"{i}.png" for i in range(20)]),
    ]


def _thinking():
    return ("#### 1. **PLAX Standard View**\n- **Clinical Findings:**\n  - LV normal.\n"
            "#### 2. **A4C View**\n- **Clinical Findings:**\n  - RV normal.\n"
            "#### 3. **Subcostal View**\n- **Clinical Findings:**\n  - IVC normal.\n")


def test_overview_turn():
    cfg = Config.from_env()
    ov = overview_turn(_clips(), cfg)
    assert len(ov["views"]) == 2
    assert ov["views"][0]["frame"].endswith("/5.png")   # midframe of 10
    assert ov["views"][1]["frame_count"] == 20


def test_build_trajectory_maps_available_views():
    cfg = Config.from_env()
    traj = build_trajectory("Describe LV.", "LV normal.", _thinking(), _clips(), cfg)
    tool_views = [t["args"]["view"] for t in traj["turns"] if t["type"] == "tool"]
    assert tool_views == ["PLAX Standard", "A4C"]      # Subcostal has no clip -> skipped
    assert traj["answer"] == "LV normal."
    first_obs = next(t for t in traj["turns"] if t["type"] == "tool")["obs"]
    assert len(first_obs["frames"]) == cfg.n_preview_frames
    assert any(t["type"] == "think" and "LV normal" in t["text"] for t in traj["turns"])


def test_build_trajectory_no_map():
    cfg = Config.from_env()
    traj = build_trajectory("q", "a", "#### 1. **Suprasternal View**\n- **Clinical Findings:**\n  - x\n",
                            _clips(), cfg)
    assert traj["turns"] == [] and traj["answer"] == "a"
