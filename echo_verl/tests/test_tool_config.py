"""Validate echo_tool_config.yaml against verl's REAL pydantic tool schemas.

This is a 🟢 test, not a 🟡 contract review: it loads
`external/verl/verl/tools/schemas.py` (pinned submodule @ v0.7.1) directly by
path with importlib. That module is self-contained (json + typing + pydantic
only), so it imports without torch/vLLM and the round-trip is the same
validation `tool_registry.initialize_tools_from_config` performs at launch.

The thing under test: `OpenAIFunctionPropertySchema` declares no `ConfigDict`,
so pydantic's default `extra="ignore"` SILENTLY DROPS any property key beyond
type/description/enum (items, minItems, nested objects, oneOf). If the `op`
enum were dropped, the model would never be told the valid ops and every
rollout would mis-call the tool.
"""
import importlib.util
import sys
from pathlib import Path

import pytest

_REPO = Path(__file__).resolve().parents[2]
_SCHEMAS = _REPO / "external" / "verl" / "verl" / "tools" / "schemas.py"
_CONFIG = _REPO / "echo_verl" / "configs" / "echo_tool_config.yaml"

yaml = pytest.importorskip("yaml")
pytest.importorskip("pydantic")
if not _SCHEMAS.exists():
    pytest.skip("external/verl submodule not checked out", allow_module_level=True)


def _load_verl_schemas():
    spec = importlib.util.spec_from_file_location("_verl_schemas_ref", _SCHEMAS)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def schemas():
    return _load_verl_schemas()


@pytest.fixture(scope="module")
def parsed(schemas):
    cfg = yaml.safe_load(_CONFIG.read_text())
    return schemas.OpenAIFunctionToolSchema.model_validate(cfg["tools"][0]["tool_schema"])


def test_tool_name_is_echo(parsed):
    # Must match BaseTool.name -> dispatch key -> extra_info.tools_kwargs key
    # -> the name the SFT serializer emits in <tool_call>.
    assert parsed.function.name == "echo"


def test_op_enum_survives_pydantic_extra_ignore(parsed):
    op = parsed.function.parameters.properties["op"]
    assert op.type == "string"
    assert op.enum == ["select_view", "select_frames", "zoom"]


def test_advertised_props_and_required(parsed):
    assert set(parsed.function.parameters.properties) == {"op", "view_name"}
    assert parsed.function.parameters.required == ["op"]


def test_no_advertised_property_relies_on_dropped_keys(schemas):
    """Guard against someone adding items/nested props that pydantic eats silently."""
    cfg = yaml.safe_load(_CONFIG.read_text())
    props = cfg["tools"][0]["tool_schema"]["function"]["parameters"]["properties"]
    allowed = set(schemas.OpenAIFunctionPropertySchema.model_fields)
    for name, prop in props.items():
        assert set(prop) <= allowed, f"property {name!r} declares keys verl drops: {set(prop) - allowed}"


def test_class_name_points_at_echo_tool():
    cfg = yaml.safe_load(_CONFIG.read_text())
    assert cfg["tools"][0]["class_name"] == "echo_verl.echo_tool.EchoTool"
    assert cfg["tools"][0]["config"]["type"] == "native"


def test_tool_response_image_contract(schemas):
    """EchoTool.execute returns ToolResponse(image=[...]) -- a list is mandatory."""
    r = schemas.ToolResponse(image=["img"], text="obs")
    assert r.image == ["img"] and not r.is_empty()
    with pytest.raises(Exception):
        schemas.ToolResponse(image="not-a-list")
