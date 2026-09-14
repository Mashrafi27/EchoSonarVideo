"""EchoPrime integration for the cold-start SFT track (frozen video encoder + Qwen3-8B text).

Ported from the local reference install (`/home/mashrafimonon/Mashrafi/EchoPrime`, weights and
code from Vukadinovic et al., arXiv:2410.09704, Cedars-Sinai Academic Software License -- academic
research use only, see docs/OPEN_ISSUES.md #9 for the license text and required acknowledgment),
NOT vendored as a git submodule since it's a personal download outside version control, not a
public git repo with a pinnable commit. Only the small amount of code this project actually needs
(the frozen encoder wrapper + its exact preprocessing) is ported here, deliberately, not imported
from that external path -- so this repo is reproducible without that specific local directory.

Distinct from `verl_bridge`/`data_core` (the Qwen3-VL native-vision GRPO track) and `tool_env` (the RL
tool environment) -- this package is the EchoSonar-R-architecture-matched track: frozen EchoPrime
mViT encoder -> trainable projector -> Qwen3-8B (text-only) LM.
"""
