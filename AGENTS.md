# BonsAI

An edge-optimized C++ inference server with an OpenAI-compatible API for models on Google's LiteRT-LM, built for Raspberry Pi, Jetson and similar hardware. Fork of arenblake/bonsAI.

## Deployment map

**Status:** dormant. Binaries ship as GitHub Releases; no host runs it in production now.

```text
tag v* → GitHub Actions build → GitHub Release asset bonsai-linux_<arch> → scripts/install.sh → /usr/local/bin → `bonsai model.litertlm` serves :8080
```

| Layer | Platform | Notes |
|---|---|---|
| Server | single C++20 binary (Oat++) | /v1/chat/completions, /v1/models, SSE streaming |
| AI | Google LiteRT-LM (submodule, branch bonsai-engine), Gemma 4 E2B-it `.litertlm` | Vision and audio input; MCP HTTP client for tools |
| Distribution | GitHub Releases (`release.yml` on `v*` tags or manual) | x86_64 on master; the aarch64 matrix is on unmerged branches. Latest: v0.1.0-perf-knobs (2026-05-19). No release-please |
| Deployments | an edge-device agent backend (CPU only, far too slow) | Paused until a desktop GPU host is available |
| Tests | pytest against a live server | Not in CI |

`scripts/install.sh` and the README point at upstream arenblake, not this fork.

_Mapped 2026-10-04 from the default branch's config. Update this section when a platform changes._
