# Configuration

`dataset-scout` keeps configuration explicit. There is no global state.
The CLI builds a `ScoutContext` from environment variables (and a
`.env` file in the current working directory); library callers
construct one directly.

```python
from dataset_scout import ScoutContext, recon

ctx = ScoutContext.from_env()           # CLI-style: reads env + .env
result = recon("find prompt injection corpora", ctx=ctx)
```

---

## 1. Azure OpenAI (Entra auth)

`dataset-scout` uses **Azure OpenAI** as its single LLM target,
authenticated via **Microsoft Entra**. There is no API-key path for
the LLM.

### Local development

```bash
az login                                                        # one-time
cp .env.example .env
# Then edit .env:
#   AZURE_OPENAI_ENDPOINT=https://your-aoai-resource.openai.azure.com/
#   AZURE_OPENAI_DEPLOYMENT=gpt-4o-mini
```

The `DefaultAzureCredential` chain checks (in order): env-var
service-principal, workload identity, managed identity,
shared-token-cache, Azure CLI (`az login`), and interactive browser.
Whichever wins first is used. Tokens are cached internally — you
don't need to re-`az login` between runs unless your token expires.

### CI / Azure-hosted

Either:

- **Service principal** — set `AZURE_TENANT_ID`, `AZURE_CLIENT_ID`,
  and `AZURE_CLIENT_SECRET` (the credential picks them up
  automatically), **or**
- **Managed identity** — no env vars needed; the credential
  discovers it.

### Recognised env vars

| Variable | Purpose | Default |
|---|---|---|
| `AZURE_OPENAI_ENDPOINT` | e.g. `https://my-aoai.openai.azure.com` (trailing slash trimmed) | — |
| `AZURE_OPENAI_DEPLOYMENT` | Deployment name (NOT model name; e.g. `gpt-4o-mini`) | — |
| `AZURE_OPENAI_API_VERSION` | Pin a specific API version | `2024-10-21` |

When **either** `AZURE_OPENAI_ENDPOINT` or `_DEPLOYMENT` is missing
the pipeline runs in **metadata-only mode** — see
[concepts.md](concepts.md#2-modes-full-vs-metadata-only).

---

## 2. HuggingFace

Public datasets are accessible without a token, but the unauthenticated
rate limits are aggressive enough to hurt during real recon. Setting
a token is recommended.

```bash
# .env
HUGGINGFACE_HUB_TOKEN=hf_...
```

Either `HUGGINGFACE_HUB_TOKEN` or `HF_TOKEN` works.

Get a token at <https://huggingface.co/settings/tokens>. Read-only
permissions are sufficient.

---

## 3. Kaggle *(M1b — not yet wired)*

Set `KAGGLE_USERNAME` and `KAGGLE_KEY` and Kaggle source will activate
once it lands.

---

## 4. Paths

| Variable | What it overrides | Default (Linux/macOS) | Default (Windows) |
|---|---|---|---|
| `DATASET_SCOUT_CACHE_DIR` | SQLite cache (M1b) | `~/.cache/dataset-scout/` | `%LOCALAPPDATA%\dataset-scout\cache\` |
| `DATASET_SCOUT_CONFIG_DIR` | Config TOML | `~/.config/dataset-scout/` | `%APPDATA%\dataset-scout\` |
| `DATASET_SCOUT_OUT_DIR` | Default `--out` for `recon` | `./dscout-out/` | `.\dscout-out\` |

XDG variables (`XDG_CACHE_HOME`, `XDG_CONFIG_HOME`) are honored on
Unix when the explicit overrides aren't set.

---

## 5. The `.env` auto-load

The CLI loads `.env` from the **current working directory** at
startup using `python-dotenv` with `override=False`. That means:

- Variables already set in your shell **win**. `.env` only fills gaps.
- Drop `.env` in a project root and `cd` there — recon picks it up.
- Library callers do **not** auto-load `.env`; pass an explicit
  `ScoutContext` (use `ScoutContext.from_env(env={"AZURE_OPENAI_ENDPOINT": ..., ...})`
  if you want to control it from a test).

`.env.example` in the repo root documents every recognised variable
with a short comment about when each is needed.

---

## 6. Source enablement

Sources can be disabled even if their auth is configured:

```python
from dataset_scout import ScoutContext
from dataset_scout.context import SourceConfig

ctx = ScoutContext(
    sources={
        "huggingface": SourceConfig(enabled=True),
        "kaggle":      SourceConfig(enabled=False),
        "pwc":         SourceConfig(enabled=False),
    },
)
```

The CLI respects the same map; future `datascout sources enable/disable`
will write to a TOML config file.

---

## 7. Programmatic ScoutContext

```python
from dataset_scout import ScoutContext

# Direct construction (e.g. from a server request context)
ctx = ScoutContext(
    aoai_endpoint="https://my-aoai.openai.azure.com",
    aoai_deployment="gpt-4o-mini",
    aoai_api_version="2024-10-21",
    api_keys={"HUGGINGFACE_HUB_TOKEN": "hf_..."},
    out_dir=Path("/tmp/scout"),
)
```

`ScoutContext` is a frozen Pydantic v2 model — `extra="forbid"`,
mutation raises. The intent is that contexts flow through call sites
unchanged; build a new one if you need different config.
