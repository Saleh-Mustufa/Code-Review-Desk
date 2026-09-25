# ReviewDesk

Paste a unified diff (`git diff` output) into the chat. Three agents — security, tests and style — review it in parallel; their findings stream in as they land, then get merged and severity-ordered, with per-agent latency and token usage at the end.

Tips:

- Settings from your first diff are reused for the session.
- Put `desk: repo=my-repo lang=python strict=strict` on the first line to change them.
- A diff containing a credential is refused rather than echoed.
