Read `SPEC.md` in full at the start of every session; it is the project-wide source of truth.
Then read the one assigned `specs/F<N>-*.md` and implement only that feature's Scope.
`SPEC.md` wins over a feature spec; if they contradict, stop and report it instead of choosing.
Never rename a type, path, env var, JSON key, or metric defined in `SPEC.md` - leave `TODO(F<N>)`.
Finish by running the feature's Verify block and pasting its real output into the summary.

Task runner: `make <target>` on Linux/Kaggle, `./tasks.ps1 <target>` on Windows (same commands).
