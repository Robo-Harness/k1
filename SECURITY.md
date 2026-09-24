# Security and private artifacts

API credentials are supplied only through `ROBO_HARNESS_API_KEY`. This project does not read a shared credential file, SSH configuration or a user's shell history. It does not install network tunnels. Remote APIs require HTTPS; proxies are explicitly selected. Never put a key in a model prompt or task instruction.

Requests, images, tool results and native actions are saved locally for reproducibility. Although authorization headers are not archived, arbitrary prompts, provider responses and error messages can contain sensitive user content. Treat **all generated outputs and training data as private** until separately reviewed. A `.gitignore` rule is not access control, and force-adding ignored files bypasses it.

Before sharing source:

1. Keep datasets, weights, logs, videos, environment files, credentials and third-party source out of the source tree.
2. Run `robo-harness-audit` on the clean tree. Review filenames as well as contents.
3. Inspect the built distribution, not only the working directory. The distribution allow-list is declared in `MANIFEST.in`.
4. Do not publish local virtual environments, Git metadata, symlinks to private directories or generated processor/adapter configs containing local paths.
5. Review external licenses separately. MIT covers this project's code, not upstream datasets or weights.

The audit is heuristic and cannot certify absence of all secrets. Tests use synthetic data only. Do not submit credentials or private trajectories in public issue reports. Serialized benchmark and optimizer files must come from trusted sources.

This is simulation research software, not a safety-certified real-robot controller. Geometric checks are incomplete and cannot replace physical safety systems.
