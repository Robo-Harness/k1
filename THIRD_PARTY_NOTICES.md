# Third-party notices

The project implementation is licensed under MIT. External software, datasets and pretrained models retain their own licenses and notices. They are not included or relicensed here.

- **LIBERO-Pro / LIBERO**: benchmark definitions, assets, initial states and simulation interface. Source: https://github.com/Zxy-MLlab/LIBERO-PRO . Follow that repository's license and original LIBERO attribution.
- **RoboSuite / MuJoCo**: native simulation, robot models, controllers and camera utilities. Source: https://github.com/ARISE-Initiative/robosuite . The RoboSuite adapter expects the pinned dependency described in `docs/environments.md`.
- **RATs / CaP-X**: external RoboSuite task wrappers used for scene construction and native success conditions. Source: https://github.com/Playful-RATs/RATs . The wrappers are imported from a user-provided checkout; their source is not copied into this repository. Citing those task definitions does not imply identical evaluation protocols.
- **SAM3, TAPNext++, GraspGen**: optional perception or grasp providers. Their weights/code must be obtained separately under their respective licenses; this project supplies interfaces, not redistribution rights.
- **Qwen, Transformers, PEFT and PyTorch**: local model loading and fine-tuning dependencies. Model weights have separate terms from this harness's MIT license.
- **NumPy, SciPy, OpenCV, HTTPX, Pillow and PyYAML**: runtime dependencies installed through their distributions.

Preserving external package/class names such as Panda, Qwen or CaP-X is necessary for technical compatibility and attribution; the algorithm implemented by this repository is consistently named **robo-harness k1**.
