# VILA Unit Tests

Unit tests are important for projects where there are multiple people contribute to the same codebase. For VILA, we suggest,

- Prepare unit tests after when adding new features to the repo
- Trigger unit tests (locally) before sending PR.

## How to trigger tests locally

```bash
~/workspace/VILA$ bash CIs/continual_local.sh
Printing reports locally.
[succeeded] ./tests/cpu_tests/success.py
[succeeded] ./tests/cpu_tests/fail.py
[succeeded] ./tests/import_tests.py
[succeeded] ./tests/cpu_tests/success_2.py
[succeeded] ./tests/test_tokenizer.py
[succeeded] ./tests/datasets/coyo25m_sam.py
[succeeded] ./tests/cpu_tests/test_preprocess_multimodal.py
[succeeded] ./tests/datasets/jukinmedia_small.py
[succeeded] ./tests/datasets/panda70m_small.py
[succeeded] ./tests/datasets/shot2story_small.py
```

## How to organize / contribute unit tests

0. If your tests require GPU to run, put them under \`tests/gpu_tests/.
1. Create folder and catogerize them by features.
1. Reference existings unit tests and make your own.
