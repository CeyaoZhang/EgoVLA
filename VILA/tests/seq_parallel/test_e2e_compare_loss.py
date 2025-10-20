import argparse
import json
import os
import time
import unittest
import uuid
from unittest import mock

os.environ["MKL_THREADING_LAYER"] = "GNU"


class TestSPTrainingLossAlignment(unittest.TestCase):
    def setUp(self):

        # Mock configuration
        # 'disable': without applying the mock function, using `flash_attn_varlen`
        # 'shortcut': attn_output = query
        # 'ignore_mask': using `flash_attn` instead of `flash_attn_varlen`

        self.mock_type = "disable"  # choices=['disable', 'shortcut', 'ignore_mask']

        # Test configuration
        self.script_path = "tests/seq_parallel/attn_mock/run_debug.sh"
        self.sp_degree = 4
        self.gpu_count = 8
        self.dataset_name = "shot2story_shotonly"  # ai2d_train_12k+chartqa_train_18k+shot2story_shotonly
        self.global_batch_size = 64
        self.project_id = str(uuid.uuid4())[:4]
        self.project_name = f"loss_test_id{self.project_id}"
        self.max_steps = 1

        # Slurm cluster launch command
        self.partition = "llm_s"
        self.gpu_per_node = 8

        # self.run_one_only = "dp"
        self.run_one_only = "false"  # run dp & sp together

    def generate_training_script(self, mode):
        self.run_name = f"{mode}_{self.project_name}"
        command = [
            f"MOCK_TYPE={self.mock_type}",
            "srun",
            "-p",
            self.partition,
            f"--job-name={self.run_name}",
            "-n",
            "1",
            f"--gres=gpu:{self.gpu_per_node}",
            f"--ntasks-per-node=1",
            "--exclusive",
            "bash",
            self.script_path,
            str(self.gpu_count),
            self.dataset_name,
            mode,
            str(self.global_batch_size),
            self.run_name,
            str(self.max_steps),
            self.mock_type,
            str(self.sp_degree),
        ]

        command = " ".join(command)
        return command

    def check_log_files(self):
        dp_log_file = os.path.join(f"./checkpoints/dp_{self.project_name}", "log_history.json")
        sp_log_file = os.path.join(f"./checkpoints/sp_{self.project_name}", "log_history.json")

        if os.path.exists(dp_log_file) and os.path.exists(sp_log_file):
            with open(dp_log_file) as f:
                dp_logs = json.load(f)
                print("dp_logs", dp_logs)
            with open(sp_log_file) as f:
                sp_logs = json.load(f)
                print("sp_logs", sp_logs)
            dp_losses = [entry["loss"] for entry in dp_logs if "loss" in entry]
            sp_losses = [entry["loss"] for entry in sp_logs if "loss" in entry]

            return dp_losses, sp_losses
        return None, None

    def test_loss_alignment(self):
        dp_command = self.generate_training_script("dp")
        sp_command = self.generate_training_script("sp")

        if self.run_one_only == "sp":
            command = sp_command
            exit_code = os.system(command)
            return
        elif self.run_one_only == "dp":
            command = dp_command
            exit_code = os.system(command)
            return
        else:
            command = f"({dp_command} &) && ({sp_command} &)"

        print(f"Executing command: {command}")

        # with self.mock_flash_attn:
        exit_code = os.system(command)
        if exit_code != 0:
            print(f"Command failed with exit code {exit_code}")
            return

        # Wait and check for the log files every 10 seconds
        for _ in range(60):  # Check for up to 10 minutes
            dp_losses, sp_losses = self.check_log_files()
            if dp_losses is not None and sp_losses is not None:
                for step, (loss_dp, loss_sp) in enumerate(zip(dp_losses, sp_losses)):
                    self.assertAlmostEqual(
                        loss_dp, loss_sp, places=1, msg=f"Loss mismatch at step {step}: DP {loss_dp} vs SP {loss_sp}"
                    )
                return
            time.sleep(10)


if __name__ == "__main__":
    # parser = argparse.ArgumentParser(description='Test SP Training Loss Alignment')
    # parser.add_argument('--mock_type', type=str, choices=['disable', 'shortcut', 'ignore_mask'], default='shortcut', help='Type of mock function to use for flash_attn')
    # args = parser.parse_args()

    # unittest.main(argv=['first-arg-is-ignored'], exit=False)
    unittest.main()
