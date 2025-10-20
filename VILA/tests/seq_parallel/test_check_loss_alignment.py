import json
import os
import sys
import unittest


class TestLosses(unittest.TestCase):
    def test_loss_alignment(self):
        project_name = sys.argv[1]
        dp_log_file = f"./checkpoints/dp_{project_name}/log_history.json"
        sp_log_file = f"./checkpoints/sp_{project_name}/log_history.json"

        self.assertTrue(os.path.exists(dp_log_file), "DP log file does not exist")
        self.assertTrue(os.path.exists(sp_log_file), "SP log file does not exist")

        if not os.path.exists(sp_log_file):
            self.fail(f"SP log file was expected but does not exist: {sp_log_file}")

        with open(dp_log_file) as f:
            dp_logs = json.load(f)
        with open(sp_log_file) as f:
            sp_logs = json.load(f)

        dp_losses = [entry["loss"] for entry in dp_logs if "loss" in entry]
        sp_losses = [entry["loss"] for entry in sp_logs if "loss" in entry]

        for step, (loss_dp, loss_sp) in enumerate(zip(dp_losses, sp_losses)):
            self.assertAlmostEqual(
                loss_dp, loss_sp, places=1, msg=f"Loss mismatch at step {step}: DP {loss_dp} vs SP {loss_sp}"
            )

        print("DP losses:", dp_losses)
        print("SP losses:", sp_losses)
        print("Losses match")


if __name__ == "__main__":
    unittest.main(argv=["first-arg-is-ignored"])
