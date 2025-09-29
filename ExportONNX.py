import torch as th
from stable_baselines3 import PPO
from stable_baselines3.common.policies import BasePolicy
class OnnxableSB3Policy(th.nn.Module):
    def __init__(self, policy: BasePolicy):
        super().__init__()
        self.policy = policy

    def forward(self, observation: th.Tensor):
        # Returns `actions, values, log_prob`
        return self.policy(observation, deterministic=True)

# Load trained PPO model
model = PPO.load("weights/keyboard.zip", device="cpu")

# Wrap the policy for ONNX export
onnx_policy = OnnxableSB3Policy(model.policy)
print(onnx_policy)

observation_size = model.observation_space.shape
dummy_input = th.randn(1, *observation_size)

onnx_path = "weights/ppo_model.onnx"
th.onnx.export(
    onnx_policy,
    dummy_input,
    onnx_path,
    opset_version=17,
    input_names=["input"],
)
print(f"Model successfully exported to {onnx_path}")