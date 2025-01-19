```bash
# Define TASK
export TASK="h1-walk-v0"
export TASK="h1-run-v0"
export WANDB_ENTITY="lyt0112-peking-university"

# Train TD-MPC2
python -m tdmpc2.train disable_wandb=False wandb_entity=${WANDB_ENTITY} exp_name=tdmpc task=humanoid_${TASK} seed=0

# Train DreamerV3
python -m embodied.agents.dreamerv3.train --configs humanoid_benchmark --run.wandb True --run.wandb_entity ${WANDB_ENTITY} --method dreamer --logdir logs --task humanoid_${TASK} --seed 0

# Train SAC
# Failed
python ./jaxrl_m/examples/mujoco/run_mujoco_sac.py --env_name ${TASK} --wandb_entity ${WANDB_ENTITY} --seed 0
```

Failed to train DreamerV3 for task walk (but run is successful).

```bash
Terminating workers due to an exception.
Traceback (most recent call last):
  File "<frozen runpy>", line 198, in _run_module_as_main
  File "<frozen runpy>", line 88, in _run_code
  File "/home/descfly/humanoid-bench-transfer/dreamerv3/embodied/agents/dreamerv3/train.py", line 219, in <module>
    main()
  File "/home/descfly/humanoid-bench-transfer/dreamerv3/embodied/agents/dreamerv3/train.py", line 74, in main
    embodied.run.train_eval(
  File "/home/descfly/humanoid-bench-transfer/dreamerv3/embodied/run/train_eval.py", line 107, in train_eval
    eval_driver = embodied.Driver(fns, args.driver_parallel)
                  ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/descfly/humanoid-bench-transfer/dreamerv3/embodied/core/driver.py", line 26, in __init__
    self.act_space = self._receive(self.pipes[0])
                     ^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/descfly/humanoid-bench-transfer/dreamerv3/embodied/core/driver.py", line 92, in _receive
    raise RuntimeError(arg)
RuntimeError: Offscreen framebuffer is not complete, error 0x8cdd
```

Failed to train SAC.

```bash
❯ python ./jaxrl_m/examples/mujoco/run_mujoco_sac.py --env_name ${TASK} --wandb_entity ${WANDB_ENTITY} 
--seed 0
Traceback (most recent call last):
  File "/home/descfly/humanoid-bench-transfer/./jaxrl_m/examples/mujoco/run_mujoco_sac.py", line 12, in <module>
    from flax.training import checkpoints
  File "/home/descfly/miniconda3/envs/humanoidbench/lib/python3.11/site-packages/flax/training/checkpoints.py", line 42, in <module>
    import orbax.checkpoint as ocp
  File "/home/descfly/miniconda3/envs/humanoidbench/lib/python3.11/site-packages/orbax/checkpoint/__init__.py", line 26, in <module>
    from orbax.checkpoint import checkpointers
  File "/home/descfly/miniconda3/envs/humanoidbench/lib/python3.11/site-packages/orbax/checkpoint/checkpointers.py", line 20, in <module>
    from orbax.checkpoint._src.checkpointers.async_checkpointer import AsyncCheckpointer
  File "/home/descfly/miniconda3/envs/humanoidbench/lib/python3.11/site-packages/orbax/checkpoint/_src/checkpointers/async_checkpointer.py", line 56, in <module>
    def _add_deadline_exceeded_notes(e: jax.errors.JaxRuntimeError):
                                        ^^^^^^^^^^^^^^^^^^^^^^^^^^
AttributeError: module 'jax.errors' has no attribute 'JaxRuntimeError'
```
