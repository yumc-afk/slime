import ray
import os

from slime.ray.placement_group import create_actor_group, create_placement_groups, create_rollout_manager
from slime.utils.arguments import parse_args
from slime.utils.wandb_utils import init_wandb_primary


def _setup_plan_act_orchestrator(args):
    """
    Setup Plan/Act orchestrator if this is the planner role.
    Uses detached Ray actor to survive job restarts.
    """
    try:
        from examples.plan_act.plan_act_orchestrator import PlanActOrchestrator
        
        # Create orchestrator config
        orchestrator_config = {
            "max_turns": 5,
            "max_concurrent_acts": 8,
            "session_timeout": args.plan_act_timeout,
            "rollout_batch_size": args.rollout_batch_size,
        }
        
        # Create detached orchestrator actor in specified namespace
        orchestrator = PlanActOrchestrator.options(
            name=f"{args.orchestrator_namespace}_orchestrator",
            namespace=args.orchestrator_namespace,
            lifetime="detached"
        ).remote(config=orchestrator_config)
        
        print(f"Plan/Act orchestrator created in namespace: {args.orchestrator_namespace}")
        
        # Store reference for later use (optional)
        args._orchestrator_ref = orchestrator
        
    except ImportError as e:
        print(f"Warning: Could not import PlanActOrchestrator: {e}")
        if not args.fallback_on_timeout:
            raise RuntimeError("Plan/Act orchestrator required but not available")
        print("Continuing without orchestrator (fallback mode)")
    except Exception as e:
        print(f"Warning: Failed to setup Plan/Act orchestrator: {e}")
        if not args.fallback_on_timeout:
            raise
        print("Continuing without orchestrator (fallback mode)")


def train(args):
    # Handle Plan/Act architecture orchestrator startup
    if args.enable_plan_act and args.agent_role == "planner":
        _setup_plan_act_orchestrator(args)
    
    # Handle port offset for Plan/Act mode to avoid conflicts
    if args.enable_plan_act and args.master_port_offset > 0:
        if 'MASTER_PORT' in os.environ:
            current_port = int(os.environ['MASTER_PORT'])
            new_port = current_port + args.master_port_offset
            os.environ['MASTER_PORT'] = str(new_port)
            print(f"Adjusted MASTER_PORT from {current_port} to {new_port} for agent role: {args.agent_role}")
    
    # allocate the GPUs
    pgs = create_placement_groups(args)
    wandb_run_id = init_wandb_primary(args)

    actor_model = create_actor_group(args, pgs["actor"], wandb_run_id=wandb_run_id)

    # create the rollout manager, with sglang engines inside.
    rollout_manager = create_rollout_manager(args, pgs["rollout"], wandb_run_id=wandb_run_id)

    # calculate num_rollout from num_epoch
    num_rollout_per_epoch = None
    if args.num_rollout is None:
        num_rollout_per_epoch = ray.get(rollout_manager.data_buffer.get_num_rollout_per_epoch.remote())
        args.num_rollout = num_rollout_per_epoch * args.num_epoch
    assert args.num_rollout > 0

    # sync the initialization (model initalization, load checkpoint, etc.)
    start_rollout_ids = ray.get(
        actor_model.async_init(args, role="actor", with_ref=args.kl_coef != 0 or args.use_kl_loss)
    )
    assert len(set(start_rollout_ids)) == 1
    if args.start_rollout_id is None:
        args.start_rollout_id = start_rollout_ids[0]

    if args.rollout_global_dataset:
        ray.get(rollout_manager.data_buffer.load.remote(args.start_rollout_id - 1))

    # initialize the connection for weight update during training
    ray.get(actor_model.async_init_weight_update_connections(rollout_manager))

    if args.offload:
        ray.get(rollout_manager.async_onload())

    # always update weight first so that sglang has the loaded weights from training.
    ray.get(actor_model.async_update_weights())

    # train loop.
    # note that for async training, one can change the position of the sync operation(ray.get).
    for rollout_id in range(args.start_rollout_id, args.num_rollout):
        # TODO extract the duplicated eval logic
        if args.eval_interval is not None and rollout_id == 0:
            eval_rollout_data_ref = ray.get(rollout_manager.async_generate(rollout_id, evaluation=True))
            ray.get(actor_model.async_eval(rollout_id, eval_rollout_data_ref))

        rollout_data_ref = ray.get(rollout_manager.async_generate(rollout_id))

        if args.offload:
            ray.get(rollout_manager.async_offload())

        ray.get(actor_model.async_train(rollout_id, rollout_data_ref))

        if args.save_interval is not None and (
            (rollout_id + 1) % args.save_interval == 0
            or (num_rollout_per_epoch is not None and (rollout_id + 1) % num_rollout_per_epoch == 0)
        ):
            ray.get(actor_model.async_save_model(rollout_id))
            if args.rollout_global_dataset:
                ray.get(rollout_manager.data_buffer.save.remote(rollout_id))

        if args.offload:
            ray.get(actor_model.async_offload())
            ray.get(rollout_manager.async_onload())

        ray.get(actor_model.async_update_weights())

        if args.eval_interval is not None and (
            (rollout_id + 1) % args.eval_interval == 0
            or (num_rollout_per_epoch is not None and (rollout_id + 1) % num_rollout_per_epoch == 0)
        ):
            eval_rollout_data_ref = ray.get(rollout_manager.async_generate(rollout_id, evaluation=True))
            ray.get(actor_model.async_eval(rollout_id, eval_rollout_data_ref))


if __name__ == "__main__":
    args = parse_args()
    train(args)
