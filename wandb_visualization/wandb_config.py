import wandb


def wandb_init(name, group):
    # login wandb
    wandb.login(key="44af7bf1f24c6aab99ae33b0ae4fa5a5c8a59590", relogin=True)
    # start a new wandb run to track this script
    wandb.init(
        # set the wandb project where this run will be logged
        project="Defect-Detection",
        entity="cosminaionut",
        name=name,
        group=group,
        # track hyperparameters and run metadata
        # config={
        #     "learning_rate": 0.02,
        #     "architecture": "CNN",
        #     "dataset": "CIFAR-100",
        #     "epochs": 10,
        # }
    )
    return wandb


#
# wandb.save(file_name)
#     wandb.log({"n_trained_episodes": result['episodes_this_iter'],
#                "mean_reward": result['episode_reward_mean'],
#                "max_reward": result['episode_reward_max'],
#                "own_mean": result['count_bookown_mean'],
#                "wait_mean": result['count_wait_mean'],
#                "share_mean": result['count_share_mean'],
#                "share_of_bookown_mean": result['share_bookown_mean'],
#                "share_of_wait_mean": result['share_wait_mean'],
#                "share_of_share_mean": result['share_share_mean'],
#                "share_to_own_ratio_max": result['share_to_own_ratio_max'],
#                "share_to_own_ratio_mean": result['share_to_own_ratio_mean'],
#                'count_steps_mean': result["count_steps_mean"],
#                'count_delivered_on_time': result["count_delivered_on_time"],
#                'count_delivered_with_delay': result["count_delivered_with_delay"],
#                'count_not_delivered': result["count_not_delivered"],
#                'share_delivered_on_time': result["count_delivered_on_time"] / result['episodes_this_iter'],
#                'boolean_has_booked_any_own': result["boolean_has_booked_any_own"],
#                'count_shared_available': result["count_shared_available"],
#                'count_shared_available_useful': result["count_shared_available_useful"],
#                'shared_taken_to_shared_available': result["shared_taken_to_shared_available"],
#                'shared_available_useful_to_shared_available': result["shared_available_useful_to_shared_available"],
#                'shared_taken_useful_to_shared_available_useful': result[
#                    "shared_taken_useful_to_shared_available_useful"],
#                "ratio_shared_available_to_all_steps": result["ratio_shared_available_to_all_steps"],
#                "ratio_delivered_without_bookown_to_all_delivered": result[
#                    "ratio_delivered_without_bookown_to_all_delivered"],
#                'distance_reduced_with_ownrides': result['distance_reduced_with_ownrides_mean'],
#                'distance_reduced_with_shared': result['distance_reduced_with_shared_mean'],
#                'distance_reduced_with_ownrides_share': result['distance_reduced_with_ownrides_share_mean'],
#                'distance_reduced_with_shared_share': result['distance_reduced_with_shared_share_mean'],
#
#                })