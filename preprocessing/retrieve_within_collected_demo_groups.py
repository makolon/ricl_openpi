import numpy as np
from collections import defaultdict
import json
from openpi.policies.utils import myprint, embed, load_dinov2, embed_with_batches, EMBED_DIM
import os
import argparse
from autofaiss import build_index

# Prevent JAX from preallocating most of the GPU memory
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"

EMBED_TYPES = ["top_image", "wrist_image"]

def create_idx_fol_mapping(ds_name):
    mapping_names = ['groups_to_ep_fols', 'ep_idxs_to_fol', 'fols_to_ep_idxs', 'groups_to_ep_idxs']
    mappings = {
        temp_name: defaultdict(list) if temp_name == 'groups_to_ep_idxs' else {}
        for temp_name in mapping_names
    }

    count = 100000  # Count starts from 100k because droid has less than 100k episodes
    groups = [f'{ds_name}/{dir}' for dir in os.listdir(ds_name)]
    mappings['groups_to_ep_fols'] = {
        group: [f'{group}/{fol}' for fol in os.listdir(group)]
        for group in groups
    }

    for group, ep_fols in mappings['groups_to_ep_fols'].items():
        for ep_fol in ep_fols:
            mappings['ep_idxs_to_fol'][count] = ep_fol
            mappings['fols_to_ep_idxs'][ep_fol] = count
            mappings['groups_to_ep_idxs'][group].append(count)
            count += 1

    # Delete existing files if they exist
    for file_stub in mapping_names:
        if os.path.exists(f'{ds_name}/{file_stub}.json'):
            os.remove(f'{ds_name}/{file_stub}.json')

    # Save mappings as JSON files
    for file_stub in mapping_names:
        with open(f'{ds_name}/{file_stub}.json', 'w') as f:
            json.dump(mappings[file_stub], f, indent=4)

    return mappings

def create_idx_fol_mapping_for_a_single_group(ds_name):
    mapping_names = ['groups_to_ep_fols', 'ep_idxs_to_fol', 'fols_to_ep_idxs', 'groups_to_ep_idxs']

    # Skip if files already exist
    for file_stub in mapping_names:
        if os.path.exists(f'{ds_name}/{file_stub}.json'):
            print(f'skipping {ds_name=} because {file_stub}.json exists. If you want to re-run, delete the four json files at {ds_name}/.')
            return None

    mappings = {
        temp_name: defaultdict(list) if temp_name == 'groups_to_ep_idxs' else {}
        for temp_name in mapping_names
    }

    count = 100000  # Count starts from 100k for this single group
    groups = [f'{ds_name}']
    mappings['groups_to_ep_fols'] = {
        group: [f'{group}/{fol}' for fol in os.listdir(group)]
        for group in groups
    }

    for group, ep_fols in mappings['groups_to_ep_fols'].items():
        for ep_fol in ep_fols:
            mappings['ep_idxs_to_fol'][count] = ep_fol
            mappings['fols_to_ep_idxs'][ep_fol] = count
            mappings['groups_to_ep_idxs'][group].append(count)
            count += 1

    # Save mappings as JSON files
    for file_stub in mapping_names:
        with open(f'{ds_name}/{file_stub}.json', 'w') as f:
            json.dump(mappings[file_stub], f, indent=4)

    return mappings

def retrieval_preprocessing(groups_to_ep_idxs, ep_idxs_to_fol, nb_cores_autofaiss, knn_k, embedding_type):
    myprint(f'[retrieval_preprocessing] starting retrieval preprocessing for {embedding_type}')

    num_groupings = len(groups_to_ep_idxs)

    for chosen_id_count, (chosen_id, ep_idxs) in enumerate(groups_to_ep_idxs.items()):
        total_episodes_in_grouping = len(ep_idxs)

        all_embeddings = []
        all_embeddings_map = {}
        all_indices = []

        for ep_count, ep_idx in enumerate(ep_idxs):
            if embedding_type in EMBED_TYPES:
                ep_embeddings = np.load(f"{ep_idxs_to_fol[ep_idx]}/processed_demo.npz")[f"{embedding_type}_embeddings"]
                all_embeddings.append(ep_embeddings)
                all_embeddings_map[ep_idx] = ep_embeddings
            elif embedding_type == "both":
                ep_embeddings = np.concatenate([
                    np.load(f"{ep_idxs_to_fol[ep_idx]}/processed_demo.npz")[f"{item}_embeddings"]
                    for item in EMBED_TYPES
                ], axis=1)
                all_embeddings.append(ep_embeddings)
                all_embeddings_map[ep_idx] = ep_embeddings
            else:
                raise ValueError(f'{embedding_type=} is not in {EMBED_TYPES} and not "both"')

            num_steps = len(ep_embeddings)
            all_indices.extend([[ep_idx, stp_idx] for stp_idx in range(num_steps)])

        all_embeddings = np.concatenate(all_embeddings, axis=0)
        all_indices = np.array(all_indices)
        embedding_dim = all_embeddings.shape[1]
        num_total = len(all_embeddings)

        myprint(f'[retrieval_preprocessing] concatenated all embeddings and indices for {total_episodes_in_grouping} episodes for {chosen_id} [chosen_id count {chosen_id_count}/{num_groupings}]')
        myprint(f'[retrieval_preprocessing] we have {num_total=} {embedding_dim=}')

        for ep_count, ep_idx in enumerate(ep_idxs):
            if os.path.exists(f"{ep_idxs_to_fol[ep_idx]}/indices_and_distances.npz"):
                myprint(f'[retrieval_preprocessing] skipping episode {ep_idx} [episode count {ep_count}/{total_episodes_in_grouping}]')
                continue

            all_other_episodes_mask = np.array([
                True if ep_idx_other != ep_idx else False
                for (ep_idx_other, stp_idx_other) in all_indices
            ])
            num_retrieval = np.sum(all_other_episodes_mask)

            this_episode_mask = np.array([
                True if ep_idx_other == ep_idx else False
                for (ep_idx_other, stp_idx_other) in all_indices
            ])
            num_query = np.sum(this_episode_mask)

            assert num_retrieval + num_query == num_total

            print(f'[retrieval_preprocessing] for episode {ep_idx} [episode count {ep_count}/{total_episodes_in_grouping}], we have {num_retrieval=} {num_query=}')

            all_other_episodes_embeddings = all_embeddings[all_other_episodes_mask]
            all_other_episodes_indices = all_indices[all_other_episodes_mask]
            this_episode_embeddings = all_embeddings[this_episode_mask]
            this_episode_indices = all_indices[this_episode_mask]

            assert all_other_episodes_embeddings.shape == (num_retrieval, embedding_dim)
            assert all_other_episodes_indices.shape == (num_retrieval, 2)
            assert this_episode_embeddings.shape == (num_query, embedding_dim)
            assert this_episode_indices.shape == (num_query, 2)
            assert this_episode_indices.dtype == np.int64
            assert all_other_episodes_indices.dtype == np.int64

            knn_index, knn_index_infos = build_index(
                embeddings=all_other_episodes_embeddings,
                save_on_disk=False,
                min_nearest_neighbors_to_retrieve=knn_k + 5,
                max_index_query_time_ms=10,
                max_index_memory_usage="25G",
                current_memory_available="50G",
                metric_type='l2',
                nb_cores=nb_cores_autofaiss,
            )

            topk_distances, topk_indices = knn_index.search(this_episode_embeddings, 2 * knn_k)

            try:
                topk_indices = np.array([
                    [idx for idx in indices if idx != -1][:knn_k]
                    for indices in topk_indices
                ])
            except:
                print(f'Too many -1s from topk_indices')
                temp_topk_indices = [
                    [idx for idx in indices if idx != -1][:knn_k]
                    for indices in topk_indices
                ]
                print(f'after -1s, min len: {min([len(indices) for indices in temp_topk_indices])}, max len {max([len(indices) for indices in temp_topk_indices])}')
                print(f'Leaving some -1s in topk_indices and continuing')
                topk_indices = np.array([
                    row + [-1 for _ in range(knn_k - len(row))]
                    for row in temp_topk_indices
                ])

            retrieved_indices = all_other_episodes_indices[topk_indices]
            assert retrieved_indices.shape == (num_query, knn_k, 2)
            assert retrieved_indices.dtype == np.int64

            retrieved_indices = retrieved_indices.astype(np.int32)
            this_episode_indices = this_episode_indices.astype(np.int32)

            myprint(f'[retrieval_preprocessing] calculating distances ...')
            all_distances = []

            for ct in range(num_query):
                retrieved_indices_row = retrieved_indices[ct]
                temp_first_embedding = all_embeddings_map[retrieved_indices_row[0][0]][retrieved_indices_row[0][1]]
                query_ep_idx, query_stp_idx = this_episode_indices[ct]
                assert query_ep_idx == ep_idx and query_stp_idx == ct

                distances = [0.0] + [
                    np.linalg.norm(all_embeddings_map[e_idx][s_idx] - temp_first_embedding)
                    for e_idx, s_idx in retrieved_indices_row[1:]
                ]
                distances.append(
                    np.linalg.norm(all_embeddings_map[query_ep_idx][query_stp_idx] - temp_first_embedding)
                )
                all_distances.append(distances)

            all_distances = np.array(all_distances)
            assert all_distances.shape == (num_query, knn_k + 1)

            np.savez(
                f"{ep_idxs_to_fol[ep_idx]}/indices_and_distances.npz",
                retrieved_indices=retrieved_indices,
                query_indices=this_episode_indices,
                distances=all_distances
            )
            myprint(f'[retrieval_preprocessing] finished and saved retrieval indices for episode {ep_idx} [episode count {ep_count}/{total_episodes_in_grouping}]')

        myprint(f'[retrieval_preprocessing] finished for {chosen_id} [chosen_id count {chosen_id_count}/{num_groupings}]')

    myprint(f'[retrieval_preprocessing] done for {embedding_type=}!')

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--nb_cores_autofaiss", type=int, default=8)
    parser.add_argument("--knn_k", type=int, default=100, help="number of nearest neighbors to retrieve")
    parser.add_argument("--embedding_type", type=str, default="top_image", choices=EMBED_TYPES + ["both"])
    parser.add_argument("--folder_name", type=str, default="collected_demos_training")
    args = parser.parse_args()

    if args.folder_name == "collected_demos_training":
        ds_name = args.folder_name
        mappings = create_idx_fol_mapping(ds_name)

        retrieval_preprocessing(
            groups_to_ep_idxs=mappings['groups_to_ep_idxs'],
            ep_idxs_to_fol=mappings['ep_idxs_to_fol'],
            nb_cores_autofaiss=args.nb_cores_autofaiss,
            knn_k=args.knn_k,
            embedding_type=args.embedding_type
        )
        print(f'done!')

    elif args.folder_name == "collected_demos":
        all_groups_in_folder = [
            f"{args.folder_name}/{fol}"
            for fol in os.listdir(args.folder_name)
            if os.path.isdir(f"{args.folder_name}/{fol}")
        ]

        for fol_count, fol_name in enumerate(all_groups_in_folder):
            ds_name = fol_name
            mappings = create_idx_fol_mapping_for_a_single_group(ds_name)
            if mappings is None:
                continue

            retrieval_preprocessing(
                groups_to_ep_idxs=mappings['groups_to_ep_idxs'],
                ep_idxs_to_fol=mappings['ep_idxs_to_fol'],
                nb_cores_autofaiss=args.nb_cores_autofaiss,
                knn_k=args.knn_k,
                embedding_type=args.embedding_type
            )
            print(f'done for {ds_name=}! [fol count {fol_count}/{len(all_groups_in_folder)}]')

        print(f'done!')
