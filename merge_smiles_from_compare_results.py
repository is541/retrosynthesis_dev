import os
import pickle
from compare_value_functions_run_splitInputSmiles import SearchResult

def read_pickle_files(folder_path, alg_name):
    # Initialize variables
    chunk_result = {}
    total_results = {'name': alg_name, 'soln_time_dict': {}}

    # Loop through files in the folder
    for filename in os.listdir(folder_path):
        if filename.startswith(f"result_{alg_name}_") and filename.endswith(".pickle"):
            file_path = os.path.join(folder_path, filename)
            with open(file_path, 'rb') as file:
                chunk_result = pickle.load(file)

                # Check if chunk_result['name'] is equal to alg_name
                if chunk_result.name != alg_name:
                    raise ValueError(f"Error: '{filename}' does not match the specified 'alg_name'")

                # Merge soln_time_dict into total_results['soln_time_dict']
                total_results['soln_time_dict'].update(chunk_result.soln_time_dict)
    
    total_results_sr = SearchResult(name=total_results['name'], soln_time_dict=total_results['soln_time_dict'] )

    # Save total_results to a new pickle file
    output_filename = f"result_{alg_name}.pickle"
    with open(f'{folder_path}/{output_filename}', 'wb') as output_file:
        pickle.dump(total_results_sr, output_file)

if __name__ == "__main__":
    folder_path = 'CompareTanimotoLearnt/MID_EASY_v3_InventPercent_0.5_ReduceCalls/'
    # folder_path = 'CompareTanimotoLearnt/MID_EASY_v3_ReduceCalls/'
    # alg_name = 'constant-0'
    # alg_name = 'Retro*'
    # alg_name = 'Embedding-from-fingerprints-TIMES100'
    # alg_name = 'Tanimoto-distance-TIMES100'
    # alg_name = 'Embedding-from-fingerprints-TIMES100'
    alg_name = 'Embedding-from-gnn-TIMES100'
    read_pickle_files(folder_path, alg_name)
