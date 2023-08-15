import os
import shutil
import pandas as pd

if __name__ == "__main__":
    runs = [
    #     "202305-2911-2320-5a95df0e-3008-4ebe-acd8-ecb3b50607c7", # PAROUTES (10K)
        "202305-3013-3922-91ae99d6-b41d-4092-8b31-246c7703f50f", # GUACAMOL (0-1396)
        "202305-3117-3758-b1be5935-f3f0-43dd-bc2e-49cb5bb1772c", # GUACAMOL (1397-2816)
        # "202306-0213-2105-5c0620c9-4b06-434f-ba5e-23d772a1f6e8", # Nothing
        "202306-0213-2303-7521e227-dc59-48bf-b3cb-bbf90bd07bec", # GUACAMOL (2816-2893)
    #     "202306-0306-4611-954de39c-6558-4550-9c67-db555d6d1d34_GRAPHS", # Visual
        "202306-0514-3401-008660ba-04f4-4436-abd2-9ce3ff851b28", # GUACAMOL (2893-3104)
        "202306-0609-0553-530e6359-3abb-46fc-ae71-ad78074b44fd", # GUACAMOL (3104-3323)
        "202306-0611-5359-7461876f-5c2b-46e4-8dd2-5af7ffeb819e", # GUACAMOL (3323-3526)
        "202306-0614-2421-d1c794db-3837-46d3-aaa7-7d15f33b9098", # GUACAMOL (3527-3726)
        "202306-0619-4747-63b3c7e8-1ed9-4126-adf1-a1cedddf4b89", # GUACAMOL (3727-3938)
        "202306-0623-5220-30d27064-9736-4889-a3ff-60cb5605ee77", # GUACAMOL (3938-4162)
    #     "202306-1400-2128-ae3826a3-5d49-4738-884b-b14efe2ebff6_ROUTES_3_STEPS", # Visual
    #     "202306-1400-2808-5013f9e7-8964-4bda-bd02-88a683e04371_ROUTES_3_STEPS_whole3template", # Visual
    #     "202306-1400-3031-4b360196-b1bd-4370-92d8-b94ea263db5d_ROUTES_3_STEPS_whole4template", # Visual
        "202306-1718-5019-ec003f2a-37d2-4ea5-a166-42cbdd2b4ca3", # GUACAMOL (4162-4381)
        "202306-1813-3327-86b8e174-6b6b-4905-8a2e-8460c84b3183", # GUACAMOL (4381-4547)
        "202306-1815-4941-332f69e1-86d9-4cda-ab22-daa5f710f7fc", # GUACAMOL (4546-4741)
        "202306-1817-1357-84b60861-3426-4668-a772-b1b62f8b2189", # GUACAMOL (4740-6308)
        "202306-1909-0914-b71b5898-64bd-4b25-8452-65bf6974d97c", # GUACAMOL (6309-7524)
        "202306-2003-0025-f30ff715-bd6e-4b9c-a5ad-3163b053016b", # GUACAMOL (7524-9037)
        "202306-2017-4122-3d173839-86e9-4dfc-a457-535a9d75ebd5", # GUACAMOL (9037-9999)
    ]
    
    output_folder = f'Runs/Guacamol_combined'
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    tot_result_df = pd.DataFrame()
    
    for run_id in runs:
        input_folder = f'Runs/{run_id}'
        result_df = pd.read_csv(f'{input_folder}/{run_id}_results.csv')
        tot_result_df = pd.concat([tot_result_df, result_df], ignore_index=True)
        graphs_folder = f'{input_folder}/constant0_graph_pickles'
        
        output_graphs_folder = f'{output_folder}/constant0_graph_pickles'
        os.makedirs(output_graphs_folder, exist_ok=True)
        
        pickle_files = [file for file in os.listdir(graphs_folder) if '.pickle' in file]

        # Copy each pickle file to the output folder
        for file in pickle_files:
            src_path = os.path.join(graphs_folder, file)
            dst_path = os.path.join(output_graphs_folder, file)
            shutil.copy2(src_path, dst_path)
        
        # for graph_pkl in [file for file in os.listdir(f'{graphs_folder}') if '.pickle' in file]:
        #     pass
        # print(graph_pkl)
    #         with open(graph_pkl, 'rb') as handle:
    #             result[name] = pickle.load(handle)
    
    tot_result_df = tot_result_df.drop_duplicates(subset='smiles', keep='last')
    tot_result_df.to_csv(f'{output_folder}/Guacamol_combined_results.csv', index=False)
        
    
    
    