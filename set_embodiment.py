import os
import h5py

def set_embodiment_for_dir(data_dir, embodiment_value):
    hdf_files = [f for f in os.listdir(data_dir) if f.endswith('.hdf5')]
    hdf_files.sort()
    print(f"\nProcessing {len(hdf_files)} files in {data_dir}")
    
    for filename in hdf_files:
        filepath = os.path.join(data_dir, filename)
        try:
            with h5py.File(filepath, 'r+') as f:
                old_embodiment = f.attrs.get('embodiment', 'NOT_SET')
                f.attrs['embodiment'] = embodiment_value
                print(f"  {filename}: {old_embodiment} -> {embodiment_value}")
        except Exception as e:
            print(f"  {filename}: ERROR - {e}")

if __name__ == '__main__':
    brainco_dirs = [
        '/home/aigc/human_policy/data/pickup_pillow_data/brainco',
        '/home/aigc/human_policy/data/pickup_pillow_data/brainco_train',
        '/home/aigc/human_policy/data/pickup_pillow_data/brainco_val',
    ]
    
    dex5_dirs = [
        '/home/aigc/human_policy/data/pickup_pillow_data/dex5',
        '/home/aigc/human_policy/data/pickup_pillow_data/dex5_train',
        '/home/aigc/human_policy/data/pickup_pillow_data/dex5_val',
    ]
    
    human_dir = '/home/aigc/human_policy/data3'
    
    for dir_path in brainco_dirs:
        if os.path.exists(dir_path):
            set_embodiment_for_dir(dir_path, 'brainco')
    
    for dir_path in dex5_dirs:
        if os.path.exists(dir_path):
            set_embodiment_for_dir(dir_path, 'dex5')
    
    if os.path.exists(human_dir):
        set_embodiment_for_dir(human_dir, 'human')
    
    print("\nDone!")