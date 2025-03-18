import numpy as np
import cv2
import h5py

def load_cmd_tuple_hdf5(path, img_key="observation.image.left"):
    data_list = []

    with h5py.File(path, 'r') as file:
        # Processed HDF5
        # Show two potential options
        assert "observation.image.left" in file
        assert "observation.image.right" in file
        
        # Get images
        compresses_img_arr = file[img_key][()]
        img_list = []
        for i in range(compresses_img_arr.shape[0]):
            cur_img = cv2.imdecode(compresses_img_arr[i], cv2.IMREAD_COLOR)
            img_list.append(cur_img)

    return img_list

def main(input_file, dt):
    # Processed HDF5
    img_list = load_cmd_tuple_hdf5(input_file)
    
    for i in range(len(img_list)):
        # By default, the image is stored in RGB format in HDF5
        rgb_image = img_list[i]
        # Convert to BGR for display in OpenCV
        bgr_image = rgb_image[:, :, ::-1]
        cv2.imshow("Image", bgr_image)
        cv2.waitKey(dt)
    
    cv2.destroyAllWindows()

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='Plot processed data from HDF5 file')
    parser.add_argument('--file', '-f', type=str, 
                        default="./recordings/processed/303-grasp_coke_random-2024_12_12-19_13_53/processed_episode_10.hdf5",
                        help='Path to the processed HDF5 file')
    parser.add_argument('--dt', '-dt', type=int, default=15, help="Image playing frequency (in MS)")
    args = parser.parse_args()

    main(args.file, args.dt)
