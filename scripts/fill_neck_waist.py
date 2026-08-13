"""
将新数据集(all_episodes)的 t=0 时刻后颈和腰部位姿填充到旧数据集(convert_whole_last)
- 后颈位置+旋转: state[58:67]
- 腰部位置+旋转: state[89:98]
同时填充 action 的对应位置，保持一致性。
"""
import h5py
import numpy as np
import glob
import os
import argparse


def get_reference_poses(new_data_file):
    """从新数据文件 t=0 时刻获取后颈和腰部位姿作为参考常量值。"""
    f = h5py.File(new_data_file, 'r')
    s = f['observation.state'][()]
    f.close()

    neck_ref = s[0, 58:67].copy()   # 后颈 t=0
    waist_ref = s[0, 89:98].copy()  # 腰部 t=0
    return neck_ref, waist_ref


def fill_single_hdf5(hdf5_path, neck_val, waist_val):
    """填充单个 hdf5 文件。同时修改 observation.state 和 action。"""
    # 以读写方式打开
    f = h5py.File(hdf5_path, 'r+')

    state = f['observation.state'][()]
    action = f['action'][()]

    # 记录填充前的状态（用于日志）
    max_neck_before = np.abs(state[:, 58:67]).max()
    max_waist_before = np.abs(state[:, 89:98]).max()

    # 填充：所有时间步的对应维度设为常量值
    state[:, 58:67] = neck_val
    state[:, 89:98] = waist_val
    action[:, 58:67] = neck_val
    action[:, 89:98] = waist_val

    # 删除旧的 dataset 并写入新的
    del f['observation.state']
    del f['action']
    f.create_dataset('observation.state', data=state, dtype=np.float64)
    f.create_dataset('action', data=action, dtype=np.float64)

    f.close()
    return max_neck_before, max_waist_before, state.shape


def main():
    parser = argparse.ArgumentParser(description='Fill old dataset (convert_whole_last) with neck/waist poses from new dataset (all_episodes)')
    parser.add_argument('--ref-file', type=str,
                        default='/home/aigc/human_policy/data/all_episodes/wholebody-10_unified_V.hdf5',
                        help='参考的新数据文件 (取其 t=0 时刻)')
    parser.add_argument('--old-dir', type=str,
                        default='/home/aigc/human_policy/data/convert_whole_last',
                        help='旧数据集目录')
    parser.add_argument('--backup', action='store_true', default=False,
                        help='是否在修改前备份原文件（不默认备份，节省空间）')
    args = parser.parse_args()

    # 获取参考值
    print('=' * 70)
    print('参考文件: ' + args.ref_file)
    neck_ref, waist_ref = get_reference_poses(args.ref_file)
    print(f'后颈(58:67) 参考值 = {neck_ref}')
    print(f'腰部(89:98) 参考值 = {waist_ref}')
    print('=' * 70)
    print()

    # 遍历所有旧数据文件
    old_files = sorted(glob.glob(os.path.join(args.old_dir, '*.hdf5')))
    print(f'发现 {len(old_files)} 个待处理文件')
    print()

    success = 0
    fail = 0
    for i, fpath in enumerate(old_files):
        fname = os.path.basename(fpath)
        try:
            if args.backup:
                backup_path = fpath + '.bak'
                if not os.path.exists(backup_path):
                    with h5py.File(fpath, 'r') as src, h5py.File(backup_path, 'w') as dst:
                        for k in src.keys():
                            src.copy(k, dst)
                        for k, v in src.attrs.items():
                            dst.attrs[k] = v

            max_neck, max_waist, shape = fill_single_hdf5(fpath, neck_ref, waist_ref)
            print(f'[{i+1:4d}/{len(old_files):4d}] {fname:<40s} '
                  f'shape={shape}  填充前 neck_max={max_neck:.6f} waist_max={max_waist:.6f}  OK')
            success += 1
        except Exception as e:
            print(f'[{i+1:4d}/{len(old_files):4d}] {fname:<40s}  FAIL: {e}')
            fail += 1

    print()
    print('=' * 70)
    print(f'完成！成功 {success}, 失败 {fail}')
    print('=' * 70)


if __name__ == '__main__':
    main()
