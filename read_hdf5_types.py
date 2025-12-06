#!/usr/bin/env python3
"""
HDF5数据类型读取脚本
用于读取和显示HDF5文件中所有数据集的类型、形状和属性信息
"""

import h5py
import numpy as np
import argparse
from pathlib import Path


def print_attrs(name, obj, indent=0):
    """打印HDF5对象的属性"""
    prefix = "  " * indent
    if obj.attrs:
        print(f"{prefix}属性:")
        for key, val in obj.attrs.items():
            print(f"{prefix}  - {key}: {val} (类型: {type(val).__name__})")


def print_dataset_info(name, obj, indent=0):
    """打印数据集的详细信息"""
    prefix = "  " * indent
    print(f"{prefix}📊 数据集: {name}")
    print(f"{prefix}  ├─ 数据类型: {obj.dtype}")
    print(f"{prefix}  ├─ 形状: {obj.shape}")
    print(f"{prefix}  ├─ 大小: {obj.size} 个元素")
    
    # # 如果数据集很小，显示一些样本数据
    # if obj.size > 0 and obj.size <= 10:
    #     print(f"{prefix}  ├─ 数据: {obj[...]}")
    # elif obj.size > 0:
    #     # 显示第一个元素作为样例
    #     try:
    #         if len(obj.shape) == 1:
    #             print(f"{prefix}  ├─ 样例数据 (前3个): {obj[:min(3, obj.shape[0])]}")
    #         else:
    #             print(f"{prefix}  ├─ 样例数据 (首项): {obj[0]}")
    #     except:
    #         print(f"{prefix}  ├─ 样例数据: (无法读取)")
    
    # # 打印属性
    # if obj.attrs:
    #     print(f"{prefix}  └─ 属性:")
    #     for key, val in obj.attrs.items():
    #         print(f"{prefix}      - {key}: {val}")
    # else:
    #     print(f"{prefix}  └─ (无属性)")


def print_group_info(name, obj, indent=0):
    """打印组的信息"""
    prefix = "  " * indent
    print(f"{prefix}📁 组: {name if name else '根目录'}")
    
    # 打印组的属性
    if obj.attrs:
        print(f"{prefix}  属性:")
        for key, val in obj.attrs.items():
            print(f"{prefix}    - {key}: {val}")


def explore_hdf5(file_path, show_data=False, max_depth=None):
    """
    递归遍历HDF5文件并打印所有数据类型信息
    
    参数:
        file_path: HDF5文件路径
        show_data: 是否显示数据样例
        max_depth: 最大递归深度，None表示无限制
    """
    print(f"\n{'='*60}")
    print(f"HDF5文件: {file_path}")
    print(f"{'='*60}\n")
    
    with h5py.File(file_path, 'r') as f:
        # 打印文件级别的属性
        if f.attrs:
            print("📄 文件属性:")
            for key, val in f.attrs.items():
                print(f"  - {key}: {val}")
            print()
        
        def visit_func(name, obj, depth=0):
            """访问函数"""
            if max_depth is not None and depth > max_depth:
                return
            
            indent = depth
            
            if isinstance(obj, h5py.Group):
                print_group_info(name, obj, indent)
            elif isinstance(obj, h5py.Dataset):
                print_dataset_info(name, obj, indent)
            
            print()  # 空行分隔
        
        # 遍历所有对象
        def recursive_visit(group, depth=0):
            """递归访问所有组和数据集"""
            if max_depth is not None and depth > max_depth:
                return
            
            for key in group.keys():
                obj = group[key]
                full_name = f"{group.name}/{key}" if group.name != '/' else f"/{key}"
                
                if isinstance(obj, h5py.Group):
                    print_group_info(full_name, obj, depth)
                    print()
                    recursive_visit(obj, depth + 1)
                elif isinstance(obj, h5py.Dataset):
                    print_dataset_info(full_name, obj, depth)
                    print()
        
        recursive_visit(f, 0)


def list_keys(file_path, group_path='/'):
    """
    列出指定组中的所有键
    
    参数:
        file_path: HDF5文件路径
        group_path: 组路径，默认为根目录
    """
    print(f"\n{'='*60}")
    print(f"HDF5文件: {file_path}")
    print(f"组路径: {group_path}")
    print(f"{'='*60}\n")
    
    with h5py.File(file_path, 'r') as f:
        if group_path in f:
            group = f[group_path]
            print(f"组 '{group_path}' 中的键:")
            for key in group.keys():
                obj = group[key]
                if isinstance(obj, h5py.Group):
                    print(f"  📁 {key} (组)")
                elif isinstance(obj, h5py.Dataset):
                    print(f"  📊 {key} (数据集, 形状: {obj.shape}, 类型: {obj.dtype})")
        else:
            print(f"错误: 组 '{group_path}' 不存在")


def get_dataset_info(file_path, dataset_path):
    """
    获取特定数据集的详细信息
    
    参数:
        file_path: HDF5文件路径
        dataset_path: 数据集路径
    """
    print(f"\n{'='*60}")
    print(f"HDF5文件: {file_path}")
    print(f"数据集路径: {dataset_path}")
    print(f"{'='*60}\n")
    
    with h5py.File(file_path, 'r') as f:
        if dataset_path in f:
            dataset = f[dataset_path]
            if isinstance(dataset, h5py.Dataset):
                print_dataset_info(dataset_path, dataset, 0)
                
                # 显示数据的统计信息（如果是数值类型）
                if np.issubdtype(dataset.dtype, np.number):
                    data = dataset[...]
                    print(f"\n统计信息:")
                    print(f"  - 最小值: {np.min(data)}")
                    print(f"  - 最大值: {np.max(data)}")
                    print(f"  - 平均值: {np.mean(data)}")
                    print(f"  - 标准差: {np.std(data)}")
            else:
                print(f"'{dataset_path}' 不是数据集，而是一个组")
        else:
            print(f"错误: 数据集 '{dataset_path}' 不存在")


def main():
    parser = argparse.ArgumentParser(
        description='读取HDF5文件的数据类型和结构信息',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  # 查看整个文件的结构
  python read_hdf5_types.py data.hdf5
  
  # 只列出根目录的键
  python read_hdf5_types.py data.hdf5 --list-keys
  
  # 列出特定组的键
  python read_hdf5_types.py data.hdf5 --list-keys --group /observations
  
  # 查看特定数据集的信息
  python read_hdf5_types.py data.hdf5 --dataset /observations/qpos
  
  # 限制显示深度
  python read_hdf5_types.py data.hdf5 --max-depth 2
        """
    )
    
    parser.add_argument('file', type=str, help='HDF5文件路径')
    parser.add_argument('--list-keys', action='store_true', 
                        help='仅列出键，不显示详细信息')
    parser.add_argument('--group', type=str, default='/',
                        help='指定要列出的组路径 (配合 --list-keys 使用)')
    parser.add_argument('--dataset', type=str,
                        help='显示特定数据集的详细信息')
    parser.add_argument('--max-depth', type=int,
                        help='最大显示深度')
    parser.add_argument('--show-data', action='store_true',
                        help='显示数据样例')
    
    args = parser.parse_args()
    
    # 检查文件是否存在
    if not Path(args.file).exists():
        print(f"错误: 文件 '{args.file}' 不存在")
        return
    
    try:
        if args.dataset:
            # 显示特定数据集的信息
            get_dataset_info(args.file, args.dataset)
        elif args.list_keys:
            # 仅列出键
            list_keys(args.file, args.group)
        else:
            # 显示完整的文件结构
            explore_hdf5(args.file, args.show_data, args.max_depth)
    except Exception as e:
        print(f"错误: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()

    '''
    # 查看整个文件的结构
    python read_hdf5_types.py your_file.hdf5

    # 只列出根目录的键
    python read_hdf5_types.py your_file.hdf5 --list-keys

    # 列出observations组的键
    python read_hdf5_types.py your_file.hdf5 --list-keys --group /observations

    # 查看特定数据集的详细信息
    python read_hdf5_types.py your_file.hdf5 --dataset /observations/qpos

    # 限制显示深度为2层
    python read_hdf5_types.py your_file.hdf5 --max-depth 2
    '''