#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
数据库管理命令行界面
提供车辆声纹数据库的管理功能
"""

# 使用公共工具模块统一导入
from core.common_utils import (
    os, sys, time, json, Path,
    List, Dict, Optional, Tuple, Any, Union,
    dataclass, asdict, warnings, np,
    config_manager, file_manager, logger_factory, TimerContext
)

import argparse
import shutil
from datetime import datetime
import threading
import concurrent.futures

# 项目内部导入
from core.similarity_utils import FeatureProcessor, SimilarityCalculator
from core.exceptions import handle_exceptions, VehicleRecognitionError

# 初始化日志器
cli_logger = logger_factory.get_logger("数据库管理CLI")

# 添加项目根目录到路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from incremental_database_manager import IncrementalDatabaseManager, create_incremental_manager, quick_add_vehicle

class DatabaseManagerCLI:
    """数据库管理命令行界面"""
    
    def __init__(self):
        self.manager = None
        self.logger = logger_factory.get_logger("数据库管理CLI")
    
    def initialize_manager(self, database_path: str = None):
        """初始化数据库管理器"""
        try:
            print("初始化数据库管理器...")
            self.manager = create_incremental_manager(database_path)
            print("✓ 数据库管理器初始化成功")
            return True
        except Exception as e:
            print(f"✗ 初始化失败: {e}")
            self.logger.error(f"数据库管理器初始化失败: {e}")
            return False
    
    def show_database_info(self):
        """显示数据库信息"""
        if not self.manager:
            print("✗ 数据库管理器未初始化")
            return
        
        try:
            info = self.manager.get_database_info()
            
            print("\n数据库信息")
            print("=" * 60)
            print(f"数据库路径: {info['database_path']}")
            print(f"车辆总数: {info['total_vehicles']}")
            print(f"音频文件总数: {info['total_files']}")
            print(f"特征维度: {info['feature_dimension']}")
            print(f"数据库大小: {info['database_size_mb']:.2f} MB")
            print(f"索引大小: {info['index_size_mb']:.2f} MB")
            print(f"最后更新: {info['last_updated']}")
            
            # 显示车辆分布
            if info.get('vehicle_distribution'):
                print("\n车辆分布:")
                print("-" * 40)
                for vehicle_id, count in sorted(info['vehicle_distribution'].items()):
                    print(f"  {vehicle_id}: {count} 个文件")
            
        except Exception as e:
            print(f"✗ 获取数据库信息失败: {e}")
            self.logger.error(f"获取数据库信息失败: {e}")
    
    def add_vehicle_interactive(self):
        """交互式添加车辆"""
        if not self.manager:
            print("✗ 数据库管理器未初始化")
            return
        
        try:
            # 获取车辆ID
            vehicle_id = input("请输入车辆ID: ").strip()
            if not vehicle_id:
                print("✗ 车辆ID不能为空")
                return
            
            # 获取音频文件
            print("\n选择音频文件来源:")
            print("1. 单个文件")
            print("2. 目录中的所有音频文件")
            
            choice = input("请选择 (1-2): ").strip()
            
            audio_files = []
            
            if choice == "1":
                file_path = input("请输入音频文件路径: ").strip()
                if not file_path:
                    return
                
                if not os.path.exists(file_path):
                    print("✗ 文件不存在")
                    return
                
                audio_files = [file_path]
                
            elif choice == "2":
                dir_path = input("请输入目录路径: ").strip()
                if not dir_path:
                    return
                
                if not os.path.exists(dir_path):
                    print("✗ 目录不存在")
                    return
                
                # 查找音频文件
                audio_extensions = ['.wav', '.mp3', '.flac', '.m4a', '.aac']
                for ext in audio_extensions:
                    audio_files.extend(Path(dir_path).glob(f"*{ext}"))
                    audio_files.extend(Path(dir_path).glob(f"*{ext.upper()}"))
                
                if not audio_files:
                    print("✗ 目录中未找到音频文件")
                    return
                
                # 转换为字符串路径
                audio_files = [str(f) for f in audio_files]
                
            else:
                print("✗ 无效选择")
                return
            
            # 显示文件列表
            print(f"\n找到 {len(audio_files)} 个音频文件:")
            for i, file_path in enumerate(audio_files[:10], 1):  # 只显示前10个
                print(f"  {i}. {Path(file_path).name}")
            
            if len(audio_files) > 10:
                print(f"  ... 还有 {len(audio_files) - 10} 个文件")
            
            # 确认添加
            print(f"\n添加确认:")
            print(f"车辆ID: {vehicle_id}")
            print(f"音频文件数: {len(audio_files)}")
            
            confirm = input("确认添加? (y/N): ").strip().lower()
            if confirm not in ['y', 'yes']:
                print("取消添加")
                return
            
            print("\n正在添加车辆数据...")
            
            # 添加车辆数据
            success, failed_files = self.manager.add_vehicle_data(
                vehicle_id=vehicle_id,
                audio_paths=audio_files,
                show_progress=True
            )
            
            # 构造结果格式以保持兼容性
            result = {
                'success': success,
                'successful_count': len(audio_files) - len(failed_files),
                'failed_count': len(failed_files),
                'failed_files': {f: "处理失败" for f in failed_files}
            }
            
            if result['success']:
                print(f"✓ 车辆数据添加成功! 成功: {result['successful_count']}, 失败: {result['failed_count']}")
                self.logger.info(f"车辆数据添加成功: {vehicle_id}, 成功: {result['successful_count']}, 失败: {result['failed_count']}")
            else:
                print(f"⚠ 车辆数据部分添加成功 - 成功: {result['successful_count']}, 失败: {result['failed_count']}")
                
                # 显示失败的文件
                if result.get('failed_files'):
                    print("\n失败的文件:")
                    for file_path, error in result['failed_files'].items():
                        print(f"  {Path(file_path).name}: {error}")
            
        except KeyboardInterrupt:
            print("\n操作被用户取消")
        except Exception as e:
            print(f"✗ 添加车辆数据失败: {e}")
            self.logger.error(f"添加车辆数据失败: {e}")
    
    def remove_vehicle_interactive(self):
        """交互式删除车辆"""
        if not self.manager:
            print("✗ 数据库管理器未初始化")
            return
        
        try:
            print("\n删除车辆数据")
            print("=" * 60)
            
            # 显示现有车辆
            info = self.manager.get_database_info()
            if not info.get('vehicle_distribution'):
                print("✗ 数据库中没有车辆数据")
                return
            
            print("现有车辆:")
            for i, (vehicle_id, count) in enumerate(sorted(info['vehicle_distribution'].items()), 1):
                print(f"  {i}. {vehicle_id} ({count} 个文件)")
            
            # 获取要删除的车辆ID
            vehicle_id = input("\n请输入要删除的车辆ID: ").strip()
            if not vehicle_id:
                print("✗ 车辆ID不能为空")
                return
            
            if vehicle_id not in info['vehicle_distribution']:
                print(f"✗ 车辆不存在: {vehicle_id}")
                return
            
            file_count = info['vehicle_distribution'][vehicle_id]
            
            print(f"\n删除确认:")
            print(f"车辆ID: {vehicle_id}")
            print(f"将删除 {file_count} 个音频文件的数据")
            
            confirm = input("确认删除? (y/N): ").strip().lower()
            if confirm not in ['y', 'yes']:
                print("取消删除")
                return
            
            print("\n正在删除车辆数据...")
            
            # 删除车辆数据
            success = self.manager.remove_vehicle(vehicle_id)
            
            if success:
                print(f"✓ 车辆数据删除成功: {vehicle_id}")
                self.logger.info(f"车辆数据删除成功: {vehicle_id}")
            else:
                print(f"✗ 车辆数据删除失败")
            
        except KeyboardInterrupt:
            print("\n操作被用户取消")
        except Exception as e:
            print(f"✗ 删除车辆数据失败: {e}")
            self.logger.error(f"删除车辆数据失败: {e}")
    
    def batch_add_from_directory(self):
        """从目录批量添加车辆数据"""
        if not self.manager:
            print("✗ 数据库管理器未初始化")
            return
        
        try:
            print("\n批量添加车辆数据")
            print("=" * 60)
            
            # 获取根目录
            root_dir = input("请输入包含车辆目录的根目录路径: ").strip()
            if not root_dir or not os.path.exists(root_dir):
                print("✗ 目录不存在")
                return
            
            # 查找子目录
            subdirs = [d for d in Path(root_dir).iterdir() if d.is_dir()]
            if not subdirs:
                print("✗ 目录中没有子目录")
                return
            
            print(f"\n找到 {len(subdirs)} 个子目录:")
            for i, subdir in enumerate(subdirs[:20], 1):  # 只显示前20个
                print(f"  {i}. {subdir.name}")
            
            if len(subdirs) > 20:
                print(f"  ... 还有 {len(subdirs) - 20} 个目录")
            
            confirm = input(f"\n确认批量添加这些目录? (y/N): ").strip().lower()
            if confirm not in ['y', 'yes']:
                print("取消批量添加")
                return
            
            print("\n正在批量添加车辆数据...")
            
            # 批量添加
            results = self.manager.batch_add_from_directory(
                directory=root_dir
            )
            
            print(f"\n✓ 批量添加完成!")
            print(f"总目录数: {results['total_vehicles']}")
            print(f"成功: {results['successful_vehicles']}")
            print(f"失败: {results['failed_vehicles']}")
            
            self.logger.info(f"批量添加完成: 总目录数: {results['total_vehicles']}, 成功: {results['successful_vehicles']}, 失败: {results['failed_vehicles']}")
            
            # 显示详细结果
            for vehicle_id, detail in results['details'].items():
                status = "✓" if detail['success'] else "✗"
                print(f"  {status} {vehicle_id} ({detail['successful_files']} 个文件)")
            
        except KeyboardInterrupt:
            print("\n操作被用户取消")
        except Exception as e:
            print(f"✗ 批量添加失败: {e}")
            self.logger.error(f"批量添加失败: {e}")
    
    def show_recent_changes(self):
        """显示最近的数据库变更"""
        if not self.manager:
            print("✗ 数据库管理器未初始化")
            return
        
        try:
            changes = self.manager.get_recent_changes(limit=20)
            
            if not changes:
                print("没有找到最近的变更记录")
                return
            
            print("\n最近的数据库变更")
            print("=" * 80)
            print(f"{'时间':<20} {'操作':<15} {'车辆ID':<20} {'详情':<25}")
            print("-" * 80)
            
            for change in changes:
                timestamp = change.get('timestamp', 'N/A')
                operation = change.get('operation', 'N/A')
                vehicle_id = change.get('vehicle_id', 'N/A')
                details = change.get('details', 'N/A')
                
                # 格式化时间戳
                if timestamp != 'N/A':
                    try:
                        dt = datetime.fromisoformat(timestamp.replace('Z', '+00:00'))
                        timestamp = dt.strftime('%Y-%m-%d %H:%M:%S')
                    except:
                        pass
                
                print(f"{timestamp:<20} {operation:<15} {vehicle_id:<20} {str(details)[:25]:<25}")
                
        except Exception as e:
            print(f"✗ 获取变更记录失败: {e}")
            self.logger.error(f"获取变更记录失败: {e}")
    

    
    def optimize_database(self):
        """优化数据库"""
        if not self.manager:
            print("✗ 数据库管理器未初始化")
            return
        
        try:
            print("数据库优化将重建索引并清理无效数据")
            confirm = input("确认优化数据库? (y/N): ").strip().lower()
            if confirm not in ['y', 'yes']:
                print("取消优化")
                return
            
            print("正在优化数据库...")
            
            start_time = time.time()
            success = self.manager.optimize_database()
            elapsed_time = time.time() - start_time
            
            if success:
                print(f"✓ 数据库优化完成! 耗时: {elapsed_time:.2f}s")
                self.logger.info(f"数据库优化完成，耗时: {elapsed_time:.2f}s")
            else:
                print("✗ 数据库优化失败")
                
        except Exception as e:
            print(f"✗ 数据库优化失败: {e}")
            self.logger.error(f"数据库优化失败: {e}")
    
    def run_interactive(self):
        """运行交互式界面"""
        print("车辆音频数据库管理系统")
        print("=" * 60)
        
        try:
            while True:
                print("\n" + "=" * 60)
                print("主菜单")
                print("=" * 60)
                print("1. 查看数据库信息")
                print("2. 添加车辆数据")
                print("3. 删除车辆数据")
                print("4. 批量添加车辆数据")
                print("5. 查看最近变更")
                print("6. 优化数据库")
                print("7. 退出")
                
                choice = input("\n请选择操作 (1-7): ").strip()
                
                if choice == "1":
                    self.show_database_info()
                elif choice == "2":
                    self.add_vehicle_interactive()
                elif choice == "3":
                    self.remove_vehicle_interactive()
                elif choice == "4":
                    self.batch_add_from_directory()
                elif choice == "5":
                    self.show_recent_changes()
                elif choice == "6":
                    self.optimize_database()
                elif choice == "7":
                    print("退出程序")
                    break
                else:
                    print("✗ 无效选择，请重新输入")
                    
        except KeyboardInterrupt:
            print("\n程序被用户中断")
        except Exception as e:
            print(f"✗ 程序运行错误: {e}")
            self.logger.error(f"程序运行错误: {e}")

def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="车辆音频数据库管理工具")
    parser.add_argument("--database", "-d", type=str, help="数据库文件路径")
    parser.add_argument("--batch-add", type=str, help="批量添加目录路径")
    parser.add_argument("--vehicle-id", type=str, help="车辆ID")
    parser.add_argument("--audio-files", nargs="+", help="音频文件路径列表")
    parser.add_argument("--info", action="store_true", help="显示数据库信息")
    parser.add_argument("--optimize", action="store_true", help="优化数据库")
    
    args = parser.parse_args()
    
    cli = DatabaseManagerCLI()
    
    # 初始化数据库管理器
    if not cli.initialize_manager(args.database):
        return 1
    
    try:
        if args.info:
            cli.show_database_info()
        elif args.optimize:
            cli.optimize_database()
        elif args.batch_add:
            # 批量添加模式
            results = cli.manager.batch_add_from_directory(
                directory=args.batch_add
            )
            success_count = results.get('successful_vehicles', 0)
            failed_count = results.get('failed_vehicles', 0)
            print(f"批量添加完成: 成功 {success_count}, 失败 {failed_count}")
        elif args.vehicle_id and args.audio_files:
            # 单个车辆添加模式
            success, failed_files = cli.manager.add_vehicle_data(
                vehicle_id=args.vehicle_id,
                audio_paths=args.audio_files,
                show_progress=True
            )
            if success:
                print(f"✓ 车辆添加成功: {args.vehicle_id}")
            else:
                print(f"✗ 车辆添加失败: {args.vehicle_id}")
        else:
            # 交互模式
            cli.run_interactive()
            
    except KeyboardInterrupt:
        print("\n程序被用户中断")
        return 1
    except Exception as e:
        print(f"✗ 程序执行错误: {e}")
        cli_logger.error(f"程序执行错误: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main())