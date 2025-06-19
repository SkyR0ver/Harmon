#!/usr/bin/env python3
"""
CC3M 数据处理脚本：按原始名称组织数据集，避免重复
"""

import json
import shutil
import tarfile
import tempfile
from pathlib import Path
import sys

class CC3MDatasetOrganizer:
    def __init__(self, dataset_dir="../dataset/cc3m", target_dir="../data/cc3m_train", target_count=1000):
        self.dataset_dir = Path(dataset_dir)
        self.target_dir = Path(target_dir)
        self.target_count = target_count
        self.temp_extract_dir = None
        
        # 确保目标目录存在
        self.local_folder = self.target_dir / "local_folder" / "000000"
        self.cap_folder = self.target_dir / "cap_folder" / "000000"
        self.local_folder.mkdir(parents=True, exist_ok=True)
        self.cap_folder.mkdir(parents=True, exist_ok=True)
        
        # 用于跟踪已处理的原始文件名，避免重复
        self.processed_files = set()
        self.original_to_new_mapping = {}  # 原始名称 -> 新名称的映射
        
    def load_existing_data_info(self):
        """加载现有的数据信息，建立原始名称映射"""
        data_info_path = self.target_dir / "data_info.json"
        existing_data = []
        
        if data_info_path.exists():
            try:
                with open(data_info_path, 'r', encoding='utf-8') as f:
                    existing_data = json.load(f)
                    
                # 如果有现有的映射文件，加载它
                mapping_path = self.target_dir / "original_mapping.json"
                if mapping_path.exists():
                    with open(mapping_path, 'r', encoding='utf-8') as f:
                        self.original_to_new_mapping = json.load(f)
                        self.processed_files = set(self.original_to_new_mapping.keys())
                else:
                    # 如果没有映射文件，但有数据，说明是旧格式，需要重建映射
                    # 这种情况下我们无法确定原始名称，所以从头开始
                    if not self.quiet:
                        print("⚠️ 检测到旧格式数据，将重新组织数据集")
                    
            except (FileNotFoundError, json.JSONDecodeError, IOError) as e:
                if not self.quiet:
                    print(f"无法加载现有数据: {e}")
                existing_data = []
                
        return existing_data
    
    def save_mapping(self):
        """保存原始名称到新名称的映射"""
        mapping_path = self.target_dir / "original_mapping.json"
        with open(mapping_path, 'w', encoding='utf-8') as f:
            json.dump(self.original_to_new_mapping, f, ensure_ascii=False, indent=2)
    
    def find_available_tar_files(self):
        """查找可用的 cc3m-train tar 文件"""
        tar_files = sorted(self.dataset_dir.glob("cc3m-train-*.tar"))
        print(f"找到 {len(tar_files)} 个 tar 文件")
        return tar_files
    
    def extract_samples_from_tar(self, tar_file, needed_count):
        """从 tar 文件中提取指定数量的样本，跳过已处理的文件"""
        extracted_samples = []
        
        try:
            with tarfile.open(tar_file, 'r') as tar:
                members = tar.getmembers()
                members.sort(key=lambda x: x.name)
                
                # 创建临时解压目录
                if self.temp_extract_dir is None:
                    self.temp_extract_dir = tempfile.mkdtemp(prefix="cc3m_extract_")
                
                sample_count = 0
                i = 0
                while i < len(members) and sample_count < needed_count:
                    member = members[i]
                    if member.name.endswith('.jpg'):
                        # 获取基础名称（去掉扩展名）
                        base_name = Path(member.name).stem
                        
                        # 检查是否已经处理过这个文件
                        if base_name in self.processed_files:
                            i += 1
                            continue
                        
                        # 查找对应的 json 文件
                        json_member = None
                        for j in range(max(0, i-5), min(len(members), i+6)):
                            if members[j].name == member.name[:-4] + '.json':
                                json_member = members[j]
                                break
                        
                        if json_member:
                            try:
                                # 解压文件
                                tar.extract(member, self.temp_extract_dir)
                                tar.extract(json_member, self.temp_extract_dir)
                                
                                img_path = Path(self.temp_extract_dir) / member.name
                                json_path = Path(self.temp_extract_dir) / json_member.name
                                
                                if img_path.exists() and json_path.exists():
                                    extracted_samples.append({
                                        'img_path': img_path,
                                        'json_path': json_path,
                                        'original_name': base_name
                                    })
                                    sample_count += 1
                                    
                            except Exception:
                                pass
                    i += 1
                
        except Exception:
            pass
        
        return extracted_samples
    
    def process_extracted_samples(self, samples, current_count):
        """处理解压出的样本，按原始名称组织"""
        existing_data = self.load_existing_data_info()
        success_count = 0
        
        for sample in samples:
            try:
                # 读取原始 JSON 文件
                with open(sample['json_path'], 'r', encoding='utf-8') as f:
                    original_data = json.load(f)
                
                # 检查数据是否有效
                if original_data.get('status') != 'success' or not original_data.get('caption'):
                    continue
                
                # 检查是否已经处理过（双重检查）
                if sample['original_name'] in self.processed_files:
                    continue
                
                # 生成新的文件编号
                new_file_id = f"{current_count + success_count + 1:07d}"
                
                # 目标文件路径
                target_img = self.local_folder / f"{new_file_id}.jpg"
                target_json = self.cap_folder / f"{new_file_id}.json"
                
                # 检查目标文件是否已存在（避免覆盖）
                if target_img.exists() or target_json.exists():
                    continue
                
                # 复制图片文件
                shutil.copy2(sample['img_path'], target_img)
                
                # 创建新的 caption JSON 文件
                caption_data = {'caption': original_data['caption']}
                with open(target_json, 'w', encoding='utf-8') as f:
                    json.dump(caption_data, f, ensure_ascii=False, indent=2)
                
                # 添加到数据信息
                existing_data.append({
                    'image': f"000000/{new_file_id}.jpg",
                    'annotation': f"000000/{new_file_id}.json"
                })
                
                # 记录映射关系
                self.original_to_new_mapping[sample['original_name']] = new_file_id
                self.processed_files.add(sample['original_name'])
                
                success_count += 1
                
                if (current_count + success_count) % 100 == 0:
                    print(f"已处理 {current_count + success_count} 个文件")
                
            except Exception as e:
                print(f"处理样本 {sample['original_name']} 时出错: {e}")
                continue
        
        # 更新 data_info.json
        data_info_path = self.target_dir / "data_info.json"
        with open(data_info_path, 'w', encoding='utf-8') as f:
            json.dump(existing_data, f, ensure_ascii=False, indent=2)
        
        # 保存映射文件
        self.save_mapping()
        
        return success_count
    
    def cleanup_temp_dir(self):
        """清理临时目录"""
        if self.temp_extract_dir and Path(self.temp_extract_dir).exists():
            shutil.rmtree(self.temp_extract_dir)
    
    def get_current_data_count(self):
        """获取当前已有的数据数量"""
        data_info_path = self.target_dir / "data_info.json"
        if data_info_path.exists():
            try:
                with open(data_info_path, 'r', encoding='utf-8') as f:
                    data_info = json.load(f)
                return len(data_info)
            except (FileNotFoundError, json.JSONDecodeError, IOError):
                return 0
        return 0
    
    def process(self):
        """主处理流程"""
        print("=" * 50)
        print("CC3M 数据集组织器")
        print("=" * 50)
        
        self.load_existing_data_info()
        current_count = self.get_current_data_count()
        needed_count = self.target_count - current_count
        
        print(f"当前数据: {current_count}")
        print(f"目标数据: {self.target_count}")
        print(f"需要处理: {needed_count}")
        
        if needed_count <= 0:
            print("✅ 已达到目标数据数量")
            return
        
        tar_files = self.find_available_tar_files()
        if not tar_files:
            print("❌ 未找到 tar 文件")
            return
        
        try:
            processed_samples = 0
            
            for i, tar_file in enumerate(tar_files):
                if processed_samples >= needed_count:
                    break
                
                print(f"\n处理第 {i+1}/{len(tar_files)} 个文件: {tar_file.name}")
                
                remaining_needed = needed_count - processed_samples
                samples_to_extract = min(remaining_needed, 1000)
                
                extracted_samples = self.extract_samples_from_tar(tar_file, samples_to_extract)
                
                if extracted_samples:
                    print(f"提取了 {len(extracted_samples)} 个样本")
                    success_count = self.process_extracted_samples(extracted_samples, current_count + processed_samples)
                    processed_samples += success_count
                    print(f"✅ 成功处理 {success_count} 个样本，累计: {current_count + processed_samples}")
                else:
                    print("⚠️ 未提取到有效样本")
                
                if current_count + processed_samples >= self.target_count:
                    print(f"✅ 已达到目标: {current_count + processed_samples}")
                    break
            
            final_count = self.get_current_data_count()
            print(f"\n处理完成！最终数量: {final_count}")
            
        except KeyboardInterrupt:
            print("\n用户中断")
        except Exception as e:
            print(f"处理出错: {e}")
        finally:
            self.cleanup_temp_dir()

def main():
    target_count = 20000
    
    i = 1
    while i < len(sys.argv):
        if sys.argv[i] == '--help' or sys.argv[i] == '-h':
            print("用法: python3 cc3m_dataset_organizer.py [数量]")
            print("示例: python3 cc3m_dataset_organizer.py 1000")
            sys.exit(0)
        else:
            try:
                target_count = int(sys.argv[i])
            except ValueError:
                print(f"错误：'{sys.argv[i]}' 不是有效的数字")
                sys.exit(1)
        i += 1
    
    organizer = CC3MDatasetOrganizer(target_count=target_count)
    organizer.process()

if __name__ == "__main__":
    main()
