#!/usr/bin/env python3
"""
CC3M 验证集数据提取脚本：从验证集 tar 文件中按需提取若干记录
"""

import shutil
import tarfile
import tempfile
from pathlib import Path
import sys

class CC3MValidationExtractor:
    def __init__(self, dataset_dir="../dataset/cc3m", target_dir="../data/cc3m_validation", target_count=100):
        self.dataset_dir = Path(dataset_dir)
        self.target_dir = Path(target_dir)
        self.target_count = target_count
        self.temp_extract_dir = None
        
        # 确保目标目录存在
        self.images_dir = self.target_dir / "images"
        self.texts_dir = self.target_dir / "texts"
        self.images_dir.mkdir(parents=True, exist_ok=True)
        self.texts_dir.mkdir(parents=True, exist_ok=True)
        
        # 用于跟踪已处理的文件，避免重复
        self.processed_files = set()
        
    def get_existing_count(self):
        """获取当前已有的文件数量"""
        if self.images_dir.exists():
            return len(list(self.images_dir.glob("*.jpg")))
        return 0
    
    def load_existing_files(self):
        """加载已存在的文件，避免重复处理"""
        existing_count = 0
        if self.images_dir.exists():
            existing_count = len(list(self.images_dir.glob("*.jpg")))
        print(f"当前已有 {existing_count} 个文件")
    
    def find_validation_tar_files(self):
        """查找可用的 cc3m-validation tar 文件"""
        tar_files = sorted(self.dataset_dir.glob("cc3m-validation-*.tar"))
        print(f"找到 {len(tar_files)} 个验证集 tar 文件")
        return tar_files
    
    def extract_samples_from_tar(self, tar_file, needed_count):
        """从 tar 文件中提取指定数量的样本"""
        extracted_samples = []
        
        try:
            with tarfile.open(tar_file, 'r') as tar:
                members = tar.getmembers()
                
                # 创建临时解压目录
                if self.temp_extract_dir is None:
                    self.temp_extract_dir = tempfile.mkdtemp(prefix="cc3m_val_extract_")
                
                sample_count = 0
                for member in members:
                    if sample_count >= needed_count:
                        break
                        
                    if member.name.endswith('.jpg'):
                        # 获取基础名称（去掉扩展名）
                        base_name = Path(member.name).stem
                        
                        # 检查是否已经处理过这个文件
                        if base_name in self.processed_files:
                            continue
                        
                        # 查找对应的 txt 文件（验证集使用txt而不是json）
                        txt_name = member.name[:-4] + '.txt'
                        txt_member = None
                        for m in members:
                            if m.name == txt_name:
                                txt_member = m
                                break
                        
                        if txt_member:
                            try:
                                # 解压文件
                                tar.extract(member, self.temp_extract_dir)
                                tar.extract(txt_member, self.temp_extract_dir)
                                
                                img_path = Path(self.temp_extract_dir) / member.name
                                txt_path = Path(self.temp_extract_dir) / txt_member.name
                                
                                if img_path.exists() and txt_path.exists():
                                    extracted_samples.append({
                                        'img_path': img_path,
                                        'txt_path': txt_path,
                                        'original_name': base_name
                                    })
                                    sample_count += 1
                                    
                            except Exception as e:
                                print(f"解压文件 {member.name} 时出错: {e}")
                                continue
                
        except Exception as e:
            print(f"处理 tar 文件 {tar_file} 时出错: {e}")
        
        return extracted_samples
    
    def process_extracted_samples(self, samples, start_index):
        """处理解压出的样本，保存到目标目录"""
        success_count = 0
        
        for i, sample in enumerate(samples):
            try:
                # 读取原始 txt 文件（验证集直接是文本）
                with open(sample['txt_path'], 'r', encoding='utf-8') as f:
                    caption = f.read().strip()
                
                # 检查caption是否有效（非空）
                if not caption:
                    continue
                
                # 生成新的文件编号（8位数字）
                new_file_id = f"{start_index + success_count:08d}"
                
                # 目标文件路径
                target_img = self.images_dir / f"{new_file_id}.jpg"
                target_txt = self.texts_dir / f"{new_file_id}.txt"
                
                # 检查目标文件是否已存在
                if target_img.exists() or target_txt.exists():
                    continue
                
                # 复制图片文件
                shutil.copy2(sample['img_path'], target_img)
                
                # 创建文本文件，保存caption
                with open(target_txt, 'w', encoding='utf-8') as f:
                    f.write(caption)
                
                # 记录已处理的文件
                self.processed_files.add(sample['original_name'])
                success_count += 1
                
                if (start_index + success_count) % 50 == 0:
                    print(f"已处理 {start_index + success_count} 个文件")
                
            except Exception as e:
                print(f"处理样本 {sample['original_name']} 时出错: {e}")
                continue
        
        return success_count
    
    def cleanup_temp_dir(self):
        """清理临时目录"""
        if self.temp_extract_dir and Path(self.temp_extract_dir).exists():
            shutil.rmtree(self.temp_extract_dir)
            print(f"已清理临时目录: {self.temp_extract_dir}")
    
    def process(self):
        """主处理流程"""
        print("=" * 50)
        print("CC3M 验证集数据提取器")
        print("=" * 50)
        
        self.load_existing_files()
        current_count = self.get_existing_count()
        needed_count = self.target_count - current_count
        
        print(f"当前文件数: {current_count}")
        print(f"目标文件数: {self.target_count}")
        print(f"需要提取: {needed_count}")
        
        if needed_count <= 0:
            print("✅ 已达到目标文件数量")
            return
        
        tar_files = self.find_validation_tar_files()
        if not tar_files:
            print("❌ 未找到验证集 tar 文件")
            return
        
        try:
            processed_samples = 0
            
            for i, tar_file in enumerate(tar_files):
                if processed_samples >= needed_count:
                    break
                
                print(f"\n处理第 {i+1}/{len(tar_files)} 个文件: {tar_file.name}")
                
                remaining_needed = needed_count - processed_samples
                samples_to_extract = min(remaining_needed, 500)  # 每次最多提取500个
                
                extracted_samples = self.extract_samples_from_tar(tar_file, samples_to_extract)
                
                if extracted_samples:
                    print(f"提取了 {len(extracted_samples)} 个样本")
                    success_count = self.process_extracted_samples(
                        extracted_samples, 
                        current_count + processed_samples
                    )
                    processed_samples += success_count
                    print(f"✅ 成功处理 {success_count} 个样本，累计: {current_count + processed_samples}")
                else:
                    print("⚠️ 未提取到有效样本")
                
                if current_count + processed_samples >= self.target_count:
                    print(f"✅ 已达到目标: {current_count + processed_samples}")
                    break
            
            final_count = self.get_existing_count()
            print(f"\n处理完成！最终数量: {final_count}")
            
        except KeyboardInterrupt:
            print("\n⚠️ 用户中断处理")
        except Exception as e:
            print(f"❌ 处理出错: {e}")
        finally:
            self.cleanup_temp_dir()

def main():
    
    if len(sys.argv) > 1 and (sys.argv[1] == "--help" or sys.argv[1] == "-h"):
        print("用法:")
        print("  python3 cc3m_validation_extract.py [数量]     # 提取指定数量的验证集")
        print("  python3 cc3m_validation_extract.py --help     # 显示帮助")
        print("\n示例:")
        print("  python3 cc3m_validation_extract.py 500       # 提取500个文件")
        return
    
    # 获取目标数量
    target_count = 1000  # 默认提取1000个
    if len(sys.argv) > 1:
        try:
            target_count = int(sys.argv[1])
        except ValueError:
            print(f"❌ 错误：'{sys.argv[1]}' 不是有效的数字")
            sys.exit(1)
    
    extractor = CC3MValidationExtractor(target_count=target_count)
    extractor.process()

if __name__ == "__main__":
    main()
